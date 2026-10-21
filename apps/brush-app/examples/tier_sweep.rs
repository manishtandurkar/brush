//! Headless detail-tier sweep for resource-pressure-aware rendering.
//!
//! Renders an (importance-ordered) scene from the views in a 3DGS `cameras.json`
//! at several detail tiers, where tier `f` renders the first `ceil(f * N)` splats.
//! Writes one PNG per (tier, view) and a CSV with render time, visible splat and
//! intersection counts, and memory per tier. Image quality (PSNR/SSIM/LPIPS against
//! the full-detail render) is computed afterwards by `tools/tier_quality.py`.
//!
//! Modes:
//! - `slice`: the full scene stays resident on the GPU and each tier renders a
//!   slice of it. This is what the live renderer does today.
//! - `resident`: only the first N splats are uploaded for each tier, i.e. what
//!   evicting the rest of the scene would save.
//!
//! Example:
//! ```text
//! cargo run --release -p brush-app --example tier_sweep -- \
//!     --ply ../../scenes/bonsai/point_cloud_ordered.ply \
//!     --cameras ../../scenes/bonsai/cameras.json --out ../../logs/tiers_bonsai
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use brush_render::TextureMode;
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, render_splats};
use brush_render::kernels::camera_model::CameraModel;
use brush_serde::SplatData;
use clap::{Parser, ValueEnum};
use glam::{Mat3, Quat, UVec2, Vec2, Vec3};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Slice,
    Resident,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    ply: PathBuf,
    #[arg(long)]
    cameras: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Fractions of the scene's splats to render, highest first. 1.0 is the reference.
    #[arg(long, value_delimiter = ',', default_value = "1.0,0.75,0.5,0.35,0.25,0.15,0.1")]
    tiers: Vec<f64>,
    /// Image scale relative to the camera's native resolution.
    #[arg(long, default_value_t = 0.5)]
    scale: f64,
    /// Use every k-th camera.
    #[arg(long, default_value_t = 8)]
    every: usize,
    /// Timed renders per view (after one warm-up render).
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, value_enum, default_value_t = Mode::Slice)]
    mode: Mode,
    /// Skip writing PNGs (timing and memory only).
    #[arg(long)]
    no_images: bool,
}

#[derive(serde::Deserialize)]
struct JsonCamera {
    img_name: String,
    width: u32,
    height: u32,
    position: [f32; 3],
    rotation: [[f32; 3]; 3],
    fx: f64,
    fy: f64,
}

struct View {
    name: String,
    camera: Camera,
    size: UVec2,
}

fn load_views(args: &Args) -> anyhow::Result<Vec<View>> {
    let cams: Vec<JsonCamera> = serde_json::from_str(&std::fs::read_to_string(&args.cameras)?)?;
    Ok(cams
        .into_iter()
        .step_by(args.every.max(1))
        .map(|c| {
            let rot = Mat3::from_cols_array_2d(&c.rotation).transpose();
            let model = CameraModel::default();
            let camera = Camera::new(
                Vec3::from(c.position),
                Quat::from_mat3(&rot).normalize(),
                focal_to_fov(c.fx, c.width, &model),
                focal_to_fov(c.fy, c.height, &model),
                Vec2::splat(0.5),
                model,
            );
            let size = UVec2::new(
                (f64::from(c.width) * args.scale).round() as u32,
                (f64::from(c.height) * args.scale).round() as u32,
            );
            View {
                name: c.img_name,
                camera,
                size,
            }
        })
        .collect())
}

fn truncate_data(data: &SplatData, n: usize) -> SplatData {
    let total = data.num_splats();
    let cut = |v: &Vec<f32>| v[..v.len() / total * n].to_vec();
    SplatData {
        means: cut(&data.means),
        rotations: data.rotations.as_ref().map(cut),
        log_scales: data.log_scales.as_ref().map(cut),
        sh_coeffs: data.sh_coeffs.as_ref().map(cut),
        raw_opacities: data.raw_opacities.as_ref().map(cut),
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

async fn run(args: Args) -> anyhow::Result<()> {
    let device = brush_process::burn_init_setup().await;
    let monitor = pressure_monitor::Monitor::start(Duration::from_millis(5), None)?;
    let metal_alloc = || {
        std::thread::sleep(Duration::from_millis(20));
        monitor.handle().latest().os.gpu_alloc_bytes
    };

    let views = load_views(&args)?;
    let bytes = std::fs::read(&args.ply)?;
    let message = brush_serde::load_splat_from_ply(std::io::Cursor::new(bytes), None).await?;
    let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
    let data = message.data;
    let total = data.num_splats();
    println!("{total} splats, {} views, mode {}", views.len(), mode_name(args.mode));

    std::fs::create_dir_all(&args.out)?;
    let csv_path = args.out.join(format!("tiers_{}.csv", mode_name(args.mode)));
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "mode,tier,num_splats,view,width,height,render_ms_median,render_ms_min,\
         num_visible,num_intersections,burn_bytes_in_use,burn_bytes_reserved,metal_alloc_bytes"
    )?;

    let full = (args.mode == Mode::Slice).then(|| data.clone().into_splats(&device, mode));

    for &tier in &args.tiers {
        let n = ((tier * total as f64).ceil() as usize).clamp(1, total);
        device.memory_cleanup();
        let splats: Splats = match &full {
            Some(full) => full.clone().first_n(n as u32),
            None => truncate_data(&data, n).into_splats(&device, mode),
        };
        let tier_dir = args.out.join(format!("tier_{tier:.2}"));
        if !args.no_images {
            std::fs::create_dir_all(&tier_dir)?;
        }

        let mut tier_ms = Vec::new();
        for view in &views {
            let mut times = Vec::with_capacity(args.repeats);
            let mut last = None;
            for i in 0..=args.repeats {
                let start = Instant::now();
                let (image, aux) = render_splats(
                    splats.clone(),
                    &view.camera,
                    view.size,
                    Vec3::ZERO,
                    None,
                    TextureMode::Packed,
                )
                .await;
                let pixels = image.into_data_async().await?.into_bytes().to_vec();
                if i > 0 {
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                last = Some((pixels, aux.num_visible, aux.num_intersections));
            }
            let (pixels, num_visible, num_intersections) = last.expect("at least one render");
            let mem = device.memory_pool_usage();
            let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
            let med_ms = median(&mut times);
            tier_ms.push(med_ms);
            writeln!(
                csv,
                "{},{tier},{n},{},{},{},{med_ms:.3},{min_ms:.3},{num_visible},{num_intersections},{},{},{}",
                mode_name(args.mode),
                view.name,
                view.size.x,
                view.size.y,
                mem.map_or(String::new(), |m| m.bytes_in_use.to_string()),
                mem.map_or(String::new(), |m| m.bytes_reserved.to_string()),
                metal_alloc().map_or(String::new(), |b| b.to_string()),
            )?;
            if !args.no_images {
                let img = image::RgbaImage::from_raw(view.size.x, view.size.y, pixels)
                    .ok_or_else(|| anyhow::anyhow!("unexpected image size for {}", view.name))?;
                image::DynamicImage::ImageRgba8(img)
                    .to_rgb8()
                    .save(tier_dir.join(format!("{}.png", view.name)))?;
            }
        }
        csv.flush()?;
        println!(
            "tier {tier:.2}: {n:>8} splats, median render {:.2} ms",
            median(&mut tier_ms)
        );
    }
    println!("wrote {}", csv_path.display());
    Ok(())
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Slice => "slice",
        Mode::Resident => "resident",
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(args))
}
