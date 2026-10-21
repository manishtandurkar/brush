//! Resource-pressure-aware detail control.
//!
//! Picks how many splats to keep on the GPU from measured device state (GPU memory
//! headroom, system memory pressure, thermal state, power state, with frame time as
//! a supporting signal), using the C++ policy in `policy/`. The scene must be
//! importance-ordered (`tools/order_gaussians.py`) so a lower tier is "the first N
//! splats".
//!
//! A worker thread keeps a CPU copy of the loaded scene. On a tier change it uploads
//! only the first N splats and swaps them into the scene slot, so the full scene is
//! freed from the GPU rather than merely skipped at render time.
//!
//! Configured by `ADAPTIVE_MODE`: `policy`, `fixed:<fraction>`, or `off` (default).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, mpsc};

use adaptive_policy::{Config, Decision, Policy, Reading};
use brush_process::slot::Slot;
use brush_render::gaussian_splats::Splats;
use burn::module::{Param, ParamId};
use burn::tensor::{Device, Tensor, TensorData, s};

/// Rows per GPU transfer. Whole-scene transfers leave scene-sized staging buffers
/// allocated on the device, which defeats the point of evicting splats.
const TRANSFER_ROWS: usize = 1 << 16;

/// Time for dropped tensors to actually be released before the allocator is asked
/// to return their pages.
const SETTLE: std::time::Duration = std::time::Duration::from_millis(300);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Off,
    Fixed(f64),
    Policy,
}

impl Mode {
    fn from_env() -> Self {
        let value = std::env::var("ADAPTIVE_MODE").unwrap_or_default();
        match value.as_str() {
            "" | "off" => Self::Off,
            "policy" => Self::Policy,
            v => match v.strip_prefix("fixed:").and_then(|f| f.parse::<f64>().ok()) {
                Some(f) if f > 0.0 && f <= 1.0 => Self::Fixed(f),
                _ => {
                    log::warn!("ignoring ADAPTIVE_MODE={v}: expected policy, fixed:<0..1> or off");
                    Self::Off
                }
            },
        }
    }
}

struct CpuSplats {
    transforms: Vec<f32>,
    sh_coeffs: Vec<f32>,
    sh_per_splat: usize,
    raw_opacities: Vec<f32>,
    render_mip: bool,
    device: Device,
}

impl CpuSplats {
    fn read_back(splats: &Splats) -> Result<Self, String> {
        let [n, sh_rows, sh_cols] = splats.sh_coeffs.dims();
        Ok(Self {
            transforms: read_rows(&splats.transforms.val(), n)?,
            sh_coeffs: read_rows(&splats.sh_coeffs.val(), n)?,
            sh_per_splat: sh_rows * sh_cols,
            raw_opacities: read_rows(&splats.raw_opacities.val(), n)?,
            render_mip: splats.render_mip,
            device: splats.device(),
        })
    }

    fn total(&self) -> usize {
        self.raw_opacities.len()
    }

    fn upload_first(&self, n: usize) -> Splats {
        let n = n.clamp(1, self.total());
        let sh_rows = self.sh_per_splat / 3;
        let transforms = upload_rows(&self.transforms, n, [10], &self.device);
        let sh_coeffs = upload_rows(&self.sh_coeffs, n, [sh_rows, 3], &self.device);
        let raw_opacities = upload_rows(&self.raw_opacities, n, [], &self.device);
        Splats {
            transforms: Param::initialized(ParamId::new(), transforms),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacities),
            render_mip: self.render_mip,
            min_scale: None,
        }
    }
}

fn read_rows<const D: usize>(tensor: &Tensor<D>, n: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for start in (0..n).step_by(TRANSFER_ROWS) {
        let end = (start + TRANSFER_ROWS).min(n);
        let data = tensor.clone().slice(s![start..end]).into_data();
        out.extend(data.try_into_vec::<f32>().map_err(|e| format!("{e:?}"))?);
    }
    Ok(out)
}

/// Upload the first `n` rows of `data`, where each row has shape `row_shape`.
fn upload_rows<const R: usize, const D: usize>(
    data: &[f32],
    n: usize,
    row_shape: [usize; R],
    device: &Device,
) -> Tensor<D> {
    let row_len: usize = row_shape.iter().product();
    let chunks = (0..n)
        .step_by(TRANSFER_ROWS)
        .map(|start| {
            let end = (start + TRANSFER_ROWS).min(n);
            let mut shape = vec![end - start];
            shape.extend(row_shape);
            let rows = data[start * row_len..end * row_len].to_vec();
            Tensor::<D>::from_data(TensorData::new(rows, shape), device)
        })
        .collect();
    Tensor::cat(chunks, 0)
}

enum Job {
    Load(Slot<Splats>),
    Apply(usize),
}

fn worker(jobs: &mpsc::Receiver<Job>, total: &AtomicU32, updated: &AtomicBool) {
    let mut scene: Option<(Slot<Splats>, CpuSplats)> = None;
    while let Ok(first) = jobs.recv() {
        let mut apply = None;
        for job in std::iter::once(first).chain(jobs.try_iter()) {
            match job {
                Job::Load(slot) => {
                    apply = None;
                    total.store(0, Ordering::Release);
                    scene = None;
                    let Some(splats) = slot.get(0) else {
                        continue;
                    };
                    let start = web_time::Instant::now();
                    match CpuSplats::read_back(&splats) {
                        Ok(cpu) => {
                            log::info!(
                                "adaptive: kept CPU copy of {} splats in {:.0} ms",
                                cpu.total(),
                                start.elapsed().as_secs_f64() * 1000.0
                            );
                            total.store(cpu.total() as u32, Ordering::Release);
                            scene = Some((slot, cpu));
                        }
                        Err(e) => log::warn!("adaptive: could not read back splats: {e}"),
                    }
                }
                Job::Apply(n) => apply = Some(n),
            }
        }
        if let (Some(n), Some((slot, cpu))) = (apply, scene.as_ref()) {
            let start = web_time::Instant::now();
            let splats = cpu.upload_first(n);
            if slot.replace(0, splats) {
                updated.store(true, Ordering::Release);
                log::info!(
                    "adaptive: now {n} splats on the GPU ({:.0} ms)",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                std::thread::sleep(SETTLE);
                cpu.device.memory_cleanup();
            }
        }
    }
}

fn reading(sample: &pressure_monitor::Sample, use_frame_time: bool) -> Reading {
    let os = &sample.os;
    let headroom = os
        .gpu_headroom_bytes()
        .zip(os.gpu_budget_bytes.filter(|b| *b > 0))
        .map(|(h, b)| h as f64 / b as f64);
    Reading {
        gpu_headroom_frac: headroom,
        sys_mem_pressure: os.sys_mem_pressure,
        sys_mem_free_pct: os.sys_mem_free_pct,
        thermal_state: os.thermal_state,
        on_battery: os.on_battery,
        battery_pct: os.battery_pct,
        low_power_mode: os.low_power_mode,
        frame_ms: sample.ui_frame_ms.filter(|_| use_frame_time),
    }
}

pub struct AdaptiveDetail {
    mode: Mode,
    policy: Option<Policy>,
    jobs: Option<mpsc::Sender<Job>>,
    total: Arc<AtomicU32>,
    updated: Arc<AtomicBool>,
    target: Option<usize>,
    last_sample_ms: f64,
    decision: Option<Decision>,
    use_frame_time: bool,
}

impl AdaptiveDetail {
    pub fn from_env(continuous: bool) -> Self {
        let mode = Mode::from_env();
        let total = Arc::new(AtomicU32::new(0));
        let updated = Arc::new(AtomicBool::new(false));
        let jobs = (mode != Mode::Off).then(|| {
            let (tx, rx) = mpsc::channel();
            let (total, updated) = (total.clone(), updated.clone());
            std::thread::Builder::new()
                .name("adaptive-detail".into())
                .spawn(move || worker(&rx, &total, &updated))
                .expect("failed to spawn adaptive-detail thread");
            tx
        });
        if mode != Mode::Off {
            log::info!("adaptive: mode {mode:?}");
        }
        Self {
            mode,
            policy: (mode == Mode::Policy).then(|| Policy::new(Config::default())).flatten(),
            jobs,
            total,
            updated,
            target: None,
            last_sample_ms: -1.0,
            decision: None,
            use_frame_time: continuous,
        }
    }

    pub fn is_active(&self) -> bool {
        self.mode != Mode::Off
    }

    /// A new scene is loading: forget the old CPU copy until loading finishes.
    pub fn on_scene_changed(&mut self) {
        self.total.store(0, Ordering::Release);
        self.target = None;
    }

    /// Loading finished: keep a CPU copy of frame 0 so tiers can be re-uploaded.
    pub fn on_scene_loaded(&mut self, slot: &Slot<Splats>, frame_count: u32) {
        self.target = None;
        if frame_count > 1 {
            log::warn!("adaptive: animated scenes are not supported, detail stays fixed");
            return;
        }
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(Job::Load(slot.clone()));
        }
    }

    /// Call once per UI frame. Returns true when the splats on the GPU changed and
    /// the view needs a re-render.
    pub fn tick(&mut self) -> bool {
        if !self.is_active() {
            return false;
        }
        let updated = self.updated.swap(false, Ordering::AcqRel);
        let total = self.total.load(Ordering::Acquire) as usize;
        if total == 0 {
            return updated;
        }

        let fraction = match self.mode {
            Mode::Off => return updated,
            Mode::Fixed(f) => f,
            Mode::Policy => {
                if let (Some(policy), Some(monitor)) =
                    (self.policy.as_mut(), pressure_monitor::global())
                {
                    let sample = monitor.latest();
                    if sample.t_ms > self.last_sample_ms {
                        self.last_sample_ms = sample.t_ms;
                        let d = policy.step(&reading(&sample, self.use_frame_time));
                        monitor.set_detail(d.tier as u32, d.pressure_reasons);
                        if d.changed {
                            log::info!(
                                "adaptive: tier {} ({:.0}%), pressure {}, blocking {}",
                                d.tier,
                                d.fraction * 100.0,
                                adaptive_policy::reason_label(d.pressure_reasons),
                                adaptive_policy::reason_label(d.blocking_reasons),
                            );
                        }
                        self.decision = Some(d);
                    }
                }
                self.decision.map_or(1.0, |d| d.fraction)
            }
        };

        let n = ((fraction * total as f64).ceil() as usize).clamp(1, total);
        if self.target != Some(n) {
            let full_and_untouched = self.target.is_none() && n == total;
            if !full_and_untouched && let Some(jobs) = &self.jobs {
                let _ = jobs.send(Job::Apply(n));
            }
            self.target = Some(n);
        }
        updated
    }
}
