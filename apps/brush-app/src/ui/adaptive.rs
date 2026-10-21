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
//! Configured by `ADAPTIVE_MODE`:
//! - `policy`: the resource-pressure-aware policy (the contribution).
//! - `off` (default): fixed full detail, Brush unchanged.
//! - `fixed:<fraction>`: a fixed fraction of the splats.
//!
//! Comparison baselines, all using the same tiers and eviction path:
//! - `device-tier`: one tier picked at load from the device's GPU memory budget
//!   (FLoD-style static per-device level).
//! - `distance`: detail from camera distance to the scene (Octree-GS/LODGE-style LOD).
//! - `budget:<MB>`: a fixed splat memory budget (MoQSplat-style fixed `M_max`).
//! - `thermal-step`: step down on a thermal warning, back up when nominal
//!   (Unity Adaptive Performance-style).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, mpsc};

use adaptive_policy::{Config, Decision, Policy, Reading};
use brush_process::slot::Slot;
use brush_render::gaussian_splats::Splats;
use burn::module::{Param, ParamId};
use burn::tensor::{Device, Tensor, TensorData, s};
use glam::Vec3;

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
    DeviceTier,
    Distance,
    Budget(f64),
    ThermalStep,
    Policy,
}

impl Mode {
    fn from_env() -> Self {
        let value = std::env::var("ADAPTIVE_MODE").unwrap_or_default();
        match value.as_str() {
            "" | "off" => Self::Off,
            "policy" => Self::Policy,
            "device-tier" => Self::DeviceTier,
            "distance" => Self::Distance,
            "thermal-step" => Self::ThermalStep,
            v => {
                let fixed = v.strip_prefix("fixed:").and_then(|f| f.parse::<f64>().ok());
                let budget = v.strip_prefix("budget:").and_then(|f| f.parse::<f64>().ok());
                match (fixed, budget) {
                    (Some(f), _) if f > 0.0 && f <= 1.0 => Self::Fixed(f),
                    (_, Some(mb)) if mb > 0.0 => Self::Budget(mb),
                    _ => {
                        log::warn!(
                            "ignoring ADAPTIVE_MODE={v}: expected policy, off, fixed:<0..1>, \
                             device-tier, distance, budget:<MB> or thermal-step"
                        );
                        Self::Off
                    }
                }
            }
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

    fn info(&self) -> SceneInfo {
        let n = self.total().max(1);
        let sum = self
            .transforms
            .as_chunks::<10>()
            .0
            .iter()
            .fold(Vec3::ZERO, |acc, t| acc + Vec3::new(t[0], t[1], t[2]));
        SceneInfo {
            center: sum / n as f32,
            bytes_per_splat: 4 * (10 + self.sh_per_splat + 1),
        }
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

#[derive(Clone, Copy, Debug)]
struct SceneInfo {
    center: Vec3,
    bytes_per_splat: usize,
}

enum Job {
    Load(Slot<Splats>),
    Apply(usize),
}

fn worker(
    jobs: &mpsc::Receiver<Job>,
    total: &AtomicU32,
    info: &std::sync::Mutex<Option<SceneInfo>>,
    updated: &AtomicBool,
) {
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
                            *info.lock().unwrap() = Some(cpu.info());
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

fn reading(sample: &pressure_monitor::Sample) -> Reading {
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
        frame_ms: sample.render_ms,
    }
}

/// FLoD-style static level: one fraction per device class, from the GPU memory budget.
fn device_tier_fraction(budget_bytes: Option<u64>) -> f64 {
    let gib = budget_bytes.map_or(0.0, |b| b as f64 / f64::from(1u32 << 30));
    match gib {
        g if g >= 8.0 => 1.0,
        g if g >= 4.0 => 0.5,
        g if g >= 2.0 => 0.25,
        _ => 0.1,
    }
}

/// Largest configured tier fraction not above `raw`, or the smallest tier.
fn snap_to_tier(tiers: &[f64], raw: f64) -> f64 {
    tiers
        .iter()
        .copied()
        .find(|f| *f <= raw)
        .unwrap_or_else(|| tiers.last().copied().unwrap_or(1.0))
}

/// Unity Adaptive Performance-style step-down: thermal warning (serious or worse)
/// lowers one tier at a time; nominal for long enough raises one tier. No other
/// signal is used.
#[derive(Default)]
struct ThermalStep {
    tier: usize,
    hot: u32,
    cool: u32,
    since_change: u32,
}

impl ThermalStep {
    const HOT_SAMPLES: u32 = 3;
    const COOL_SAMPLES: u32 = 100;
    const MIN_GAP: u32 = 50;

    fn step(&mut self, thermal: Option<u32>, num_tiers: usize) -> bool {
        self.since_change += 1;
        let t = thermal.unwrap_or(0);
        self.hot = if t >= 2 { self.hot + 1 } else { 0 };
        self.cool = if t == 0 { self.cool + 1 } else { 0 };
        let before = self.tier;
        if self.hot >= Self::HOT_SAMPLES && self.since_change >= Self::MIN_GAP {
            self.tier = (self.tier + 1).min(num_tiers - 1);
        } else if self.cool >= Self::COOL_SAMPLES && self.since_change >= Self::MIN_GAP {
            self.tier = self.tier.saturating_sub(1);
        }
        if self.tier != before {
            self.since_change = 0;
            self.hot = 0;
            self.cool = 0;
        }
        self.tier != before
    }
}

pub struct AdaptiveDetail {
    mode: Mode,
    tiers: Vec<f64>,
    policy: Option<Policy>,
    jobs: Option<mpsc::Sender<Job>>,
    total: Arc<AtomicU32>,
    info: Arc<std::sync::Mutex<Option<SceneInfo>>>,
    updated: Arc<AtomicBool>,
    target: Option<usize>,
    last_sample_ms: f64,
    decision: Option<Decision>,
    device_fraction: Option<f64>,
    camera_position: Option<Vec3>,
    reference_distance: Option<f32>,
    thermal_step: ThermalStep,
    fraction: f64,
}

impl AdaptiveDetail {
    pub fn from_env() -> Self {
        let mode = Mode::from_env();
        let total = Arc::new(AtomicU32::new(0));
        let info = Arc::new(std::sync::Mutex::new(None));
        let updated = Arc::new(AtomicBool::new(false));
        let jobs = (mode != Mode::Off).then(|| {
            let (tx, rx) = mpsc::channel();
            let (total, info, updated) = (total.clone(), info.clone(), updated.clone());
            std::thread::Builder::new()
                .name("adaptive-detail".into())
                .spawn(move || worker(&rx, &total, &info, &updated))
                .expect("failed to spawn adaptive-detail thread");
            tx
        });
        let config = Config::default();
        if mode != Mode::Off {
            log::info!("adaptive: mode {mode:?}");
        }
        Self {
            mode,
            tiers: config.tiers().to_vec(),
            policy: (mode == Mode::Policy).then(|| Policy::new(config)).flatten(),
            jobs,
            total,
            info,
            updated,
            target: None,
            last_sample_ms: -1.0,
            decision: None,
            device_fraction: None,
            camera_position: None,
            reference_distance: None,
            thermal_step: ThermalStep::default(),
            fraction: 1.0,
        }
    }

    /// Camera position in splat space, used by the `distance` baseline.
    pub fn set_camera_position(&mut self, position: Vec3) {
        self.camera_position = Some(position);
    }

    /// Lines for the on-screen overlay.
    pub fn status_lines(&self) -> Vec<String> {
        let total = self.total.load(Ordering::Acquire) as usize;
        let n = self.target.unwrap_or(total);
        let mut lines = vec![
            format!("mode      {}", mode_label(self.mode)),
            format!(
                "detail    {:.0}%  {n} / {total} splats",
                self.fraction * 100.0
            ),
        ];
        if let Some(d) = &self.decision {
            lines.push(format!(
                "tier      {}  pressure {}",
                d.tier,
                adaptive_policy::reason_label(d.pressure_reasons)
            ));
        }
        if let Some(monitor) = pressure_monitor::global() {
            let s = monitor.latest();
            let mb = |b: u64| b as f64 / f64::from(1u32 << 20);
            if let (Some(alloc), Some(budget)) = (s.os.gpu_alloc_bytes, s.os.gpu_budget_bytes) {
                lines.push(format!("gpu mem   {:.0} / {:.0} MB", mb(alloc), mb(budget)));
            }
            let opt = |v: Option<u32>| v.map_or("-".into(), |v| v.to_string());
            lines.push(format!(
                "sys mem   {}% free  pressure {}",
                opt(s.os.sys_mem_free_pct),
                opt(s.os.sys_mem_pressure)
            ));
            let thermal = ["nominal", "fair", "serious", "critical"];
            lines.push(format!(
                "thermal   {}",
                s.os.thermal_state
                    .and_then(|t| thermal.get(t as usize))
                    .unwrap_or(&"-")
            ));
            let ms = |v: Option<f64>| v.map_or("-".into(), |v| format!("{v:.1} ms"));
            lines.push(format!(
                "render    {}   frame {}",
                ms(s.render_ms),
                ms(s.ui_frame_ms)
            ));
        }
        lines
    }

    pub fn is_active(&self) -> bool {
        self.mode != Mode::Off
    }

    /// A new scene is loading: forget the old CPU copy until loading finishes.
    pub fn on_scene_changed(&mut self) {
        self.total.store(0, Ordering::Release);
        self.target = None;
        self.reference_distance = None;
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

        let monitor = pressure_monitor::global();
        let new_sample = monitor.map(|m| m.latest()).filter(|s| s.t_ms > self.last_sample_ms);
        if let Some(s) = &new_sample {
            self.last_sample_ms = s.t_ms;
        }
        let info = *self.info.lock().unwrap();

        let fraction = match self.mode {
            Mode::Off => return updated,
            Mode::Fixed(f) => f,
            Mode::DeviceTier => match (self.device_fraction, &new_sample) {
                (Some(f), _) => f,
                (None, Some(s)) => {
                    let f = device_tier_fraction(s.os.gpu_budget_bytes);
                    log::info!("adaptive: device tier {:.0}%", f * 100.0);
                    *self.device_fraction.insert(f)
                }
                (None, None) => return updated,
            },
            Mode::Budget(mb) => {
                let bytes_per_splat = info.map_or(1, |i| i.bytes_per_splat.max(1));
                let fit = mb * f64::from(1u32 << 20) / bytes_per_splat as f64;
                (fit / total as f64).min(1.0)
            }
            Mode::Distance => {
                let (Some(cam), Some(info)) = (self.camera_position, info) else {
                    return updated;
                };
                let d = (cam - info.center).length().max(1e-3);
                let d0 = *self.reference_distance.get_or_insert(d);
                snap_to_tier(&self.tiers, f64::from(d0 / d))
            }
            Mode::ThermalStep => {
                if let Some(s) = &new_sample
                    && self.thermal_step.step(s.os.thermal_state, self.tiers.len())
                {
                    log::info!("adaptive: thermal step to tier {}", self.thermal_step.tier);
                }
                if let Some(m) = monitor {
                    m.set_detail(self.thermal_step.tier as u32, 0);
                }
                self.tiers[self.thermal_step.tier]
            }
            Mode::Policy => {
                if let (Some(policy), Some(monitor), Some(sample)) =
                    (self.policy.as_mut(), monitor, &new_sample)
                {
                    let d = policy.step(&reading(sample));
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
                self.decision.map_or(1.0, |d| d.fraction)
            }
        };

        self.fraction = fraction;
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

fn mode_label(mode: Mode) -> String {
    match mode {
        Mode::Off => "off (full detail)".into(),
        Mode::Fixed(f) => format!("fixed {:.0}%", f * 100.0),
        Mode::DeviceTier => "device tier".into(),
        Mode::Distance => "distance LOD".into(),
        Mode::Budget(mb) => format!("budget {mb:.0} MB"),
        Mode::ThermalStep => "thermal step-down".into(),
        Mode::Policy => "resource-pressure policy".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_tier_maps_budgets() {
        assert_eq!(device_tier_fraction(Some(12 << 30)), 1.0);
        assert_eq!(device_tier_fraction(Some(6 << 30)), 0.5);
        assert_eq!(device_tier_fraction(Some(3 << 30)), 0.25);
        assert_eq!(device_tier_fraction(None), 0.1);
    }

    #[test]
    fn snap_picks_largest_tier_not_above() {
        let tiers = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1];
        assert_eq!(snap_to_tier(&tiers, 2.0), 1.0);
        assert_eq!(snap_to_tier(&tiers, 0.6), 0.5);
        assert_eq!(snap_to_tier(&tiers, 0.01), 0.1);
    }

    #[test]
    fn thermal_step_reacts_only_to_warning_levels() {
        let mut t = ThermalStep::default();
        for _ in 0..200 {
            t.step(Some(1), 7);
        }
        assert_eq!(t.tier, 0);
        for _ in 0..30 {
            t.step(Some(2), 7);
        }
        assert_eq!(t.tier, 1);
        for _ in 0..200 {
            t.step(Some(0), 7);
        }
        assert_eq!(t.tier, 0);
    }
}
