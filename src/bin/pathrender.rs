//! Render splatcam path JSON directly through web-splat's WGPU renderer.
use anyhow::{bail, Context, Result};
use cgmath::{Matrix3, Point3, Quaternion, Vector2, Vector3};
use clap::Parser;
use serde::Deserialize;
use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use web_splats::{
    io::GenericGaussianPointCloud, GaussianRenderer, PerspectiveCamera, PerspectiveProjection,
    PointCloud, SplattingArgs, WGPUContext,
};

#[derive(Parser)]
#[command(about = "Headless 3DGS camera path renderer; output is H.264 MP4")]
struct Opt {
    input: PathBuf,
    #[arg(long)]
    path: Option<PathBuf>,
    #[arg(short, long)]
    out: PathBuf,
    #[arg(long, default_value = "1920x1080")]
    size: String,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    #[arg(long, default_value_t = 0.0)]
    tail: f32,
    #[arg(long, default_value = "#CDCECF")]
    bg: String,
    #[arg(long, default_value_t = 2.5)]
    duration: f32,
    #[arg(long, default_value_t = 0.55)]
    zoom_from: f32,
    #[arg(long, default_value_t = 1.0)]
    zoom_to: f32,
    #[arg(long, default_value_t = 360.0)]
    spin: f32,
    #[arg(long, default_value_t = 0.0)]
    tilt: f32,
    #[arg(long, default_value_t = 1.0)]
    splat_scale: f32,
    /// First source-attached point index; inferred from input .json when omitted.
    #[arg(long)]
    cap_start: Option<u32>,
    #[arg(long)]
    limit_frames: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default)]
struct Key {
    t: f32,
    zoom: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    pan_x: f32,
    pan_y: f32,
    ease: Option<[f32; 4]>,
}
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default)]
struct Frame {
    scale: f32,
    x: f32,
    y: f32,
}
impl Default for Frame {
    fn default() -> Self {
        Self {
            scale: 1.,
            x: 0.,
            y: 0.,
        }
    }
}
#[derive(Deserialize)]
struct Path {
    ease: Option<[f32; 4]>,
    #[serde(default)]
    frame: Frame,
    keys: Vec<Key>,
}
#[derive(Deserialize)]
struct AssetMetadata {
    mesh_samples: u32,
}

fn bezier(c: [f32; 4], x: f32) -> f32 {
    let f = |s: f32, a: f32, b: f32| {
        3. * (1. - s) * (1. - s) * s * a + 3. * (1. - s) * s * s * b + s * s * s
    };
    let (mut lo, mut hi) = (0., 1.);
    for _ in 0..32 {
        let s = (lo + hi) * 0.5;
        if f(s, c[0], c[2]) < x.clamp(0., 1.) {
            lo = s
        } else {
            hi = s
        }
    }
    f((lo + hi) * 0.5, c[1], c[3])
}
fn sample(keys: &[Key], ease: [f32; 4], t: f32) -> Key {
    if t <= keys[0].t {
        return keys[0];
    }
    if t >= keys.last().unwrap().t {
        return *keys.last().unwrap();
    }
    let i = keys.windows(2).position(|k| t < k[1].t).unwrap();
    let (a, b) = (keys[i], keys[i + 1]);
    let e = bezier(b.ease.unwrap_or(ease), (t - a.t) / (b.t - a.t).max(1e-6));
    let l = |x: f32, y: f32| x + (y - x) * e;
    Key {
        t,
        zoom: l(a.zoom, b.zoom),
        yaw: l(a.yaw, b.yaw),
        pitch: l(a.pitch, b.pitch),
        roll: l(a.roll, b.roll),
        pan_x: l(a.pan_x, b.pan_x),
        pan_y: l(a.pan_y, b.pan_y),
        ease: None,
    }
}
fn background(hex: &str) -> Result<wgpu::Color> {
    let s = hex.strip_prefix('#').unwrap_or(hex);
    if s.len() != 6 {
        bail!("--bg must be #RRGGBB")
    }
    let n = u32::from_str_radix(s, 16)?;
    Ok(wgpu::Color {
        r: ((n >> 16) & 255) as f64 / 255.,
        g: ((n >> 8) & 255) as f64 / 255.,
        b: (n & 255) as f64 / 255.,
        a: 1.,
    })
}
fn camera(st: Key, frame: Frame, center: Point3<f32>, w: u32, h: u32) -> PerspectiveCamera {
    // Match splatcam: y is down, +z is away, yaw/pitch orbit around the point-cloud centroid.
    let (sy, cy) = st.yaw.to_radians().sin_cos();
    let (sp, cp) = st.pitch.to_radians().sin_cos();
    let (sr, cr) = st.roll.to_radians().sin_cos();
    let x = Vector3::new(cy, 0., sy);
    let y = Vector3::new(sp * sy, cp, -sp * cy);
    let z = Vector3::new(-cp * sy, sp, cp * cy);
    let xr = x * cr + y * sr;
    let yr = y * cr - x * sr;
    // web-splat's `world2view` expects world-to-camera rows, so transpose the
    // camera basis before converting it to the stored quaternion.
    let world_to_camera = Matrix3::from_cols(
        Vector3::new(xr.x, yr.x, z.x),
        Vector3::new(xr.y, yr.y, z.y),
        Vector3::new(xr.z, yr.z, z.z),
    );
    let rotation = Quaternion::from(world_to_camera);
    let dist = 3. * 576.;
    let zoom = (st.zoom * frame.scale).max(0.01);
    // Native splatcam placement keeps the source centroid's screen offset.
    let offset_x = ((st.pan_x + frame.x) * 576. + center.x) / zoom;
    let offset_y = ((st.pan_y + frame.y) * 576. + center.y) / zoom;
    let position = center - z * dist - xr * offset_x - yr * offset_y;
    let fy = 2. * (576. / (2. * dist * zoom)).atan();
    let fx = 2. * ((w as f32 / h as f32) * (fy * 0.5).tan()).atan();
    PerspectiveCamera::new(
        position,
        rotation,
        PerspectiveProjection::new(
            Vector2::new(w, h),
            Vector2::new(cgmath::Rad(fx), cgmath::Rad(fy)),
            0.1,
            dist * 3.,
        ),
    )
}

#[pollster::main]
async fn main() -> Result<()> {
    let opt = Opt::parse();
    let (w, h) = opt.size.split_once('x').context("--size must be WxH")?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    let (keys, ease, frame) = if let Some(p) = &opt.path {
        let p: Path = serde_json::from_slice(&std::fs::read(p)?)?;
        if p.keys.len() < 2 {
            bail!("path requires at least two keys")
        }
        (p.keys, p.ease.unwrap_or([0., 1., 0., 1.]), p.frame)
    } else {
        (
            vec![
                Key {
                    t: 0.,
                    zoom: opt.zoom_from,
                    pitch: opt.tilt,
                    ..Default::default()
                },
                Key {
                    t: opt.duration,
                    zoom: opt.zoom_to,
                    yaw: opt.spin,
                    pitch: opt.tilt,
                    ..Default::default()
                },
            ],
            [0., 1., 0., 1.],
            Frame::default(),
        )
    };
    let bg = background(&opt.bg)?;
    let bg8 = [
        (bg.r * 255.) as u16,
        (bg.g * 255.) as u16,
        (bg.b * 255.) as u16,
    ];
    let total = ((keys.last().unwrap().t + opt.tail) * opt.fps as f32).round() as u32 + 1;
    let total = opt.limit_frames.map_or(total, |n| n.min(total));
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let context = WGPUContext::new(&instance, None).await;
    eprintln!("adapter: {}", context.adapter.get_info().name);
    let device = &context.device;
    let queue = &context.queue;
    let raw = GenericGaussianPointCloud::load(File::open(&opt.input)?)?;
    let pc = PointCloud::new(device, raw)?;
    let cap_start = opt.cap_start.or_else(|| {
        std::fs::read(opt.input.with_extension("json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<AssetMetadata>(&bytes).ok())
            .map(|meta| meta.mesh_samples)
    });
    let center = pc.center();
    eprintln!(
        "{} splats, centroid {:?}, {}x{}, {} frames",
        pc.num_points(),
        center,
        w,
        h,
        total
    );
    let mut renderer = GaussianRenderer::new(
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pathrender frame"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let pitch = ((w * 4 + 255) / 256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: pitch as u64 * h as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut ff = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &opt.size,
            "-r",
            &opt.fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&opt.out)
        .stdin(Stdio::piped())
        .spawn()
        .context("launch ffmpeg")?;
    let mut pipe = ff.stdin.take().unwrap();
    let start = Instant::now();
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for i in 0..total {
        let st = sample(&keys, ease, keys[0].t + i as f32 / opt.fps as f32);
        let angle = st.yaw.rem_euclid(360.);
        let angle = angle.min(360. - angle);
        let cap_alpha = ((35. - angle) / 23.).clamp(0., 1.);
        let mut cam = camera(st, frame, center, w, h);
        cam.fit_near_far(pc.bbox());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("path frame"),
        });
        renderer.prepare(
            &mut encoder,
            device,
            queue,
            &pc,
            SplattingArgs {
                camera: cam,
                viewport: Vector2::new(w, h),
                gaussian_scaling: opt.splat_scale,
                max_sh_deg: 0,
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::from_secs(100),
                scene_center: None,
                scene_extend: None,
                background_color: bg,
                cap_start,
                cap_alpha,
            },
            &mut None,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path splats"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.render(&mut pass, &pc);
        }
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(h),
                },
            },
            texture.size(),
        );
        let idx = queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })?;
        rx.receive().await.context("readback channel")??;
        {
            let data = slice.get_mapped_range();
            for row in 0..h as usize {
                for col in 0..w as usize {
                    let a = row * pitch as usize + col * 4;
                    let b = (row * w as usize + col) * 3;
                    let remaining = 255u16 - data[a + 3] as u16;
                    for c in 0..3 {
                        rgb[b + c] = (data[a + c] as u16 + bg8[c] * remaining / 255).min(255) as u8;
                    }
                }
            }
        }
        readback.unmap();
        pipe.write_all(&rgb)?;
        if i % 10 == 0 {
            eprintln!(
                "frame {i}/{total}, {:.0} ms/frame",
                start.elapsed().as_secs_f64() * 1000. / (i + 1) as f64
            )
        }
    }
    drop(pipe);
    let status = ff.wait()?;
    if !status.success() {
        bail!("ffmpeg failed: {status}")
    }
    eprintln!(
        "rendered {total} frames in {:.2}s ({:.1} ms/frame): {}",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() * 1000. / total as f64,
        opt.out.display()
    );
    Ok(())
}
