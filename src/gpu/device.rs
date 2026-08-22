//! Adapter selection, device creation and bind-group boilerplate.

use crate::config::Config;
use anyhow::{bail, Result};

fn parse_backends(name: &str) -> Option<wgpu::Backends> {
    match name.to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => Some(wgpu::Backends::VULKAN),
        "dx12" | "d3d12" => Some(wgpu::Backends::DX12),
        "metal" => Some(wgpu::Backends::METAL),
        "gl" | "opengl" => Some(wgpu::Backends::GL),
        "all" => Some(wgpu::Backends::all()),
        _ => None,
    }
}

/// Pick an adapter and open a device.
///
/// Never silently falls back to a software adapter: if the only thing wgpu can
/// find is a CPU device, that is an error, because a render that quietly takes
/// a thousand times longer is worse than one that stops.
pub fn create(
    cfg: &Config,
) -> Result<(wgpu::Device, wgpu::Queue, String, wgpu::Limits, bool)> {
    let backends = match &cfg.gpu_backend {
        Some(b) => parse_backends(b)
            .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        None => wgpu::Backends::all(),
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let mut candidates: Vec<wgpu::Adapter> = instance
        .enumerate_adapters(backends)
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect();

    if let Some(want) = &cfg.gpu_adapter {
        let want = want.to_ascii_lowercase();
        candidates.retain(|a| a.get_info().name.to_ascii_lowercase().contains(&want));
        if candidates.is_empty() {
            bail!("no gpu adapter matched {want:?}");
        }
    }

    // Discrete first, then integrated. Within a tier, prefer Vulkan over DX12
    // over GL: the compute paths are better tested there.
    let rank = |a: &wgpu::Adapter| -> (u8, u8) {
        let i = a.get_info();
        let t = match i.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            _ => 3,
        };
        let b = match i.backend {
            wgpu::Backend::Vulkan => 0,
            wgpu::Backend::Metal => 0,
            wgpu::Backend::Dx12 => 1,
            _ => 2,
        };
        (t, b)
    };
    candidates.sort_by_key(rank);

    let adapter = candidates.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "no usable gpu found. wgpu saw no non-software adapter; \
             install or enable a graphics driver, or render with --backend cpu"
        )
    })?;

    let info = adapter.get_info();
    let name = format!("{} ({:?})", info.name, info.backend);
    let adapter_limits = adapter.limits();

    let has_timestamps = adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut features = wgpu::Features::empty();
    if has_timestamps && cfg.profile {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kestrel"),
        required_features: features,
        required_limits: adapter_limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))?;

    // A device error that arrives asynchronously would otherwise show up as
    // silently wrong audio.
    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        log::error!("wgpu device error: {e}");
        panic!("wgpu device error: {e}");
    }));

    Ok((device, queue, name, adapter_limits, has_timestamps))
}

/// Compute-only bind group layout. `read_only[i]` says whether binding i is a
/// read-only storage buffer; binding 0 is always the uniform block.
pub fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    read_only: &[bool],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if i == 0 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: ro },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        })
        .collect();

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

/// List every adapter wgpu can reach, for working out which device a render
/// will land on.
pub fn print_adapters() -> Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = a.get_info();
        let l = a.limits();
        println!(
            "{:?}  {}  [{:?}]  driver {} {}",
            info.backend, info.name, info.device_type, info.driver, info.driver_info
        );
        println!(
            "    storage buffer max {} MiB | workgroup storage {} B | \
             {} invocations/wg | {} workgroups/dim",
            l.max_storage_buffer_binding_size / 1048576,
            l.max_compute_workgroup_storage_size,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroups_per_dimension
        );
        // The number people actually want off this listing. `gpu-info` used to
        // print the binding size and leave the division to the reader.
        let steal = Config::default().max_steal_percent;
        let binding = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
        println!(
            "    max --max-voices {} at --steal-percent {} ({} pool slots)",
            crate::gpu::max_voices_for_binding(binding, steal),
            steal,
            binding / 96,
        );
        println!(
            "    subgroups {} | int64 {} | timestamps {}",
            a.features().contains(wgpu::Features::SUBGROUP),
            a.features().contains(wgpu::Features::SHADER_INT64),
            a.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        );
    }
    Ok(())
}
