//! Prints every GPU adapter `render` can open: backend, device type, driver, the optional
//! features the spikes use, and the limits that size compute work.
//!
//! `cargo run -p render --example adapters`. The output goes into the GPU spike report, one
//! block per machine.

use std::io::{self, Write as _};

use wgpu::Features;

fn main() -> io::Result<()> {
    let mut out = io::stdout().lock();
    let adapters = render::gpu::adapters();
    if adapters.is_empty() {
        writeln!(out, "no adapters")?;
    }
    for adapter in adapters {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();
        writeln!(out, "{}", info.name)?;
        writeln!(out, "  backend:      {:?}", info.backend)?;
        writeln!(out, "  device type:  {:?}", info.device_type)?;
        writeln!(
            out,
            "  vendor:       {:#06X}, device {:#06X}",
            info.vendor, info.device
        )?;
        writeln!(
            out,
            "  driver:       {} ({})",
            info.driver, info.driver_info
        )?;
        writeln!(
            out,
            "  timestamps:   {}",
            if features.contains(Features::TIMESTAMP_QUERY) {
                "yes"
            } else {
                "no"
            }
        )?;
        writeln!(
            out,
            "  shader f16:   {}",
            if features.contains(Features::SHADER_F16) {
                "yes"
            } else {
                "no"
            }
        )?;
        writeln!(
            out,
            "  storage buffer binding: {} MiB, buffer: {} MiB",
            limits.max_storage_buffer_binding_size >> 20,
            limits.max_buffer_size >> 20
        )?;
        writeln!(
            out,
            "  workgroup:    {} invocations, size {} × {} × {}, {} KiB shared",
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroup_storage_size >> 10
        )?;
        writeln!(
            out,
            "  workgroups per dimension: {}, storage buffers per stage: {}",
            limits.max_compute_workgroups_per_dimension,
            limits.max_storage_buffers_per_shader_stage
        )?;
    }
    Ok(())
}
