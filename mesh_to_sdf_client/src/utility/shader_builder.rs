use anyhow::Result;

use rust_embed::RustEmbed;

use std::borrow::Cow;

/// Shader helpers
/// Will load from file in native debug mode to allow reloading at runtime
/// and embed in binary in wasm/release mode.
#[derive(RustEmbed)]
#[folder = "shaders"]
pub struct ShaderBuilder;

impl ShaderBuilder {
    /// Load a shader file.
    /// Does not do any pre-processing here, but returns the raw content.
    pub fn load(name: &str) -> Result<String> {
        // read file.
        Self::get(name)
            .ok_or(anyhow::anyhow!("Shader not found: {name}"))
            // Try parsing to utf8.
            .and_then(|file| {
                std::str::from_utf8(file.data.as_ref())
                    .map(|x| x.to_owned())
                    .map_err(|e| anyhow::anyhow!(e))
            })
    }

    /// Build a shader file by importing all its dependencies.
    /// TODO: Add #ifdef #else #endif #ifndef support.
    pub fn build(name: &str) -> Result<String> {
        Self::build_with_seen(name, &mut vec![])
    }

    /// Create a shader module from a shader file.
    pub fn create_module(device: &wgpu::Device, name: &str) -> Result<wgpu::ShaderModule> {
        let shader = ShaderBuilder::build(name)?;

        // device.create_shader_module panics if the shader is malformed
        // only check this on native debug builds.
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader.as_str())),
        });

        // device.create_shader_module panics if the shader is malformed
        // only check this on native debug builds.
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            anyhow::bail!("Shader {name} is malformed: {error}")
        }

        Ok(module)
    }

    pub fn create_compute_pipeline(
        device: &wgpu::Device,
        descriptor: &wgpu::ComputePipelineDescriptor,
    ) -> Result<wgpu::ComputePipeline> {
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let pipeline = device.create_compute_pipeline(descriptor);

        // device.create_shader_module panics if the shader is malformed
        // only check this on native debug builds.
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            anyhow::bail!(
                "Compute pipeline {:?} is malformed: {error}",
                descriptor.label
            )
        }

        Ok(pipeline)
    }

    pub fn create_render_pipeline(
        device: &wgpu::Device,
        descriptor: &wgpu::RenderPipelineDescriptor,
    ) -> Result<wgpu::RenderPipeline> {
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let pipeline = device.create_render_pipeline(descriptor);

        // device.create_shader_module panics if the shader is malformed
        // only check this on native debug builds.
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            anyhow::bail!(
                "Render pipeline {:?} is malformed: {error}",
                descriptor.label
            )
        }

        Ok(pipeline)
    }

    /// Build a shader file by importing all its dependencies.
    /// We use seen to make sure we do not import the same file twice.
    /// Order of import does not matter in wgsl, as it does not in rust
    /// so we don't need to sort the imports depending on their dependencies.
    /// However we cannot define the same symbol twice, so we need to make sure
    /// we do not import the same file twice.
    fn build_with_seen(name: &str, seen: &mut Vec<String>) -> Result<String> {
        // File was already included, return empty string.
        let owned_name = name.to_owned();
        if seen.contains(&owned_name) {
            return Ok("".to_owned());
        }
        seen.push(owned_name);

        Self::load(name)?
            .lines()
            .map(|line| {
                // example of valid import: #import "common.wgsl"
                // note: this follow the bevy preprocessor syntax.
                // wgsl-analyzer is also based on the bevy preprocessor.
                // but does not support #import "file" as of August 2023.
                if line.starts_with("#import") {
                    let include = line
                        .split('"')
                        .nth(1)
                        .expect("Invalid import syntax: expected #import \"file\"");
                    let include_content = Self::build_with_seen(include, seen)?;
                    // We keep the import commented for debugging purposes.
                    Ok(format!("//{line}\n {include_content}"))
                } else {
                    Ok(format!("{line}\n"))
                }
            })
            .collect::<Result<String>>()
    }
}
