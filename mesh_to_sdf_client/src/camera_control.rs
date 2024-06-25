use winit::event::MouseButton;
use winit_input_helper::WinitInputHelper;

// Naive look-at camera.
// This version removes the use of quaternion to avoid adding a dependency.
// To avoid having to do linear algebra ourselves, most computations are done in the shader.
// This is sub-optimal. Improving this is left as an exercise to the reader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraLookAt {
    /// Object the camera is looking at.
    pub center: glam::Vec3,
    /// Angle around the object, in radians.
    pub longitude: f32,
    /// latitude between -PI/2 and PI/2, 0 is flat, PI/2 is zenith, -PI/2 is nadir
    pub latitude: f32,
    /// Distance from center
    pub distance: f32,
}

impl Default for CameraLookAt {
    fn default() -> Self {
        // See object in 0,0,0 from the front top left
        CameraLookAt {
            center: glam::Vec3::ZERO,
            longitude: 2.0 * std::f32::consts::FRAC_PI_3,
            latitude: std::f32::consts::FRAC_PI_3,
            distance: 5.0,
        }
    }
}

impl CameraLookAt {
    /// Pan the camera with middle mouse click, zoom with scroll wheel, orbit with right mouse click.
    pub fn update(&mut self, input: &WinitInputHelper, window_size: [f32; 2]) -> bool {
        let mut captured = false;

        // change input mapping for orbit and panning here
        let orbit_button = MouseButton::Right;
        let translation_button = MouseButton::Middle;

        let mouse_delta = input.cursor_diff();
        if mouse_delta.0 != 0.0 || mouse_delta.1 != 0.0 {
            if input.mouse_held(orbit_button) {
                // Rotate around the object
                let delta_x = mouse_delta.0 / window_size[0] * std::f32::consts::TAU;
                let delta_y = mouse_delta.1 / window_size[1] * std::f32::consts::PI;
                self.longitude += delta_x;
                self.latitude += delta_y;
                self.latitude = self.latitude.clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.001,
                    std::f32::consts::FRAC_PI_2 - 0.001,
                );

                captured = true;
            }

            if input.mouse_held(translation_button) {
                // Translate the center.
                // TODO: this is not exact, we should move along the camera plane.
                // this is especially visible when near nadir or zenith.
                let dir = glam::Vec2::from_angle(self.longitude);
                let translation_dir = dir.perp();
                // The further away we are, the faster we move.
                let translation_weight = mouse_delta.0 / window_size[0] * self.distance;

                self.center.x += translation_dir.x * translation_weight;
                self.center.z += translation_dir.y * translation_weight;
                self.center.y += mouse_delta.1 / window_size[1] * self.distance;

                captured = true;
            }
        }

        if input.scroll_diff().1 != 0.0 {
            // Zoom
            self.distance -= input.scroll_diff().1 * self.distance * 0.2;
            // Don't allow zoom to reach 0 or 1e6 to avoid getting stuck / in float precision issue realm.
            self.distance = self.distance.clamp(0.05, 1e6);

            captured = true;
        }

        captured
    }

    /// Get the view matrix.
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let eye = self.get_eye();
        glam::Mat4::look_at_rh(eye.truncate(), self.center, glam::Vec3::Y)
    }

    /// Get the eye position.
    pub fn get_eye(&self) -> glam::Vec4 {
        let dir = glam::Vec3::new(
            self.longitude.cos() * self.latitude.cos(),
            self.latitude.sin(),
            self.longitude.sin() * self.latitude.cos(),
        );
        let eye: glam::Vec3 = self.center + dir * self.distance;
        glam::Vec4::from((eye, 1.0))
    }
}
