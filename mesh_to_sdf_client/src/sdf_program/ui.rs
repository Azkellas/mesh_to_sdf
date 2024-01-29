impl super::SdfProgram {
    pub fn add_color_widget(ui: &mut egui::Ui, label: &str, color: [f32; 3]) -> Option<[f32; 3]> {
        ui.label(label);
        let mut new_color = color;
        egui::color_picker::color_edit_button_rgb(ui, &mut new_color);

        if !float_cmp::approx_eq!(f32, color[0], new_color[0], ulps = 2, epsilon = 1e-6)
            || !float_cmp::approx_eq!(f32, color[1], new_color[1], ulps = 2, epsilon = 1e-6)
            || !float_cmp::approx_eq!(f32, color[2], new_color[2], ulps = 2, epsilon = 1e-6)
        {
            Some(new_color)
        } else {
            None
        }
    }
}
