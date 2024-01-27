/// TODO: rust hot-reload was disabled in this project, remove this enum.
/// Library state in hot reload mode
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum LibState {
    /// Library is stable: nothing to do
    Stable,
    /// Library is reloading: avoid calls to its function until it's done
    Reloading,
    /// Library is done reloading
    Reloaded,
}

/// Reload flags contain the state of the library / shader folder
/// `shaders` contains the shaders that were updated until last rebuild
/// `lib` is the state of the library
#[derive(Debug)]
pub struct ReloadFlags {
    pub shaders: Vec<String>,
    pub lib: LibState,
}
