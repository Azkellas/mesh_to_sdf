/// Reload flags contain the state of the library / shader folder
/// `shaders` contains the shaders that were updated until last rebuild
/// `lib` is the state of the library
#[derive(Debug)]
pub struct ReloadFlags {
    pub shaders: Vec<String>,
}
