cargo build --target wasm32-unknown-unknown
wasm-bindgen --out-dir target/generated --web .\target\wasm32-unknown-unknown\%1\mesh_to_sdf_client.wasm