//! Build script: make the DX12 backend hardware-RT-capable by staging the DXC shader compiler next
//! to the executable. DX12 ray tracing requires DXC (`dxcompiler.dll` + `dxil.dll`) — the default
//! FXC cannot compile ray-query shaders. Source order: a vendored in-repo `dll/` dir, else the
//! installed Windows SDK. If neither is found the build still succeeds; DX12 then falls back to FXC
//! (the software DDA tracer) and only the Vulkan backend provides hardware RT. (DLSS is Vulkan-only.)

use std::path::{Path, PathBuf};

const DLLS: [&str; 2] = ["dxcompiler.dll", "dxil.dll"];

fn main() {
    println!("cargo:rerun-if-changed=dll");

    let Some(exe_dir) = exe_output_dir() else {
        return;
    };
    let Some(src_dir) = dxc_source_dir() else {
        println!(
            "cargo:warning=DXC not found (looked in ./dll and the Windows SDK); the DX12 backend \
             will fall back to FXC = software DDA tracer. Vulkan hardware RT is unaffected."
        );
        return;
    };
    for dll in DLLS {
        let src = src_dir.join(dll);
        let dst = exe_dir.join(dll);
        if let Err(e) = std::fs::copy(&src, &dst) {
            println!("cargo:warning=failed to stage {} next to the exe: {e}", src.display());
        }
    }
}

/// The directory the final executable lands in (`target/<profile>/`), derived from `OUT_DIR`
/// (`target/<profile>/build/<pkg>-<hash>/out`).
fn exe_output_dir() -> Option<PathBuf> {
    let out_dir = std::env::var("OUT_DIR").ok()?;
    Path::new(&out_dir).ancestors().nth(3).map(Path::to_path_buf)
}

/// Where to copy the DXC DLLs from: a vendored in-repo `dll/` dir if present, else the
/// highest-versioned installed Windows SDK `bin/<version>/x64`.
fn dxc_source_dir() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let vendored = Path::new(&manifest).join("dll");
    if vendored.join("dxcompiler.dll").exists() {
        return Some(vendored);
    }

    let pf86 =
        std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let bin = Path::new(&pf86).join("Windows Kits").join("10").join("bin");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&bin)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("x64").join("dxcompiler.dll").exists())
        .collect();
    candidates.sort();
    candidates.pop().map(|p| p.join("x64"))
}
