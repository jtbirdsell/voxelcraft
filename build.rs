//! Build script: make the DX12 backend hardware-RT-capable by staging the DXC shader compiler next
//! to the executable. DX12 ray tracing requires DXC (`dxcompiler.dll` + `dxil.dll`) — the default
//! FXC cannot compile ray-query shaders. Source order: a vendored in-repo `dll/` dir, else the
//! installed Windows SDK. If neither is found the build still succeeds; DX12 then falls back to FXC
//! (the software DDA tracer) and only the Vulkan backend provides hardware RT. (DLSS is Vulkan-only.)

use std::path::{Path, PathBuf};

const DLLS: [&str; 2] = ["dxcompiler.dll", "dxil.dll"];

fn main() {
    println!("cargo:rerun-if-changed=dll");
    println!("cargo:rerun-if-env-changed=STREAMLINE_SDK");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");

    // DXC/Streamline staging is Windows-only (DX12 + DLSS). On macOS/Linux there is nothing to
    // stage and the "DXC not found" warning would be noise, so bail before it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let Some(exe_dir) = exe_output_dir() else {
        return;
    };

    // DXC (hardware RT on DX12). Missing DXC is non-fatal — DX12 falls back to FXC (software DDA);
    // Vulkan hardware RT is unaffected.
    match dxc_source_dir() {
        Some(src_dir) => {
            for dll in DLLS {
                let src = src_dir.join(dll);
                if let Err(e) = std::fs::copy(&src, exe_dir.join(dll)) {
                    println!("cargo:warning=failed to stage {} next to the exe: {e}", src.display());
                }
            }
        }
        None => println!(
            "cargo:warning=DXC not found (looked in ./dll and the Windows SDK); the DX12 backend \
             will fall back to FXC = software DDA tracer. Vulkan hardware RT is unaffected."
        ),
    }

    // Stage the Streamline + DLSS-G DLLs only when Frame Generation is built. Build scripts see
    // enabled cargo features via CARGO_FEATURE_<NAME> (not cfg!); "frame-generation" -> _FRAME_GENERATION.
    if std::env::var("CARGO_FEATURE_FRAME_GENERATION").is_ok() {
        stage_frame_generation_dlls(&exe_dir);
    }
}

/// Stage the NVIDIA Streamline interposer + plugins (`$STREAMLINE_SDK/bin/x64`) and `nvngx_dlssg.dll`
/// (`$DLSS_SDK/lib/Windows_x86_64/rel`) next to the exe, for DLSS Frame Generation at runtime
/// (M33-G8-FG). Skips quietly when the SDK env vars aren't set (FG just stays unavailable). These
/// DLLs are NVIDIA-redistributable and are never committed. (build.rs reruns when the env vars change.)
fn stage_frame_generation_dlls(exe_dir: &Path) {
    if let Ok(sl) = std::env::var("STREAMLINE_SDK") {
        let bin = Path::new(&sl).join("bin").join("x64");
        for dll in [
            "sl.interposer.dll",
            "sl.common.dll",
            "sl.dlss_g.dll",
            "sl.reflex.dll",
            "sl.pcl.dll",
        ] {
            let src = bin.join(dll);
            if src.exists() {
                let _ = std::fs::copy(&src, exe_dir.join(dll));
            } else {
                println!("cargo:warning=FG: Streamline DLL not found: {}", src.display());
            }
        }
    }
    if let Ok(dlss) = std::env::var("DLSS_SDK") {
        let src = Path::new(&dlss)
            .join("lib")
            .join("Windows_x86_64")
            .join("rel")
            .join("nvngx_dlssg.dll");
        if src.exists() {
            let _ = std::fs::copy(&src, exe_dir.join("nvngx_dlssg.dll"));
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
