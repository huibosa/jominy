#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    net::TcpListener,
    sync::Mutex,
    time::{Duration, Instant},
};
use tauri::menu::{Menu, MenuItemBuilder, SubmenuBuilder};
use tauri::{Manager, WindowEvent};

struct SidecarProcess(Mutex<Option<std::process::Child>>);

fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind to free port");
    listener.local_addr().unwrap().port()
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            // Brief pause so uvicorn finishes startup after the port opens
            std::thread::sleep(Duration::from_millis(300));
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn log_dir() -> std::path::PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from(".")));
    base.join("Jominy").join("logs")
}

fn spawn_sidecar(app: &tauri::AppHandle, port: u16) -> Result<std::process::Child, String> {
    #[cfg(debug_assertions)]
    {
        // Dev mode: run the Python script directly (no bundled binary yet)
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let project_root = std::path::Path::new(manifest_dir)
            .parent()
            .expect("manifest_dir has no parent");
        let script = project_root.join("webapp").join("backend").join("main.py");
        std::process::Command::new("python3")
            .args([script.to_str().unwrap(), "--port", &port.to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map(pipe_child_output)
            .map_err(|e| format!("Failed to start dev backend (python3 {script:?}): {e}"))
    }
    #[cfg(not(debug_assertions))]
    {
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| format!("resource_dir: {e}"))?;
        let exe = resource_dir.join("backend").join("main.exe");
        std::process::Command::new(&exe)
            .args(["--host=127.0.0.1", &format!("--port={port}")])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map(pipe_child_output)
            .map_err(|e| format!("Failed to start backend ({}): {e}", exe.display()))
    }
}

fn pipe_child_output(mut child: std::process::Child) -> std::process::Child {
    let log_path = log_dir().join("tauri.log");
    if let Ok(()) = std::fs::create_dir_all(log_dir()) {
        use std::io::{BufRead, Write};

        let write_lines = |reader: Box<dyn std::io::Read + Send>, path: std::path::PathBuf| {
            std::thread::spawn(move || {
                let buf = std::io::BufReader::new(reader);
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    for line in buf.lines().flatten() {
                        let _ = writeln!(f, "{line}");
                    }
                }
            });
        };

        if let Some(stdout) = child.stdout.take() {
            write_lines(Box::new(stdout), log_path.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            write_lines(Box::new(stderr), log_path);
        }
    }
    child
}

/// Attach the child process to a Windows Job Object so the OS terminates it
/// automatically if the parent (Tauri) process dies unexpectedly.
#[cfg(target_os = "windows")]
fn attach_job_object(child: &std::process::Child) {
    use windows::Win32::{
        Foundation::CloseHandle,
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
                SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            },
            Threading::{OpenProcess, PROCESS_ALL_ACCESS},
        },
    };
    unsafe {
        let Ok(job) = CreateJobObjectW(None, None) else {
            return;
        };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let _ = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            std::ptr::addr_of!(info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        let Ok(proc) = OpenProcess(PROCESS_ALL_ACCESS, false, child.id()) else {
            let _ = CloseHandle(job);
            return;
        };
        let _ = AssignProcessToJobObject(job, proc);
        let _ = CloseHandle(proc);
        // Leak the job handle intentionally: when Tauri exits (normally or by crash),
        // the leaked HANDLE is closed by the OS, which triggers KILL_ON_JOB_CLOSE.
        std::mem::forget(job);
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let port = pick_free_port();

            // Build the menu: File → Quit, Help → Open Log Folder
            let quit_item = MenuItemBuilder::new("Quit")
                .id("quit")
                .accelerator("CmdOrCtrl+Q")
                .build(app)?;
            let open_logs_item = MenuItemBuilder::new("Open Log Folder")
                .id("open-log-folder")
                .build(app)?;
            let file_menu = SubmenuBuilder::new(app, "File").item(&quit_item).build()?;
            let help_menu = SubmenuBuilder::new(app, "Help")
                .item(&open_logs_item)
                .build()?;
            let menu = Menu::new(app)?;
            menu.append(&file_menu)?;
            menu.append(&help_menu)?;

            // Build the window here (not from tauri.conf.json windows[]) so we can
            // inject __JOMINY_API__ via initialization_script before any JS executes.
            let window = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("Jominy Hardenability Predictor")
            .inner_size(900.0, 780.0)
            .min_inner_size(720.0, 640.0)
            .resizable(true)
            .initialization_script(&format!(
                "window.__JOMINY_API__ = 'http://127.0.0.1:{port}';"
            ))
            .menu(menu)
            .build()?;

            // Spawn the backend sidecar
            let child = match spawn_sidecar(app.handle(), port) {
                Ok(c) => c,
                Err(msg) => {
                    use tauri_plugin_dialog::DialogExt;
                    app.dialog()
                        .message(format!(
                            "Backend failed to start:\n{msg}\n\nLogs: {}",
                            log_dir().display()
                        ))
                        .title("Jominy — Startup Error")
                        .blocking_show();
                    app.handle().exit(1);
                    return Ok(());
                }
            };

            #[cfg(target_os = "windows")]
            attach_job_object(&child);

            app.manage(SidecarProcess(Mutex::new(Some(child))));

            // Background thread: poll the port, then fire backend-ready into the WebView.
            let win = window.clone();
            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                if !wait_for_port(port, Duration::from_secs(90)) {
                    use tauri_plugin_dialog::DialogExt;
                    app_handle
                        .dialog()
                        .message(format!(
                            "Backend did not become ready within 15 seconds.\n\nLogs: {}",
                            log_dir().display()
                        ))
                        .title("Jominy — Startup Timeout")
                        .blocking_show();
                    app_handle.exit(1);
                    return;
                }
                let _ = win.eval("window.dispatchEvent(new CustomEvent('backend-ready'))");
            });

            Ok(())
        })
        .on_menu_event(|app, event| match event.id().0.as_str() {
            "quit" => app.exit(0),
            "open-log-folder" => {
                use tauri_plugin_shell::ShellExt;
                let dir = log_dir();
                let _ = std::fs::create_dir_all(&dir);
                let _ = app.shell().open(dir.to_str().unwrap_or("."), None);
            }
            _ => {}
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                let state = window.state::<SidecarProcess>();
                let mut guard = state.0.lock().unwrap();
                if let Some(mut child) = guard.take() {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
