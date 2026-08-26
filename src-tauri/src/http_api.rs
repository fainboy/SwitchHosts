//! Local HTTP API server.
//!
//! Reproduces the four routes the Electron build exposed via Hono on
//! port 50761:
//!
//! | Method | Path           | Body |
//! |--------|----------------|------|
//! | GET    | `/`            | `Hello SwitchHosts!` |
//! | GET    | `/remote-test` | `# remote-test\n# <timestamp>` |
//! | GET    | `/api/list`    | `{success, data: flat_list}` JSON |
//! | GET    | `/api/toggle?id=<id>` | `ok` / `bad id.` / `not found.` / see below |
//!
//! When the toggle is applied in the backend (no renderer, see below)
//! it can also answer `cancelled.` (user dismissed the OS auth
//! prompt), `write mode not set.`, `applied but not persisted.` or
//! `apply failed.`. All replies are 200 with a terse body, matching
//! the existing ones.
//!
//! Lifecycle:
//!
//! - `start(app, only_local)` binds to `127.0.0.1:50761` (only_local =
//!   true) or `0.0.0.0:50761` (only_local = false), spawns a tokio
//!   task that runs the axum router, and stores the join handle in a
//!   process-wide `Mutex`. Subsequent `start` calls with the same
//!   `only_local` are no-ops; calls with a different value tear down
//!   and rebind.
//! - `stop()` aborts the join handle and clears the slot.
//! - The bootstrap path in `lib.rs::run` calls `start` once at startup
//!   if `config.http_api_on == true`. The `config_set` /
//!   `config_update` commands call `start` / `stop` whenever the
//!   `http_api_on` or `http_api_only_local` keys change so the server
//!   stays in sync with the renderer's preferences pane without a
//!   restart.
//!
//! Toggle behaviour: with a main window alive this matches the
//! Electron implementation byte for byte — the handler emits
//! `toggle_item` with the flipped `on` value and the window's
//! `onToggleItem` runs the apply pipeline, keeping `choice_mode` /
//! folder semantics from `setOnStateOfItem` in one place.
//!
//! A Tauri event with no listener is dropped, though, and there are
//! two supported configurations with no main window: `hide_at_launch`
//! skips window creation at setup, and `lightweight_mode` destroys the
//! window when the user closes it. In both, the broadcast reaches
//! nobody while the endpoint still answers `ok`. The v5 storage plan
//! anticipated this — it noted that applying directly inside the HTTP
//! handler works even when no renderer is alive — so that is what the
//! no-window path does, reusing the renderer's selection rules
//! (`manifest::set_on_state_of_item`) and the same apply pipeline
//! (`commands::apply_aggregated_content`) rather than a bare write.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Mutex;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, Wry};

use crate::commands;
use crate::hosts_apply::{self, HostsApplyError};
use crate::lifecycle::MAIN_WINDOW_LABEL;
use crate::storage::{manifest, manifest::Manifest, AppState};
use crate::tray;

// We pin the HTTP API to the default `Wry` runtime instead of staying
// generic over `R: Runtime`. axum's `Handler` trait requires the
// extracted state to be `Clone + Send + Sync + 'static`, and a derived
// `Clone` on a `<R>`-parameterised wrapper struct requires `R: Clone`
// which `Runtime` doesn't guarantee. Pinning to `Wry` is harmless —
// it's the only runtime we ship, the test runtime never reaches this
// code path.

pub const HTTP_API_PORT: u16 = 50761;

struct ServerHandle {
    task: tauri::async_runtime::JoinHandle<()>,
    only_local: bool,
}

static SERVER: Mutex<Option<ServerHandle>> = Mutex::new(None);

/// Start the HTTP server. Idempotent: a second call with the same
/// `only_local` value is a no-op; with a different value the existing
/// server is stopped and a new one is bound.
pub fn start(app: AppHandle<Wry>, only_local: bool) -> Result<(), String> {
    let mut guard = SERVER.lock().expect("http server mutex poisoned");
    if let Some(existing) = guard.as_ref() {
        if existing.only_local == only_local {
            return Ok(());
        }
    }
    if let Some(prev) = guard.take() {
        prev.task.abort();
    }

    let ip = if only_local {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    };
    let addr = SocketAddr::new(ip, HTTP_API_PORT);

    // Bind synchronously so port-conflict errors surface to the caller
    // (and through it the renderer / config-update flow). Doing the
    // bind inside the spawned task would only log the failure while
    // `start()` returned `Ok`, leaving the preferences pane reporting
    // "HTTP API on" against a dead listener and blocking later
    // same-`only_local` calls via the early-return above. The std
    // listener is handed off to tokio inside `serve()`.
    let std_listener =
        std::net::TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
    std_listener
        .set_nonblocking(true)
        .map_err(|e| format!("set_nonblocking {addr}: {e}"))?;

    let app_for_task = app.clone();
    let task = tauri::async_runtime::spawn(async move {
        if let Err(e) = serve(app_for_task, std_listener).await {
            log::error!("serve error: {e}");
        }
    });

    *guard = Some(ServerHandle { task, only_local });
    log::info!("listening on http://{addr}");
    Ok(())
}

/// Stop the HTTP server if it's running.
pub fn stop() {
    let mut guard = SERVER.lock().expect("http server mutex poisoned");
    if let Some(handle) = guard.take() {
        handle.task.abort();
        log::info!("stopped");
    }
}

// ---- routes ----------------------------------------------------------------

async fn serve(app: AppHandle<Wry>, std_listener: std::net::TcpListener) -> Result<(), String> {
    let router = Router::new()
        .route("/", get(home))
        .route("/remote-test", get(remote_test))
        .route("/api/list", get(api_list))
        .route("/api/toggle", get(api_toggle))
        .with_state(AppRouterState { app });

    let listener = tokio::net::TcpListener::from_std(std_listener)
        .map_err(|e| format!("tokio listener from_std: {e}"))?;
    axum::serve(listener, router)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Clone)]
struct AppRouterState {
    app: AppHandle<Wry>,
}

async fn home() -> &'static str {
    "Hello SwitchHosts!"
}

async fn remote_test() -> String {
    let now = chrono::Local::now().format("%a %b %e %Y %H:%M:%S GMT%z");
    format!("# remote-test\n# {now}")
}

async fn api_list(State(state): State<AppRouterState>) -> Response {
    let app_state = state.app.state::<AppState>();
    match Manifest::load(&app_state.paths) {
        Ok(manifest) => {
            let flat = flatten_root(&manifest.root);
            Json(json!({ "success": true, "data": flat })).into_response()
        }
        Err(e) => Json(json!({
            "success": false,
            "message": e.to_string(),
        }))
        .into_response(),
    }
}

#[derive(Deserialize)]
struct ToggleQuery {
    id: Option<String>,
}

async fn api_toggle(
    State(state): State<AppRouterState>,
    Query(q): Query<ToggleQuery>,
) -> &'static str {
    let Some(id) = q.id else {
        return "bad id.";
    };
    if id.is_empty() {
        return "bad id.";
    }
    log::info!("toggle: {id}");

    let app_state = state.app.state::<AppState>();
    let manifest = match Manifest::load(&app_state.paths) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("manifest load failed: {e}");
            return "not found.";
        }
    };
    let Some(node) = find_node(&manifest.root, &id) else {
        return "not found.";
    };
    let on = node.get("on").and_then(Value::as_bool).unwrap_or(false);

    // Mirror Electron: broadcast `toggle_item` so the main window's
    // existing onToggleItem handler runs the full apply pipeline,
    // including choice_mode / folder cascading semantics from
    // `setOnStateOfItem`. The envelope is the same `_args` shape every
    // other Tauri broadcast in this codebase uses.
    if state.app.get_webview_window(MAIN_WINDOW_LABEL).is_some() {
        let _ = state.app.emit("toggle_item", json!({ "_args": [id, !on] }));
        return "ok";
    }

    // No main window: a Tauri event with no listener is dropped, so the
    // broadcast above would silently do nothing. Apply in the handler
    // instead, reusing the same selection rules the renderer applies.
    match apply_toggle_in_backend(&state.app, manifest, &id, !on).await {
        Ok(()) => "ok",
        Err(e) => {
            // A cancelled prompt is a deliberate user action, not a fault;
            // the renderer path stays quiet about it too.
            if matches!(e, ToggleError::Apply(HostsApplyError::Cancelled)) {
                log::info!("toggle cancelled by the user: {id}");
            } else {
                log::warn!("toggle failed for {id}: {e}");
            }
            e.as_body()
        }
    }
}

/// Serialises backend applies. Without it two concurrent HTTP toggles
/// would each stack an OS auth prompt and race to write the same files.
static BACKEND_APPLY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Backend-side equivalent of the renderer's `onToggleItem`: flip the
/// node, apply through the same pipeline a UI-driven apply uses, then
/// persist the tree. Used only when no renderer is alive to receive
/// `toggle_item`.
async fn apply_toggle_in_backend(
    app: &AppHandle<Wry>,
    manifest: Manifest,
    id: &str,
    on: bool,
) -> Result<(), ToggleError> {
    // One backend apply at a time. The privileged write can sit on an OS
    // auth prompt indefinitely, and `cmd_after_hosts_apply` adds up to
    // 30s on top; letting requests overlap would stack prompts and make
    // the store races below unavoidable.
    let _apply_guard = BACKEND_APPLY_LOCK.lock().await;

    let app_state = app.state::<AppState>();
    app_state
        .require_data_dir_usable()
        .map_err(|e| ToggleError::Storage(e.to_string()))?;

    let (choice_mode, multi_chose_folder_switch_all, remove_duplicate, write_mode) = {
        let cfg = app_state.config.lock().expect("config mutex poisoned");
        (
            cfg.choice_mode as u64,
            cfg.multi_chose_folder_switch_all,
            cfg.remove_duplicate_records,
            cfg.write_mode.clone(),
        )
    };

    // The renderer refuses to apply before a write mode is chosen and
    // opens the picker instead (`onToggleItem` in List/index.tsx). We
    // have no UI to fall back on, so refuse rather than silently taking
    // `apply_to_system_hosts`'s overwrite default — that would wipe
    // hand-written entries for anyone still carrying the empty value
    // from an Electron-era config.
    if write_mode.is_empty() {
        return Err(ToggleError::WriteModeUnset);
    }

    let mut proposed = manifest;
    manifest::set_on_state_of_item(
        &mut proposed.root,
        id,
        on,
        choice_mode,
        multi_chose_folder_switch_all,
    );

    let content =
        hosts_apply::aggregate_selected_content(&proposed.root, &app_state.paths, remove_duplicate)
            .map_err(|e| ToggleError::Storage(e.to_string()))?;

    commands::apply_aggregated_content(app, app_state.inner(), &content)
        .await
        .map_err(|commands::ApplyPipelineError::Apply(e)| ToggleError::Apply(e))?;

    // Re-read under the store lock and re-apply the flip, rather than
    // saving the tree we loaded before the write. The apply above can
    // block on an auth prompt for minutes, and the refresh scanner or a
    // tray window may have legitimately rewritten manifest.json in the
    // meantime — saving our stale snapshot would clobber that. Same
    // reasoning as the remote-refresh path in `refresh.rs`.
    {
        let _guard = app_state.store_lock.lock().expect("store lock poisoned");
        let mut fresh =
            Manifest::load(&app_state.paths).map_err(|e| ToggleError::Persist(e.to_string()))?;
        manifest::set_on_state_of_item(
            &mut fresh.root,
            id,
            on,
            choice_mode,
            multi_chose_folder_switch_all,
        );
        fresh
            .save(&app_state.paths)
            .map_err(|e| ToggleError::Persist(e.to_string()))?;
    }

    // `tray::refresh_title` reads manifest.json from disk, so the call
    // inside the apply pipeline saw the pre-toggle tree. Refresh again
    // now that the new one has landed, otherwise the menubar title
    // trails one toggle behind — and with no window open it is the only
    // place the user can see which profile is active.
    if let Err(e) = tray::refresh_title(app, app_state.inner()) {
        log::warn!("failed to refresh tray title after backend toggle: {e}");
    }
    // Mirrors the renderer's post-apply broadcast: a tray mini window is
    // built lazily and then reused, so without this its list keeps
    // showing the pre-toggle state.
    let _ = app.emit("tray_list_updated", json!({ "_args": [] }));

    Ok(())
}

/// Why a backend toggle failed. Kept distinct so the endpoint can say
/// which, instead of collapsing a user-cancelled prompt, a policy denial
/// and a full disk into one opaque string.
enum ToggleError {
    WriteModeUnset,
    Apply(HostsApplyError),
    /// The write succeeded but the tree could not be persisted — the
    /// system file and manifest.json now disagree.
    Persist(String),
    Storage(String),
}

impl ToggleError {
    /// Response body. Terse and stable, like the existing `bad id.` /
    /// `not found.` replies.
    fn as_body(&self) -> &'static str {
        match self {
            ToggleError::WriteModeUnset => "write mode not set.",
            ToggleError::Apply(HostsApplyError::Cancelled) => "cancelled.",
            ToggleError::Apply(_) => "apply failed.",
            ToggleError::Persist(_) => "applied but not persisted.",
            ToggleError::Storage(_) => "apply failed.",
        }
    }
}

impl std::fmt::Display for ToggleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToggleError::WriteModeUnset => write!(f, "write mode is not set"),
            ToggleError::Apply(e) => write!(f, "{e}"),
            ToggleError::Persist(e) => write!(f, "applied but failed to persist the tree: {e}"),
            ToggleError::Storage(e) => write!(f, "{e}"),
        }
    }
}

// ---- tree helpers ----------------------------------------------------------

fn flatten_root(nodes: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    walk(nodes, &mut out);
    out
}

fn walk(nodes: &[Value], out: &mut Vec<Value>) {
    for node in nodes {
        out.push(node.clone());
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            walk(children, out);
        }
    }
}

fn find_node(nodes: &[Value], id: &str) -> Option<Value> {
    for node in nodes {
        if node.get("id").and_then(Value::as_str) == Some(id) {
            return Some(node.clone());
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            if let Some(found) = find_node(children, id) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_fixture() -> Vec<Value> {
        // root
        // ├── local-1
        // ├── folder-a
        // │   ├── local-2
        // │   └── folder-b
        // │       └── local-3
        // └── local-4
        json!([
            { "id": "local-1", "type": "local", "on": true },
            {
                "id": "folder-a",
                "type": "folder",
                "children": [
                    { "id": "local-2", "type": "local", "on": false },
                    {
                        "id": "folder-b",
                        "type": "folder",
                        "children": [
                            { "id": "local-3", "type": "local", "on": true },
                        ]
                    }
                ]
            },
            { "id": "local-4", "type": "local", "on": false },
        ])
        .as_array()
        .cloned()
        .unwrap()
    }

    #[test]
    fn flatten_root_emits_parents_before_descendants_in_dfs_order() {
        let flat = flatten_root(&tree_fixture());
        let ids: Vec<&str> = flat.iter().filter_map(|n| n.get("id")?.as_str()).collect();
        assert_eq!(
            ids,
            vec!["local-1", "folder-a", "local-2", "folder-b", "local-3", "local-4"]
        );
    }

    #[test]
    fn flatten_root_handles_empty_tree() {
        assert!(flatten_root(&[]).is_empty());
    }

    #[test]
    fn find_node_locates_top_level_id() {
        let n = find_node(&tree_fixture(), "local-4").unwrap();
        assert_eq!(n.get("type").and_then(Value::as_str), Some("local"));
    }

    #[test]
    fn find_node_recurses_into_nested_folders() {
        // Two levels deep — exercises the recursive arm.
        let n = find_node(&tree_fixture(), "local-3").unwrap();
        assert_eq!(n.get("on").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn find_node_returns_none_for_missing_id() {
        assert!(find_node(&tree_fixture(), "does-not-exist").is_none());
    }

    #[test]
    fn find_node_skips_folder_with_non_array_children_field() {
        // A malformed node whose `children` is not an array should not
        // panic and should not be treated as a parent.
        let nodes = json!([
            { "id": "weird", "children": "not-an-array" },
            { "id": "real", "type": "local" },
        ])
        .as_array()
        .cloned()
        .unwrap();
        assert!(find_node(&nodes, "real").is_some());
        assert!(find_node(&nodes, "missing").is_none());
    }

    #[tokio::test]
    async fn home_route_returns_static_greeting() {
        assert_eq!(home().await, "Hello SwitchHosts!");
    }

    #[tokio::test]
    async fn remote_test_route_starts_with_marker_and_carries_timestamp() {
        let body = remote_test().await;
        assert!(
            body.starts_with("# remote-test\n# "),
            "unexpected body prefix: {body:?}"
        );
        // Timestamp must be non-empty (the chrono format string is dynamic).
        let ts = &body["# remote-test\n# ".len()..];
        assert!(!ts.is_empty());
    }
}
