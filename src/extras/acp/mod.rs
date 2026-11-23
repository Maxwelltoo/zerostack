pub mod config;

use std::collections::HashMap;
use std::sync::Arc;

use agent_client_protocol::on_receive_request;
use agent_client_protocol::v1::schema::*;
use agent_client_protocol::v1::{ConnectionTo, Responder};
use agent_client_protocol::{Agent, ByteStreams, Client, ConnectTo, Role, Stdio};
use compact_str::CompactString;
use rig::completion::Message;
use tokio::sync::Mutex;

use crate::cli::Cli;
use crate::config::Config;
use crate::context::ContextFiles;
use crate::event::AgentEvent;
use crate::permission::SecurityMode;
use crate::permission::ask::AskSender;
use crate::permission::checker::{PermCheck, PermissionChecker};
use crate::sandbox::{SandboxSettings, SandboxSetup};

const AGENT_VERSION: &str = "1.0.5";

const REASONING_EFFORTS: &[(&str, &str)] = &[
    ("off", "Off"),
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("max", "Max"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct AcpModelOption {
    id: String,
    name: String,
    provider: Option<String>,
}

struct SessionState {
    messages: Vec<Message>,
    provider: String,
    model: String,
    reasoning_effort: String,
    available_models: Vec<AcpModelOption>,
}

impl SessionState {
    fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            provider: String::new(),
            model: String::new(),
            reasoning_effort: "off".to_string(),
            available_models: Vec::new(),
        }
    }
    /// Return the conversation so far and record `prompt` as this turn's user
    /// message. The snapshot excludes the new prompt because the runner sends
    /// it separately, as a prompt rather than history.
    fn begin_turn(&mut self, prompt: &str) -> Vec<Message> {
        let history = self.messages.clone();
        self.messages.push(Message::user(prompt.to_string()));
        history
    }

    fn record_assistant_response(&mut self, response: &str) {
        if !response.is_empty() {
            self.messages.push(Message::assistant(response.to_string()));
        }
    }
}

struct AcpState {
    cli: Cli,
    cfg: Config,
    context: ContextFiles,
    sessions: Mutex<HashMap<SessionId, SessionState>>,
}

/// The session sandbox and the warnings building it produced, from this
/// server's resolved settings. Shared by `handle_new_session` (which logs the
/// warnings once) and `run_prompt` (which needs the sandbox on every prompt),
/// so the two can never disagree about what is masked or exposed.
fn sandbox_setup(state: &AcpState) -> SandboxSetup {
    crate::sandbox::build_sandbox(&SandboxSettings {
        enabled: state.cli.resolve_sandbox(&state.cfg),
        required: state.cli.resolve_sandbox_required(&state.cfg),
        backend: &state.cli.resolve_sandbox_backend(&state.cfg),
        shell: &state.cli.resolve_shell(&state.cfg),
        expose: &state.cli.resolve_sandbox_expose(&state.cfg),
        network: state.cli.resolve_sandbox_network(&state.cfg),
    })
}

// --- TCP Transport ---

struct TcpTransport {
    host: String,
    port: u16,
}

impl<Counterpart: Role> ConnectTo<Counterpart> for TcpTransport {
    async fn connect_to(
        self,
        client: impl ConnectTo<Counterpart::Counterpart>,
    ) -> Result<(), agent_client_protocol::Error> {
        use std::net::TcpListener;

        let addr = format!("{}:{}", self.host, self.port);
        let listener = TcpListener::bind(&addr).map_err(|e| {
            agent_client_protocol::Error::internal_error().data(format!("TCP bind {}: {}", addr, e))
        })?;

        tracing::info!("ACP TCP listening on {}", addr);

        let (stream, peer_addr) = listener.accept().map_err(|e| {
            agent_client_protocol::Error::internal_error().data(format!("TCP accept: {}", e))
        })?;

        tracing::info!("ACP client connected from {}", peer_addr);

        let read_half = stream.try_clone().map_err(|e| {
            agent_client_protocol::Error::internal_error().data(format!("TCP clone: {}", e))
        })?;
        let write_half = stream;

        let read_unblock = blocking::Unblock::new(read_half);
        let write_unblock = blocking::Unblock::new(write_half);

        ConnectTo::<Counterpart>::connect_to(ByteStreams::new(write_unblock, read_unblock), client)
            .await
    }
}

// --- Server Entry Point ---

pub async fn serve(cli: Cli, cfg: Config, context: ContextFiles) -> anyhow::Result<()> {
    let transport_mode = if cli.acp_host.is_some() {
        "tcp"
    } else {
        "stdio"
    };
    tracing::info!("ACP server starting: transport={}", transport_mode);

    // Extract transport config before moving cli into Arc
    let acp_host = cli.acp_host.clone();
    let acp_port = cli.acp_port;

    let state = Arc::new(AcpState {
        cli,
        cfg,
        context,
        sessions: Mutex::new(HashMap::new()),
    });

    let builder = Agent.v1().name("zerostack");

    let builder = builder
        .on_receive_request(
            {
                let state = state.clone();
                move |req: InitializeRequest, responder, _cx| {
                    let state = state.clone();
                    async move { handle_initialize(req, responder, &state).await }
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |req: NewSessionRequest, responder, cx| {
                    let state = state.clone();
                    async move { handle_new_session(req, responder, cx, &state).await }
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |req: SetSessionConfigOptionRequest, responder, _cx| {
                    let state = state.clone();
                    async move { handle_set_config_option(req, responder, &state).await }
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |req: PromptRequest, responder, cx| {
                    let state = state.clone();
                    async move { handle_prompt(req, responder, cx, state).await }
                }
            },
            on_receive_request!(),
        );

    // Choose transport: TCP if host is set, otherwise stdio
    if let Some(host) = acp_host {
        let port = acp_port.unwrap_or(7243);
        builder
            .connect_to(TcpTransport { host, port })
            .await
            .map_err(|e| anyhow::anyhow!("ACP TCP server error: {}", e))?;
    } else {
        builder
            .connect_to(Stdio::new())
            .await
            .map_err(|e| anyhow::anyhow!("ACP stdio server error: {}", e))?;
    }

    Ok(())
}

// --- Request Handlers ---

async fn handle_initialize(
    req: InitializeRequest,
    responder: Responder<InitializeResponse>,
    _state: &AcpState,
) -> Result<(), agent_client_protocol::Error> {
    let caps = AgentCapabilities::new();

    let resp = InitializeResponse::new(req.protocol_version)
        .agent_capabilities(caps)
        .agent_info(Implementation::new("zerostack", AGENT_VERSION));

    responder.respond(resp)
}

fn resolve_acp_model(cli: &Cli, cfg: &Config, provider: &str) -> String {
    let mut model = cli.resolve_model(cfg);
    if (model.as_str() == "deepseek/deepseek-v4-pro" || cli.model.is_none())
        && let Some(custom) = cfg.custom_providers_map().get(provider)
        && let Some(ref custom_model) = custom.model
    {
        model = custom_model.clone();
    }
    model.to_string()
}

fn initial_reasoning_effort(cfg: &Config, model: &str) -> String {
    let Some(extra) = crate::config::resolve_extra_body(cfg, model) else {
        return "off".to_string();
    };
    if let Some(effort) = extra.get("reasoning_effort").and_then(|v| v.as_str())
        && REASONING_EFFORTS.iter().any(|(id, _)| *id == effort)
    {
        return effort.to_string();
    }
    if let Some(effort) = extra
        .get("output_config")
        .and_then(|v| v.get("effort"))
        .and_then(|v| v.as_str())
        && REASONING_EFFORTS.iter().any(|(id, _)| *id == effort)
    {
        return effort.to_string();
    }
    if extra
        .get("thinking")
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
        == Some("adaptive")
    {
        return "medium".to_string();
    }
    "off".to_string()
}

fn push_unique_model(
    out: &mut Vec<AcpModelOption>,
    id: impl Into<String>,
    name: impl Into<String>,
    provider: Option<String>,
) {
    let id = id.into();
    if id.is_empty() || out.iter().any(|m| m.id == id) {
        return;
    }
    let name = name.into();
    out.push(AcpModelOption {
        name: if name.is_empty() { id.clone() } else { name },
        id,
        provider,
    });
}

async fn collect_acp_models(
    cli: &Cli,
    cfg: &Config,
    active_provider: &str,
    active_model: &str,
) -> Vec<AcpModelOption> {
    let mut models = Vec::new();
    push_unique_model(
        &mut models,
        active_model,
        active_model,
        Some(active_provider.to_string()),
    );

    // Best-effort live discovery with a short timeout so offline or unauthenticated
    // startups never block `session/new`.
    let custom_providers = cfg.custom_providers_map();
    let live_models = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        if custom_providers.contains_key(active_provider) {
            crate::provider::list_models_manual(
                active_provider,
                cli.api_key.as_deref(),
                &custom_providers,
                cfg.api_keys.as_ref(),
            )
            .await
            .ok()
        } else if let Ok(client) = crate::provider::create_client(
            active_provider,
            cli.api_key.as_deref(),
            &custom_providers,
            cfg.api_keys.as_ref(),
        ) {
            client.list_models().await.ok()
        } else {
            None
        }
    })
    .await
    .ok()
    .flatten();

    if let Some(entries) = live_models {
        for entry in entries {
            push_unique_model(
                &mut models,
                entry.id,
                entry.display,
                Some(active_provider.to_string()),
            );
        }
    }

    let qm = crate::config::quick_models_map_ref(cfg);
    let mut qm_keys: Vec<&String> = qm.keys().collect();
    qm_keys.sort();
    for key in qm_keys {
        let q = &qm[key];
        let label = if q.model.as_str() == key.as_str() {
            q.model.to_string()
        } else {
            format!("{key} ({})", q.model)
        };
        push_unique_model(
            &mut models,
            q.model.as_str(),
            label,
            Some(q.provider.to_string()),
        );
    }

    let mut cp_keys: Vec<&String> = custom_providers.keys().collect();
    cp_keys.sort();
    for key in cp_keys {
        if let Some(ref model) = custom_providers[key].model {
            push_unique_model(
                &mut models,
                model.as_str(),
                model.as_str(),
                Some(key.clone()),
            );
        }
    }

    models
}

fn build_config_options(
    available_models: &[AcpModelOption],
    current_model: &str,
    current_reasoning: &str,
) -> Vec<SessionConfigOption> {
    let model_options: Vec<SessionConfigSelectOption> = available_models
        .iter()
        .map(|m| SessionConfigSelectOption::new(m.id.clone(), m.name.clone()))
        .collect();
    let reasoning_options: Vec<SessionConfigSelectOption> = REASONING_EFFORTS
        .iter()
        .map(|(id, name)| SessionConfigSelectOption::new(*id, *name))
        .collect();

    vec![
        SessionConfigOption::select(
            "model",
            "Model",
            current_model.to_string(),
            model_options,
        )
        .category(SessionConfigOptionCategory::Model),
        SessionConfigOption::select(
            "reasoning_effort",
            "Reasoning Effort",
            current_reasoning.to_string(),
            reasoning_options,
        )
        .category(SessionConfigOptionCategory::ThoughtLevel),
    ]
}

fn apply_reasoning_effort(
    base: Option<serde_json::Value>,
    provider: &str,
    cfg: &Config,
    effort: &str,
) -> Option<serde_json::Value> {
    let is_anthropic = crate::provider::resolve_provider_config(provider, &cfg.custom_providers_map())
        .is_ok_and(|p| p.kind == crate::auth::ProviderKind::Anthropic);

    if effort == "off" {
        if is_anthropic {
            return crate::provider::merge_extra_body(
                base,
                Some(serde_json::json!({ "thinking": { "type": "disabled" } })),
            );
        }
        return base;
    }

    let patch = if is_anthropic {
        serde_json::json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": effort }
        })
    } else {
        serde_json::json!({ "reasoning_effort": effort })
    };
    crate::provider::merge_extra_body(base, Some(patch))
}

async fn handle_new_session(
    req: NewSessionRequest,
    responder: Responder<NewSessionResponse>,
    _cx: ConnectionTo<Client>,
    state: &AcpState,
) -> Result<(), agent_client_protocol::Error> {
    let session_id = SessionId::new(uuid::Uuid::new_v4().to_string());

    tracing::info!(
        "ACP new session: {} (cwd: {})",
        session_id,
        req.cwd.display()
    );

    if state.cli.sandbox_setting_conflict(&state.cfg) {
        tracing::warn!(
            "sandbox is set to false but sandbox-required is set, enabling the sandbox anyway"
        );
    }
    // Sandbox warnings are emitted once per session, so they live here and not
    // in run_prompt (which runs on every prompt). The sandbox itself is rebuilt
    // per prompt from the same settings; building it here is what makes the
    // warnings describe the sandbox that will actually run, including the
    // directory it binds, which is this process's working directory and not
    // `req.cwd` (the ACP server never chdirs to it).
    for warning in &sandbox_setup(state).warnings {
        tracing::warn!("{warning}");
    }

    let provider = state.cli.resolve_provider(&state.cfg).to_string();
    let model = resolve_acp_model(&state.cli, &state.cfg, &provider);
    let reasoning_effort = initial_reasoning_effort(&state.cfg, &model);
    let available_models = collect_acp_models(&state.cli, &state.cfg, &provider, &model).await;
    let config_options = build_config_options(&available_models, &model, &reasoning_effort);

    let session_state = SessionState {
        messages: Vec::new(),
        provider,
        model,
        reasoning_effort,
        available_models,
    };
    state
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), session_state);

    let resp = NewSessionResponse::new(session_id).config_options(config_options);
    responder.respond(resp)
}

async fn handle_set_config_option(
    req: SetSessionConfigOptionRequest,
    responder: Responder<SetSessionConfigOptionResponse>,
    state: &AcpState,
) -> Result<(), agent_client_protocol::Error> {
    let value_str = match &req.value {
        SessionConfigOptionValue::ValueId { value } => value.0.as_ref().to_owned(),
        _ => {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("config option value must be a string id"));
        }
    };

    let mut sessions = state.sessions.lock().await;
    let Some(sess) = sessions.get_mut(&req.session_id) else {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unknown sessionId: {}", req.session_id)));
    };

    match req.config_id.0.as_ref() {
        "model" => {
            let Some(chosen) = sess
                .available_models
                .iter()
                .find(|m| m.id == value_str)
                .cloned()
            else {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown model: {value_str}")));
            };
            sess.model = chosen.id;
            if let Some(p) = chosen.provider {
                sess.provider = p;
            }
        }
        "reasoning_effort" => {
            if !REASONING_EFFORTS.iter().any(|(id, _)| *id == value_str) {
                return Err(agent_client_protocol::Error::invalid_params()
                    .data(format!("invalid reasoning_effort: {value_str}")));
            }
            sess.reasoning_effort = value_str;
        }
        other => {
            return Err(agent_client_protocol::Error::invalid_params()
                .data(format!("unknown configId: {other}")));
        }
    }

    let config_options = build_config_options(
        &sess.available_models,
        &sess.model,
        &sess.reasoning_effort,
    );
    drop(sessions);

    responder.respond(SetSessionConfigOptionResponse::new(config_options))
}

async fn handle_prompt(
    req: PromptRequest,
    responder: Responder<PromptResponse>,
    cx: ConnectionTo<Client>,
    state: Arc<AcpState>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id = req.session_id.clone();

    tracing::info!("ACP prompt for session {}", session_id);

    let prompt_text = req
        .prompt
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Snapshot the prior turns for the runner, then record this prompt. The
    // prompt itself travels separately, so it must not also be in `history`
    // (providers reject two consecutive user messages).
    let mut sessions = state.sessions.lock().await;
    let sess = sessions
        .entry(session_id.clone())
        .or_insert_with(|| SessionState::new(Vec::new()));
    let history = sess.begin_turn(&prompt_text);
    let provider_str = if sess.provider.is_empty() {
        state.cli.resolve_provider(&state.cfg).to_string()
    } else {
        sess.provider.clone()
    };
    let model_str = if sess.model.is_empty() {
        resolve_acp_model(&state.cli, &state.cfg, &provider_str)
    } else {
        sess.model.clone()
    };
    let reasoning_effort = sess.reasoning_effort.clone();
    drop(sessions);

    cx.spawn({
        let cx = cx.clone();
        async move {
            run_prompt(
                &state,
                &prompt_text,
                history,
                provider_str,
                model_str,
                reasoning_effort,
                session_id,
                responder,
                cx,
            )
            .await
        }
    })
}

// --- Prompt Execution ---

async fn run_prompt(
    state: &AcpState,
    prompt_text: &str,
    history: Vec<Message>,
    provider_str: String,
    model_str: String,
    reasoning_effort: String,
    session_id: SessionId,
    responder: Responder<PromptResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    tracing::debug!(
        "ACP run_prompt: provider={}, model={}, reasoning_effort={}, prompt_len={}",
        provider_str,
        model_str,
        reasoning_effort,
        prompt_text.len(),
    );

    let client = crate::provider::create_client(
        &provider_str,
        state.cli.api_key.as_deref(),
        &state.cfg.custom_providers_map(),
        state.cfg.api_keys.as_ref(),
    )
    .map_err(|e| agent_client_protocol::Error::new(-32603, e.to_string()))?;

    let model = client.completion_model(model_str.clone());

    let (permission, ask_tx) = build_acp_permission(state);
    // Warnings are dropped here on purpose: handle_new_session already logged
    // them once for this session.
    let sandbox = sandbox_setup(state).sandbox;

    let temperature = crate::config::resolve_temperature(&state.cli, &state.cfg, &model_str);
    let extra_body = apply_reasoning_effort(
        crate::config::resolve_extra_body(&state.cfg, &model_str),
        &provider_str,
        &state.cfg,
        &reasoning_effort,
    );
    let reasoning_enabled = reasoning_effort != "off";
    let agent = crate::provider::build_agent(
        model,
        crate::agent::builder::AgentBuild {
            cli: &state.cli,
            cfg: &state.cfg,
            context: &state.context,
            permission,
            ask_tx,
            sandbox,
            reasoning_enabled,
            temperature,
            extra_body,
            #[cfg(feature = "mcp")]
            mcp_manager: None,
        },
    )
    .await;

    let runner = agent
        .spawn_runner(
            crate::agent::runner::UserPrompt::text(prompt_text.to_string()),
            history,
            crate::retry::RetryConfig::default(),
            #[cfg(feature = "hooks")]
            None,
        )
        .await;
    let mut rx = runner.event_rx;

    // In-flight main-agent calls by `AgentEvent` id (rig's
    // `internal_call_id`) to the ACP ToolCallId announced for them. A map,
    // not a single slot: a parallel batch streams every `ToolCall` before
    // the first `ToolResult`.
    let mut tool_call_ids: HashMap<CompactString, ToolCallId> = HashMap::new();
    let mut final_response = String::new();

    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::Token(text) => {
                final_response.push_str(&text);
                let chunk =
                    ContentChunk::new(ContentBlock::Text(TextContent::new(text.to_string())));
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::AgentMessageChunk(chunk),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send token notification: {}", e);
                }
            }
            AgentEvent::Reasoning(text) => {
                let chunk =
                    ContentChunk::new(ContentBlock::Text(TextContent::new(text.to_string())));
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::AgentThoughtChunk(chunk),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send reasoning notification: {}", e);
                }
            }
            AgentEvent::ToolCall {
                call_id: event_id,
                name,
                args,
            } => {
                let id = ToolCallId::new(uuid::Uuid::new_v4().to_string());
                tool_call_ids.insert(event_id, id.clone());
                let args_str = args.to_string();
                let tool_call = ToolCall::new(id.clone(), name.to_string())
                    .raw_input(serde_json::from_str(&args_str).ok());
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::ToolCall(tool_call),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send tool call notification: {}", e);
                }
            }
            AgentEvent::SubagentToolCall { name, args } => {
                // Announce-only: subagent calls carry no correlating id, so
                // they never receive a ToolCallUpdate. (Previously they
                // hijacked the single pending slot, so the enclosing `task`
                // call's result got attached to the subagent's entry.)
                // Announced as already Completed, since nothing will ever
                // update it out of the default Pending status.
                let id = ToolCallId::new(uuid::Uuid::new_v4().to_string());
                let args_str = args.to_string();
                let tool_call = ToolCall::new(id.clone(), format!("[subagent] {}", name))
                    .status(ToolCallStatus::Completed)
                    .raw_input(serde_json::from_str(&args_str).ok());
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::ToolCall(tool_call),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send subagent tool call notification: {}", e);
                }
            }
            AgentEvent::ToolResult {
                call_id: event_id,
                output,
                ..
            } => {
                // No announced ToolCall to update: an update carrying a
                // ToolCallId the client was never told about is worse than
                // silence, so drop it.
                let Some(id) = tool_call_ids.remove(&event_id) else {
                    tracing::warn!(
                        "ACP tool result with no announced tool call (id={}); \
                         skipping update",
                        event_id.escape_debug(),
                    );
                    continue;
                };
                let fields = ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .content(vec![ToolCallContent::from(ContentBlock::Text(
                        TextContent::new(output.to_string()),
                    ))]);
                let update = ToolCallUpdate::new(id, fields);
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::ToolCallUpdate(update),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send tool result notification: {}", e);
                }
            }
            AgentEvent::Retrying { attempt, max } => {
                // ACP has no status bar, so surface the retry as an agent
                // thought. This keeps the client from going silent during the
                // backoff delay and mirrors how `Reasoning` is forwarded.
                let text = format!("retrying... ({}/{})", attempt, max);
                let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::AgentThoughtChunk(chunk),
                );
                if let Err(e) = cx.send_notification(notif) {
                    tracing::warn!("ACP failed to send retry notification: {}", e);
                }
            }
            AgentEvent::CompletionCall { .. } => {
                // Mid-stream provider usage; ACP has no status bar to update, so
                // there is nothing to surface for this event.
            }
            AgentEvent::Done { response, .. } => {
                final_response = response.to_string();
                break;
            }
            AgentEvent::Error(err) => {
                // Surface the error to the client instead of silently
                // reporting EndTurn.
                let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(format!(
                    "[error: {}]",
                    err
                ))));
                let notif = SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::AgentMessageChunk(chunk),
                );
                let _ = cx.send_notification(notif);
                let _ = responder.respond(PromptResponse::new(StopReason::Refusal));
                return Ok(());
            }
        }
    }

    // Store assistant response in session history
    if !final_response.is_empty() {
        let mut sessions = state.sessions.lock().await;
        if let Some(sess) = sessions.get_mut(&session_id) {
            sess.record_assistant_response(&final_response);
        }
    }

    let _ = responder.respond(PromptResponse::new(StopReason::EndTurn));
    Ok(())
}

// --- Permission ---

fn build_acp_permission(state: &AcpState) -> (Option<PermCheck>, Option<AskSender>) {
    use std::sync::Mutex as StdMutex;

    let no_tools = state.cli.resolve_no_tools(&state.cfg);
    if no_tools || state.cli.dangerously_skip_permissions {
        return (None, None);
    }

    let perm_config = state.cfg.build_permission_config();

    let mode = resolve_acp_mode(&state.cli, &state.cfg);
    let permission_modes = state.cfg.permission_modes.clone();
    let checker = PermissionChecker::new(&perm_config, mode, None, permission_modes);
    let perm: PermCheck = Arc::new(StdMutex::new(checker));

    let (ask_tx, mut ask_rx) = tokio::sync::mpsc::channel::<crate::permission::ask::AskRequest>(64);
    // ACP is headless — there is no interactive user to prompt. Auto-approve
    // Ask requests so tools don't fail with "Permission system unavailable".
    // Log a warning so the auto-approval is visible in logs.
    tokio::spawn(async move {
        while let Some(req) = ask_rx.recv().await {
            tracing::warn!(
                "ACP auto-approving tool call: tool={}, input_len={}",
                req.tool,
                req.input.len()
            );
            let _ = req
                .reply
                .send(crate::permission::ask::UserDecision::AllowOnce);
        }
    });

    (Some(perm), Some(ask_tx))
}

pub(crate) fn resolve_acp_mode(cli: &Cli, cfg: &Config) -> SecurityMode {
    if cli.dangerously_skip_permissions {
        SecurityMode::Standard
    } else if cli.yolo || cfg.yolo.unwrap_or(false) {
        SecurityMode::Yolo
    } else if cli.accept_all || cfg.accept_all.unwrap_or(false) {
        SecurityMode::Standard
    } else if cli.restrictive || cfg.restrictive.unwrap_or(false) {
        SecurityMode::Restrictive
    } else if let Some(m) = &cfg.default_permission_mode {
        match m.as_str() {
            "yolo" => SecurityMode::Yolo,
            "accept" | "standard" => SecurityMode::Standard,
            "guarded" => SecurityMode::Guarded,
            "readonly" => SecurityMode::ReadOnly,
            "restrictive" => SecurityMode::Restrictive,
            _ => SecurityMode::Standard,
        }
    } else {
        SecurityMode::Standard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_turns_receive_the_earlier_conversation() {
        let mut session = SessionState::new(Vec::new());

        assert!(session.begin_turn("Step 1").is_empty());
        session.record_assistant_response("acknowledged");

        let history = session.begin_turn("Step 2");
        assert_eq!(
            history,
            vec![Message::user("Step 1"), Message::assistant("acknowledged"),]
        );
        session.record_assistant_response("created utils.py");

        let history = session.begin_turn("Step 3");
        assert_eq!(
            history,
            vec![
                Message::user("Step 1"),
                Message::assistant("acknowledged"),
                Message::user("Step 2"),
                Message::assistant("created utils.py"),
            ]
        );
    }

    #[test]
    fn empty_assistant_response_is_not_recorded() {
        let mut session = SessionState::new(Vec::new());

        session.begin_turn("hello");
        session.record_assistant_response("");

        assert_eq!(session.begin_turn("second"), vec![Message::user("hello")]);
    }

    #[test]
    fn config_options_expose_model_and_thought_level() {
        let models = vec![
            AcpModelOption {
                id: "m1".to_string(),
                name: "Model One".to_string(),
                provider: Some("openai".to_string()),
            },
            AcpModelOption {
                id: "m2".to_string(),
                name: "Model Two".to_string(),
                provider: Some("openai".to_string()),
            },
        ];
        let opts = build_config_options(&models, "m1", "medium");
        let json = serde_json::to_value(&opts).expect("serialize configOptions");
        let arr = json.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["id"], "model");
        assert_eq!(arr[0]["category"], "model");
        assert_eq!(arr[0]["currentValue"], "m1");
        assert_eq!(arr[0]["options"].as_array().unwrap().len(), 2);
        assert_eq!(arr[1]["id"], "reasoning_effort");
        assert_eq!(arr[1]["category"], "thought_level");
        assert_eq!(arr[1]["currentValue"], "medium");
        assert_eq!(arr[1]["options"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn apply_reasoning_effort_formats_openai_and_anthropic_params() {
        let cfg = Config::default();
        assert_eq!(apply_reasoning_effort(None, "openai", &cfg, "off"), None);
        let openai_high =
            apply_reasoning_effort(None, "openai", &cfg, "high").expect("openai high");
        assert_eq!(openai_high["reasoning_effort"], "high");

        let anth_high =
            apply_reasoning_effort(None, "anthropic", &cfg, "high").expect("anthropic high");
        assert_eq!(anth_high["thinking"]["type"], "adaptive");
        assert_eq!(anth_high["output_config"]["effort"], "high");

        let anth_off =
            apply_reasoning_effort(None, "anthropic", &cfg, "off").expect("anthropic off");
        assert_eq!(anth_off["thinking"]["type"], "disabled");
    }
}
