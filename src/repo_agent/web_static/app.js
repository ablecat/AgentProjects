"use strict";

const elements = {
  form: document.querySelector("#run-form"),
  task: document.querySelector("#task"),
  repoPath: document.querySelector("#repo-path"),
  baseRef: document.querySelector("#base-ref"),
  autoApprove: document.querySelector("#auto-approve"),
  allowRemoteModel: document.querySelector("#allow-remote-model"),
  agentSettings: document.querySelector("#agent-settings"),
  demoSettings: document.querySelector("#demo-settings"),
  maxSteps: document.querySelector("#max-steps"),
  timeoutSeconds: document.querySelector("#timeout-seconds"),
  maxOutputBytes: document.querySelector("#max-output-bytes"),
  runButton: document.querySelector("#run-button"),
  runButtonLabel: document.querySelector("#run-button-label"),
  providerBadge: document.querySelector("#provider-badge"),
  formError: document.querySelector("#form-error"),
  resultPanel: document.querySelector(".result-panel"),
  emptyState: document.querySelector("#empty-state"),
  summaryView: document.querySelector("#summary-view"),
  jsonView: document.querySelector("#json-view"),
  summaryTab: document.querySelector("#summary-tab"),
  jsonTab: document.querySelector("#json-tab"),
  jsonOutput: document.querySelector("#json-output"),
  copyJson: document.querySelector("#copy-json"),
  downloadJson: document.querySelector("#download-json"),
  runStatus: document.querySelector("#run-status"),
  runStatusLabel: document.querySelector("#run-status-label"),
  runStatusTask: document.querySelector("#run-status-task"),
  workflowActions: document.querySelector("#workflow-actions"),
  approveRun: document.querySelector("#approve-run"),
  rejectRun: document.querySelector("#reject-run"),
  resumeRun: document.querySelector("#resume-run"),
  cancelRun: document.querySelector("#cancel-run"),
  metricNode: document.querySelector("#metric-node"),
  metricTools: document.querySelector("#metric-tools"),
  metricDuration: document.querySelector("#metric-duration"),
  metricTokens: document.querySelector("#metric-tokens"),
  planSection: document.querySelector("#plan-section"),
  planGoal: document.querySelector("#plan-goal"),
  planFileCount: document.querySelector("#plan-file-count"),
  planFiles: document.querySelector("#plan-files"),
  planSteps: document.querySelector("#plan-steps"),
  planChecks: document.querySelector("#plan-checks"),
  planRisks: document.querySelector("#plan-risks"),
  answerText: document.querySelector("#answer-text"),
  runError: document.querySelector("#run-error"),
  toolSummary: document.querySelector("#tool-summary"),
  toolList: document.querySelector("#tool-list"),
  artifactSection: document.querySelector("#artifact-section"),
  artifactLinks: document.querySelector("#artifact-links"),
  runList: document.querySelector("#run-list"),
  queueSummary: document.querySelector("#queue-summary"),
  refreshState: document.querySelector("#refresh-state"),
  headerStatusDot: document.querySelector("#header-status-dot"),
  headerStatusText: document.querySelector("#header-status-text"),
  serviceStatus: document.querySelector("#service-status"),
  serviceQueue: document.querySelector("#service-queue"),
  serviceModel: document.querySelector("#service-model"),
  repoBranch: document.querySelector("#repo-branch"),
  repoHead: document.querySelector("#repo-head"),
  repoDirty: document.querySelector("#repo-dirty"),
  commitNotice: document.querySelector("#commit-notice"),
  dockerStatus: document.querySelector("#docker-status"),
  dockerContext: document.querySelector("#docker-context"),
  dockerServer: document.querySelector("#docker-server"),
  imageList: document.querySelector("#image-list"),
  policyGrid: document.querySelector("#policy-grid"),
  checkProfileList: document.querySelector("#check-profile-list"),
  checkBootstrap: document.querySelector("#check-bootstrap"),
  checkVerify: document.querySelector("#check-verify"),
  checkTimeout: document.querySelector("#check-timeout"),
};

const statusPresentation = {
  queued: { label: "Queued", tone: "neutral" },
  planning: { label: "Planning", tone: "neutral" },
  awaiting_approval: { label: "Awaiting approval", tone: "warning" },
  running: { label: "Running", tone: "neutral" },
  interrupted: { label: "Interrupted", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "success" },
  unverified: { label: "Unverified", tone: "warning" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "warning" },
  policy_denied: { label: "Policy denied", tone: "danger" },
  rejected: { label: "Rejected", tone: "warning" },
  completed: { label: "Completed", tone: "success" },
  completed_with_errors: { label: "Completed with errors", tone: "warning" },
  max_steps: { label: "Step limit reached", tone: "warning" },
  repeated_call: { label: "Repeated call blocked", tone: "warning" },
  provider_error: { label: "Provider error", tone: "danger" },
  setup_error: { label: "Setup error", tone: "danger" },
};

const terminalStatuses = new Set([
  "succeeded", "unverified", "failed", "cancelled", "policy_denied", "rejected",
]);
let currentResult = null;
let currentMode = "agent";
let currentRunId = null;
let pollTimer = null;

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`Server returned an unreadable response (${response.status})`);
  }
  return { response, payload };
}

async function loadState() {
  elements.refreshState.disabled = true;
  elements.refreshState.classList.add("is-loading");
  try {
    const { response, payload } = await fetchJson("/api/state");
    if (!response.ok) throw new Error(apiErrorMessage(payload, `State request failed (${response.status})`));
    renderState(payload);
    if (payload.service && payload.service.enabled) await loadRuns();
  } catch (error) {
    elements.headerStatusText.textContent = "Server unavailable";
    elements.headerStatusDot.dataset.tone = "danger";
    elements.dockerStatus.textContent = errorMessage(error);
  } finally {
    elements.refreshState.disabled = false;
    elements.refreshState.classList.remove("is-loading");
  }
}

function renderState(state) {
  const repository = state.repository || {};
  elements.repoPath.value = repository.path || "Unavailable";
  elements.repoBranch.textContent = repository.branch || "-";
  elements.repoHead.textContent = repository.head || "-";
  const dirtyCount = Number.isInteger(repository.dirty_count) ? repository.dirty_count : null;
  elements.repoDirty.textContent = dirtyCount === null ? "-" : String(dirtyCount);
  elements.commitNotice.hidden = !dirtyCount;
  elements.commitNotice.textContent = dirtyCount ? `${dirtyCount} uncommitted item${dirtyCount === 1 ? "" : "s"}; durable runs require a clean worktree.` : "";

  const service = state.service || {};
  elements.serviceStatus.textContent = service.ready ? "Ready" : service.enabled ? "Unavailable" : "Disabled";
  elements.serviceQueue.textContent = Number.isInteger(service.queue_depth) ? String(service.queue_depth) : "-";
  elements.serviceModel.textContent = service.model_configured ? "Configured" : "Not configured";

  const docker = state.docker || {};
  elements.headerStatusText.textContent = !service.ready ? "Agent unavailable" : !service.model_configured ? "Model not configured" : docker.available ? "Agent ready" : "Docker offline";
  elements.headerStatusDot.dataset.tone = service.ready && service.model_configured && docker.available ? "success" : "danger";
  elements.dockerStatus.textContent = docker.available ? "Ready" : "Unavailable";
  elements.dockerContext.textContent = docker.context || "-";
  const server = docker.server || {};
  elements.dockerServer.textContent = docker.available ? `${server.os || "?"}/${server.arch || "?"} ${server.version || ""}`.trim() : "-";
  renderImages(docker.images || []);
  renderPolicy(state.policy || {});
  renderCheckContract(state.checks || {}, docker.images || []);
}

function renderImages(images) {
  elements.imageList.replaceChildren();
  for (const image of images) {
    const row = document.createElement("div"); row.className = "image-row";
    const indicator = document.createElement("span"); indicator.className = "image-indicator"; indicator.dataset.tone = image.present ? "success" : "neutral"; indicator.setAttribute("aria-hidden", "true");
    const copy = document.createElement("div"); copy.className = "image-copy";
    const label = document.createElement("strong"); label.textContent = `${image.label || image.tag} - ${image.present ? "ready" : "missing"}`;
    const tag = document.createElement("span"); const shortId = typeof image.id === "string" ? image.id.slice(7, 19) : ""; tag.textContent = shortId ? `${image.tag || "unknown"} / ${shortId}` : image.tag || "unknown";
    copy.append(label, tag); row.append(indicator, copy); elements.imageList.append(row);
  }
}

function renderPolicy(policy) {
  const labels = { snapshot: "Snapshot", mutations: "Mutations", network: "Network", rootfs: "Root FS", user: "User", cpus: "CPU", memory: "Memory", pids: "PIDs", capabilities: "Caps", security: "Security", logs: "Logs" };
  elements.policyGrid.replaceChildren();
  for (const [key, label] of Object.entries(labels)) {
    const row = document.createElement("div"); const term = document.createElement("dt"); const value = document.createElement("dd");
    term.textContent = label; value.textContent = policy[key] === undefined ? "-" : String(policy[key]); row.append(term, value); elements.policyGrid.append(row);
  }
}

function renderCheckContract(checks, images) {
  const readiness = new Map(images.map((image) => [image.tag, Boolean(image.present)]));
  elements.checkProfileList.replaceChildren();
  for (const profile of checks.profiles || []) {
    const row = document.createElement("div"); row.className = "check-profile";
    const indicator = document.createElement("span"); indicator.className = "image-indicator"; const ready = readiness.get(profile.image) === true; indicator.dataset.tone = ready ? "success" : "neutral";
    const copy = document.createElement("div"); const label = document.createElement("strong"); const identifier = document.createElement("span");
    label.textContent = profile.label || profile.id || "Unknown profile"; identifier.textContent = `${profile.id || "unknown"} / ${ready ? "ready" : "missing image"}`;
    copy.append(label, identifier); row.append(indicator, copy); elements.checkProfileList.append(row);
  }
  const bootstrap = checks.bootstrap || {}; const verify = checks.verify || {}; const limits = checks.limits || {};
  elements.checkBootstrap.textContent = bootstrap.authorization && bootstrap.network ? `${bootstrap.authorization} / ${bootstrap.network}` : "-";
  elements.checkVerify.textContent = verify.network && verify.workspace ? `${verify.network} / ${verify.workspace}` : "-";
  elements.checkTimeout.textContent = Number.isInteger(limits.phase_seconds) && Number.isInteger(limits.run_seconds) ? `${limits.phase_seconds}s phase / ${limits.run_seconds}s run` : "-";
}

function switchMode(mode) {
  currentMode = mode;
  elements.agentSettings.hidden = mode !== "agent";
  elements.demoSettings.hidden = mode !== "demo";
  elements.providerBadge.textContent = mode === "agent" ? "OpenAI" : "Demo provider";
  elements.runButtonLabel.textContent = mode === "agent" ? "Create maintenance run" : "Run inspection";
  elements.task.value = mode === "agent" ? "Fix the reported bug and add a focused regression test" : "Inspect repository status and TODO markers";
}

function selectedImage() {
  const selected = document.querySelector('input[name="image"]:checked');
  return selected ? selected.value : "repo-agent-python:0.1";
}

async function submitRun(event) {
  event.preventDefault();
  elements.formError.hidden = true;
  const task = elements.task.value.trim();
  if (!task) { showFormError("Task must not be empty."); elements.task.focus(); return; }
  if (currentMode === "demo") await runInspection(task); else await createAgentRun(task);
}

async function createAgentRun(task) {
  setSubmitting(true);
  const baseRef = elements.baseRef.value.trim();
  const payload = { task, auto_approve: elements.autoApprove.checked, allow_remote_model: elements.allowRemoteModel.checked };
  if (baseRef) payload.base_ref = baseRef;
  try {
    const { response, payload: result } = await fetchJson("/api/agent/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(apiErrorMessage(result, `Run creation failed (${response.status})`));
    currentRunId = result.run_id; renderAgentRun(result); await loadRuns(); schedulePoll(result);
  } catch (error) { showFormError(errorMessage(error)); }
  finally { setSubmitting(false); }
}

async function runInspection(task) {
  setSubmitting(true); renderPendingDemo(task);
  const payload = { task, image: selectedImage(), max_steps: Number(elements.maxSteps.value), timeout_seconds: Number(elements.timeoutSeconds.value), max_output_bytes: Number(elements.maxOutputBytes.value) };
  try {
    const { response, payload: result } = await fetchJson("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (result && typeof result.status === "string") renderDemoResult(result);
    else throw new Error(apiErrorMessage(result, `Inspection failed (${response.status})`));
  } catch (error) { showFormError(errorMessage(error)); }
  finally { setSubmitting(false); await loadState(); }
}

function setSubmitting(active) {
  elements.runButton.disabled = active; elements.task.disabled = active; elements.resultPanel.setAttribute("aria-busy", String(active));
  elements.runButtonLabel.textContent = active ? "Submitting" : currentMode === "agent" ? "Create maintenance run" : "Run inspection";
}

function renderPendingDemo(task) {
  currentRunId = null; currentResult = null; showResultShell();
  elements.runStatus.dataset.tone = "neutral"; elements.runStatusLabel.textContent = "Running"; elements.runStatusTask.textContent = task;
  elements.answerText.textContent = "Waiting for sandbox results..."; elements.toolList.replaceChildren(); elements.toolSummary.textContent = "In progress";
  elements.metricNode.textContent = "demo"; elements.metricTools.textContent = "-"; elements.metricDuration.textContent = "-"; elements.metricTokens.textContent = "0";
  elements.planSection.hidden = true; elements.artifactSection.hidden = true; elements.workflowActions.hidden = true;
}

async function loadRuns() {
  try {
    const { response, payload } = await fetchJson("/api/agent/runs");
    if (!response.ok) throw new Error(apiErrorMessage(payload, `Run list failed (${response.status})`));
    const runs = Array.isArray(payload.runs) ? payload.runs : [];
    renderRunList(runs);
    elements.queueSummary.textContent = `Queue ${Number.isInteger(payload.queue_depth) ? payload.queue_depth : 0}`;
    elements.serviceQueue.textContent = Number.isInteger(payload.queue_depth) ? String(payload.queue_depth) : "-";
    if (!currentRunId && currentMode === "agent" && runs.length) await selectAgentRun(runs[0].run_id);
  } catch (error) {
    elements.runList.replaceChildren(); const message = document.createElement("p"); message.className = "tool-meta"; message.textContent = errorMessage(error); elements.runList.append(message);
  }
}

function renderRunList(runs) {
  elements.runList.replaceChildren();
  if (!runs.length) { const empty = document.createElement("p"); empty.className = "tool-meta"; empty.textContent = "No durable runs"; elements.runList.append(empty); return; }
  for (const run of runs) {
    const button = document.createElement("button"); button.type = "button"; button.className = "run-list-item"; button.dataset.selected = String(run.run_id === currentRunId);
    const row = document.createElement("span"); row.className = "run-list-heading";
    const status = document.createElement("strong"); status.textContent = (statusPresentation[run.status] || { label: run.status }).label;
    const id = document.createElement("code"); id.textContent = String(run.run_id || "").slice(0, 8); row.append(status, id);
    const task = document.createElement("span"); task.className = "run-list-task"; task.textContent = run.task || "";
    button.append(row, task); button.addEventListener("click", () => selectAgentRun(run.run_id)); elements.runList.append(button);
  }
}

async function selectAgentRun(runId) {
  clearPoll(); currentMode = "agent"; currentRunId = runId;
  try {
    const { response, payload } = await fetchJson(`/api/agent/runs/${runId}`);
    if (!response.ok) throw new Error(apiErrorMessage(payload, `Run request failed (${response.status})`));
    renderAgentRun(payload); await loadRuns(); schedulePoll(payload);
  } catch (error) { showFormError(errorMessage(error)); }
}

function schedulePoll(run) {
  clearPoll();
  if (!run || terminalStatuses.has(run.status) || run.status === "awaiting_approval") return;
  pollTimer = window.setTimeout(async () => { await selectAgentRun(run.run_id); }, 900);
}

function clearPoll() { if (pollTimer !== null) { window.clearTimeout(pollTimer); pollTimer = null; } }

function renderAgentRun(run) {
  currentMode = "agent"; currentRunId = run.run_id; currentResult = run; showResultShell();
  const presentation = statusPresentation[run.status] || { label: run.status || "Unknown", tone: "danger" };
  elements.runStatus.dataset.tone = presentation.tone; elements.runStatusLabel.textContent = presentation.label; elements.runStatusTask.textContent = run.task || "";
  const metrics = run.metrics || {};
  elements.metricNode.textContent = run.current_node || "-"; elements.metricTools.textContent = String(metrics.tool_calls ?? 0);
  elements.metricDuration.textContent = Number.isFinite(metrics.duration_ms) ? formatDuration(metrics.duration_ms) : "-"; elements.metricTokens.textContent = String(metrics.tokens ?? 0);
  elements.answerText.textContent = run.summary || statusMessage(run.status); elements.runError.hidden = !run.error; elements.runError.textContent = run.error || "";
  renderPlan(run.plan); renderVerification(Array.isArray(run.checks) ? run.checks : [], run.run_id); renderArtifacts(run.artifacts || {}); renderWorkflowActions(run);
  finishResultRender(run);
}

function renderDemoResult(result) {
  currentResult = result; showResultShell();
  const presentation = statusPresentation[result.status] || { label: result.status || "Unknown", tone: "danger" }; const tools = Array.isArray(result.tool_results) ? result.tool_results : []; const metadata = result.metadata || {};
  elements.runStatus.dataset.tone = presentation.tone; elements.runStatusLabel.textContent = presentation.label; elements.runStatusTask.textContent = result.task || "";
  elements.metricNode.textContent = "demo"; elements.metricTools.textContent = String(tools.length); elements.metricDuration.textContent = Number.isFinite(metadata.duration_ms) ? formatDuration(metadata.duration_ms) : "-"; elements.metricTokens.textContent = "0";
  elements.answerText.textContent = result.answer || "No final answer."; elements.runError.hidden = !result.error; elements.runError.textContent = result.error || "";
  elements.planSection.hidden = true; elements.artifactSection.hidden = true; elements.workflowActions.hidden = true; renderDemoTools(tools); finishResultRender(result);
}

function showResultShell() { elements.emptyState.hidden = true; elements.summaryView.hidden = false; elements.copyJson.disabled = false; elements.downloadJson.disabled = false; elements.jsonTab.disabled = false; selectTab("summary"); }

function finishResultRender(result) { elements.jsonOutput.textContent = JSON.stringify(result, null, 2); elements.copyJson.disabled = false; elements.downloadJson.disabled = false; elements.jsonTab.disabled = false; selectTab("summary"); }

function renderPlan(plan) {
  elements.planSection.hidden = !plan;
  if (!plan) return;
  elements.planGoal.textContent = plan.goal || ""; const files = Array.isArray(plan.files) ? plan.files : [];
  elements.planFileCount.textContent = `${files.length} file${files.length === 1 ? "" : "s"}`;
  renderTextList(elements.planFiles, files); renderTextList(elements.planSteps, Array.isArray(plan.steps) ? plan.steps : []); renderTextList(elements.planChecks, Array.isArray(plan.checks) ? plan.checks : []); renderTextList(elements.planRisks, Array.isArray(plan.risks) && plan.risks.length ? plan.risks : ["None recorded"]);
}

function renderTextList(parent, values) { parent.replaceChildren(); for (const value of values) { const item = document.createElement("li"); item.textContent = String(value); parent.append(item); } }

function renderWorkflowActions(run) {
  const awaiting = run.status === "awaiting_approval"; const interrupted = run.status === "interrupted"; const cancellable = ["queued", "planning", "running", "awaiting_approval"].includes(run.status);
  elements.approveRun.hidden = !awaiting; elements.rejectRun.hidden = !awaiting; elements.resumeRun.hidden = !interrupted; elements.cancelRun.hidden = !cancellable;
  elements.workflowActions.hidden = !awaiting && !interrupted && !cancellable;
}

function renderVerification(checks, runId) {
  elements.toolList.replaceChildren(); const passed = checks.filter((check) => check.ok).length; elements.toolSummary.textContent = `${passed}/${checks.length} passed`;
  if (!checks.length) { const empty = document.createElement("p"); empty.className = "tool-meta"; empty.textContent = "No verification recorded"; elements.toolList.append(empty); return; }
  checks.forEach((check) => {
    const details = document.createElement("details"); details.className = "tool-result"; details.open = !check.ok;
    const summary = document.createElement("summary"); const name = document.createElement("span"); name.className = "tool-name"; name.textContent = `Attempt ${check.attempt}: ${check.check_id || "check"}`;
    const meta = document.createElement("span"); meta.className = "tool-meta"; meta.textContent = formatDuration(check.duration_ms || 0);
    const outcome = document.createElement("span"); outcome.className = "tool-outcome"; outcome.dataset.tone = check.ok ? "success" : "danger"; outcome.textContent = check.status || (check.ok ? "Passed" : "Failed"); summary.append(name, meta, outcome);
    const content = document.createElement("div"); content.className = "tool-content"; const pre = document.createElement("pre"); pre.textContent = check.error || "Verification completed without a recorded error."; content.append(pre);
    if (check.log_artifact_href) { const link = document.createElement("a"); link.className = "inline-link"; link.href = check.log_artifact_href; link.textContent = "Download log"; content.append(link); }
    details.append(summary, content); elements.toolList.append(details);
  });
}

function renderDemoTools(tools) {
  elements.toolList.replaceChildren(); const passed = tools.filter((tool) => tool.ok).length; elements.toolSummary.textContent = `${passed}/${tools.length} passed`;
  if (!tools.length) { const empty = document.createElement("p"); empty.className = "tool-meta"; empty.textContent = "No tool calls recorded"; elements.toolList.append(empty); return; }
  tools.forEach((tool, index) => {
    const details = document.createElement("details"); details.className = "tool-result"; details.open = !tool.ok;
    const summary = document.createElement("summary"); const name = document.createElement("span"); name.className = "tool-name"; name.textContent = `${index + 1}. ${tool.name || "unknown_tool"}`;
    const meta = document.createElement("span"); meta.className = "tool-meta"; meta.textContent = tool.exit_code === null || tool.exit_code === undefined ? "no exit" : `exit ${tool.exit_code}`;
    const outcome = document.createElement("span"); outcome.className = "tool-outcome"; outcome.dataset.tone = tool.ok ? "success" : "danger"; outcome.textContent = tool.ok ? "Passed" : "Failed"; summary.append(name, meta, outcome);
    const content = document.createElement("div"); content.className = "tool-content"; const output = document.createElement("pre"); output.textContent = tool.error || tool.output || "No output."; content.append(output); details.append(summary, content); elements.toolList.append(details);
  });
}

function renderArtifacts(artifacts) {
  elements.artifactLinks.replaceChildren(); const labels = { patch: "Patch", report: "Report", result: "Run JSON", trace: "Trace", checks: "Check logs" };
  for (const [kind, label] of Object.entries(labels)) {
    if (typeof artifacts[kind] !== "string") continue; const link = document.createElement("a"); link.className = "artifact-link"; link.href = artifacts[kind]; link.textContent = label; elements.artifactLinks.append(link);
  }
  elements.artifactSection.hidden = elements.artifactLinks.childElementCount === 0;
}

async function runAction(action, payload = {}) {
  if (!currentRunId) return;
  disableActions(true);
  try {
    const { response, payload: result } = await fetchJson(`/api/agent/runs/${currentRunId}/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(apiErrorMessage(result, `${action} failed (${response.status})`));
    renderAgentRun(result); await loadRuns(); schedulePoll(result);
  } catch (error) { showFormError(errorMessage(error)); }
  finally { disableActions(false); }
}

function disableActions(disabled) { for (const button of elements.workflowActions.querySelectorAll("button")) button.disabled = disabled; }

function statusMessage(status) {
  return { queued: "Waiting for the single workflow worker.", planning: "Inspecting the committed repository snapshot.", awaiting_approval: "Review the generated plan before implementation.", running: "Implementing or verifying the candidate patch.", interrupted: "The saved checkpoint can be resumed.", policy_denied: "The repository or requested operation did not satisfy policy.", unverified: "A candidate exists, but deterministic verification did not pass." }[status] || "No summary recorded.";
}

function selectTab(name) {
  const summarySelected = name === "summary"; elements.summaryTab.setAttribute("aria-selected", String(summarySelected)); elements.jsonTab.setAttribute("aria-selected", String(!summarySelected));
  elements.summaryView.hidden = !summarySelected; elements.jsonView.hidden = summarySelected; if (!summarySelected) elements.emptyState.hidden = true;
}

function showFormError(message) { elements.formError.textContent = message; elements.formError.hidden = false; }
function apiErrorMessage(payload, fallback) { return payload && payload.error && payload.error.message ? payload.error.message : fallback; }
function errorMessage(error) { return error instanceof Error ? error.message : String(error); }
function formatDuration(milliseconds) { return milliseconds < 1000 ? `${milliseconds} ms` : `${(milliseconds / 1000).toFixed(1)} sec`; }

async function copyCurrentJson() {
  if (!currentResult) return;
  try { await navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2)); elements.copyJson.textContent = "Copied"; }
  catch (error) { showFormError("Clipboard access was denied."); }
  finally { window.setTimeout(() => { elements.copyJson.textContent = "Copy JSON"; }, 1200); }
}

function downloadCurrentJson() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" }); const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
  anchor.href = url; anchor.download = currentRunId ? `repo-agent-${currentRunId}.json` : `repo-agent-demo-${Date.now()}.json`; document.body.append(anchor); anchor.click(); anchor.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

elements.form.addEventListener("submit", submitRun);
elements.refreshState.addEventListener("click", loadState);
elements.summaryTab.addEventListener("click", () => selectTab("summary")); elements.jsonTab.addEventListener("click", () => selectTab("json"));
elements.copyJson.addEventListener("click", copyCurrentJson); elements.downloadJson.addEventListener("click", downloadCurrentJson);
elements.approveRun.addEventListener("click", () => runAction("decision", { approve: true })); elements.rejectRun.addEventListener("click", () => runAction("decision", { approve: false }));
elements.resumeRun.addEventListener("click", () => runAction("resume")); elements.cancelRun.addEventListener("click", () => runAction("cancel"));
for (const input of document.querySelectorAll('input[name="mode"]')) input.addEventListener("change", () => switchMode(input.value));
elements.task.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") elements.form.requestSubmit(); });
window.addEventListener("beforeunload", clearPoll);
loadState();
