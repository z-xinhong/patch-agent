const element = (id) => document.getElementById(id);
const timeline = element("timeline");
const welcome = timeline.querySelector(".welcome");
const events = element("events");
const composer = element("composer");
const input = element("message");
const approval = element("approval");
const changes = element("changes");
const tools = element("tools");
const contextPanel = element("context");
const workspaceForm = element("workspace-form");
const sessionList = element("sessions");

let active;
let sessions = [];
let defaultWorkspace = "";
let currentTab = "changes";
const busy = new Set();
let listRevision = 0;
let refreshing = false;
const drafts = new Map();

function requestKey(session, action) {
  return `${session?.id}:${action}`;
}

function inProgress(status) {
  return status === "running" || status === "stopping";
}

function node(tag, className, text = "") {
  const result = document.createElement(tag);
  if (className) result.className = className;
  result.textContent = text;
  return result;
}

async function api(path, data) {
  const options = data === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  };
  const response = await fetch("/api/sessions" + path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function showError(error) {
  element("error").textContent = error ? error.message : "";
  element("error").hidden = !error;
}

function setConnection(online) {
  element("connection").textContent = online ? "本地已连接" : "本地服务未连接";
  element("connection").parentElement.classList.toggle("offline", !online);
}

function renderSessions() {
  sessionList.replaceChildren();
  const groups = new Map();
  for (const session of [...sessions].reverse()) {
    if (!groups.has(session.workspace)) groups.set(session.workspace, []);
    groups.get(session.workspace).push(session);
  }
  const workspaces = [...groups.keys()];
  const index = workspaces.indexOf(defaultWorkspace);
  if (index > 0) workspaces.unshift(...workspaces.splice(index, 1));
  for (const workspace of workspaces) {
    const group = node("section", "workspace-sessions");
    const name = node("h3", "workspace-heading", workspace.split(/[/\\]/).filter(Boolean).at(-1) || workspace);
    name.title = workspace;
    group.append(name);
    for (const session of groups.get(workspace)) {
      const button = node("button", "session");
      button.type = "button";
      button.title = session.title;
      button.setAttribute("aria-current", String(active?.id === session.id));
      const label = {
        running: session.needs_approval ? "待确认" : "运行中",
        stopping: "正在停止", interrupted: "上次中断", cancelled: "已停止",
      }[session.status] || session.id.slice(0, 8);
      button.append(node("strong", "", session.title), node("span", inProgress(session.status) ? "busy-session" : "", label));
      button.addEventListener("click", () => selectSession(session));
      group.append(button);
    }
    sessionList.append(group);
  }
}

function selectSession(session) {
  if (active?.id === session.id) return;
  if (active) drafts.set(active.id, input.value);
  active = {...session, contextSettings: session.context_settings, cursor: 0, status: "loading", pending: null};
  sessionStorage.setItem("mini-workbench-session", session.id);
  input.value = drafts.get(session.id) || "";
  events.replaceChildren();
  tools.replaceChildren(node("p", "empty", "执行后的命令和退出码会显示在这里。"));
  contextPanel.replaceChildren(node("p", "empty", "模型请求的预算和实际用量会显示在这里。"));
  element("verification-state").hidden = true;
  showResume(null);
  selectTab(currentTab);
  changes.replaceChildren(node("p", "empty", "正在读取文件改动…"));
  welcome.hidden = false;
  element("welcome-title").textContent = "想在这个项目里做什么？";
  element("welcome-text").textContent = "描述要修改的功能，或从查看目录和运行测试开始。";
  element("suggestions").hidden = false;
  element("title").textContent = session.title;
  element("current-workspace").textContent = session.workspace;
  element("current-workspace").title = session.workspace;
  element("calls").textContent = "0 次调用";
  showError();
  renderSessions();
  renderState();
  refreshChanges();
  updateSession();
}

function renderState() {
  const state = element("session-state");
  const current = active?.workspaceAvailable === false ? "missing"
    : active?.pending ? "approval" : active?.status;
  state.dataset.state = current || "";
  state.textContent = ({
    loading: "加载中", ready: "可继续", running: "运行中", approval: "待确认", error: "会话异常",
    interrupted: "上次中断 · 可续聊", missing: "工作区不可用",
    stopping: "正在停止", cancelled: "已停止 · 可续聊",
  })[current] || "待创建";
  const running = inProgress(active?.status);
  const sending = busy.has(requestKey(active, "send"));
  const deleting = busy.has(requestKey(active, "delete"));
  const applying = busy.has(requestKey(active, "context"));
  const ready = ["ready", "interrupted", "cancelled"].includes(current);
  input.disabled = !ready || sending || deleting || applying;
  element("send").hidden = running;
  element("send").disabled = input.disabled || !input.value.trim();
  element("stop-turn").hidden = !running;
  element("stop-turn").disabled = !active?.turnId || active.status === "stopping" || busy.has(requestKey(active, "stop"));
  element("stop-turn").textContent = active?.status === "stopping" ? "正在停止…" : "停止本轮";
  element("delete-session").hidden = !active;
  element("delete-session").disabled = running || sending || deleting || applying;
  element("delete-session").title = running ? "请先停止本轮" : "";
  element("refresh-changes").disabled = !active;
  approval.hidden = !active?.pending;
  element("approval-title").textContent = ({
    preflight: "确认环境预检", verification: "确认提交验证",
  })[active?.pending?.purpose] || "等待命令确认";
  element("approval-commands").textContent = active?.pending?.commands.join("\n\n") || "";
  const deciding = busy.has(requestKey(active, "approval"));
  element("approve").disabled = deciding;
  element("reject").disabled = deciding;
  renderContextSettings();
}

function commandCard(event) {
  const card = node("details");
  const heading = node("summary");
  const passed = event.returncode === 0;
  heading.append(node("strong", passed ? "exit-ok" : "exit-fail", `[${event.returncode}] `));
  heading.append(document.createTextNode(event.command.replaceAll("\n", " ")));
  card.append(heading, node("pre", "", [
    event.recall_heading || event.command, event.text || "（无输出）", event.error, event.truncated ? "（输出预览已截断）" : "",
  ].filter(Boolean).join("\n\n")));
  const evidence = event.recalled || event.evidence;
  if (evidence) {
    const session = active;
    const form = node("form", "evidence-form");
    const field = node("input");
    field.type = "number";
    field.min = "1";
    field.value = String(event.start_line ?? 1);
    field.setAttribute("aria-label", "起始行");
    const button = node("button", "quiet", "回读 " + evidence);
    button.type = "submit";
    const raw = node("pre");
    raw.hidden = true;
    form.append(field, button);
    form.addEventListener("submit", async (eventObject) => {
      eventObject.preventDefault();
      try {
        const result = await api(`/${session.id}/evidence/${evidence}?start=${Number(field.value)}`);
        raw.textContent = result.output || "（没有更多行）";
      } catch (error) {
        raw.textContent = error.message;
      }
      raw.hidden = false;
    });
    card.append(form, raw);
  }
  return card;
}

const verificationLabels = {
  unconfigured: "未配置", not_run: "待验证", passed: "通过",
  failed: "失败", stale: "结果已过期", declined: "已拒绝",
};

const actionLabels = {
  awaiting_approval: "等待确认", approved: "已批准，未记录开始",
  started: "已开始", unknown: "结果未知", completed: "已执行",
  failed: "执行失败", declined: "已拒绝", cancelled: "已取消", skipped: "未执行",
};

function showResume(resume) {
  element("resume-card").hidden = !resume;
  if (!resume) return;
  element("resume-goal").textContent = `上次需求 · ${resume.goal}`;
  const action = resume.action;
  const actionText = action
    ? `${actionLabels[action.state]} · ${action.command}`
    : "这次旧记录没有可确认的单条执行状态。";
  element("resume-action").textContent = [actionText, ...(resume.warnings || [])].join(" · ");
}

function showVerification(check, backend) {
  if (!check) return;
  const target = element("verification-state");
  target.hidden = false;
  target.dataset.result = check.status;
  const signature = JSON.stringify([backend, check.status, check.evidence, check.version]);
  if (target.dataset.signature === signature && target.dataset.session === active?.id) return;
  target.dataset.signature = signature;
  target.dataset.session = active?.id || "";
  target.replaceChildren(node("p", "",
    `${backend === "docker" ? "Docker 隔离" : "本地执行"} · 验证${verificationLabels[check.status] || check.status}`));
  if (!check.evidence || !active) return;
  const session = active;
  const details = node("details", "verification-details");
  details.append(node("summary", "", `验证输出 ${check.evidence}`));
  details.append(node("p", "", [check.command, `退出码 ${check.returncode}`].join(" · ")));
  if (check.version) details.append(node("p", "", `工作区版本 ${check.version}`));
  const output = node("pre");
  details.append(output);
  details.addEventListener("toggle", async () => {
    if (!details.open || output.textContent) return;
    try {
      const evidence = await api(`/${session.id}/evidence/${check.evidence}`);
      output.textContent = evidence.output || "（无输出）";
    } catch (error) {
      output.textContent = error.message;
    }
  });
  target.append(details);
}

const tokenLabels = {
  task: "任务", project_rules: "项目规则", instructions: "系统指令",
  history: "对话历史", observations: "工具观察", tools: "工具定义", summary: "旧任务摘要",
  background: "稳定背景", working_memory: "任务状态", relevant_memory: "相关旧记忆", current_request: "当前请求",
};

function appendContext(event) {
  const session = active;
  const number = (value) => Number.isInteger(value) ? value.toLocaleString() : "未提供";
  if (event.kind === "evidence_search") {
    contextPanel.append(node("p", "context-line", `搜索证据 ${event.query} · ${event.hits} 条结果`));
    return;
  }
  if (event.kind === "recall") {
    session.recalls = (session.recalls || 0) + 1;
    contextPanel.append(node("p", "context-line", `回读 ${event.evidence} · ${event.characters} 字符`));
    return;
  }
  if (event.kind === "summary") {
    const card = node("details");
    const saved = event.status === "saved";
    const skipped = event.status === "skipped";
    const title = saved ? `已摘要 ${event.tasks.length} 条旧需求` :
      skipped ? "旧需求较短，跳过摘要" :
      event.error === "The summary did not reduce the request." ? "摘要未节省空间，保留原文" : "摘要失败，保留原文";
    card.append(node("summary", "", title));
    const detail = skipped
      ? `旧需求估算 ${number(event.input_tokens)} token，达到 ${number(event.min_tokens)} 才调用摘要模型。`
      : saved ? event.progress : event.error;
    card.append(node("p", "context-line", detail));
    for (const constraint of event.constraints || []) {
      card.append(node("p", "context-line", `保留的用户约束 · ${constraint.quote}`));
    }
    contextPanel.append(card);
    return;
  }
  if (event.kind === "usage") {
    const stats = session.usage ??= {actual: 0, cached: 0, uncached: 0, missing: 0, missingCache: 0, summaries: 0};
    if (Number.isInteger(event.input)) stats.actual += event.input;
    else stats.missing++;
    if (Number.isInteger(event.cached) && Number.isInteger(event.uncached)) {
      stats.cached += event.cached;
      stats.uncached += event.uncached;
    } else stats.missingCache++;
    const totals = [`累计输入 ${number(stats.actual)}`, `命中 ${number(stats.cached)}`, `未命中 ${number(stats.uncached)}`];
    if (stats.missing || stats.missingCache) totals.push("部分用量未提供");
    if (event.purpose === "summary") stats.summaries++;
    if (stats.summaries) totals.push(`摘要调用 ${stats.summaries} 次`);
    totals.push(`回读 ${session.recalls || 0} 次`);
    setContextTotals(totals.join(" · "));
    const card = contextPanel.querySelector(`[data-request="${event.request}"]`);
    card?.append(node("p", "context-line",
      `实际 ${number(event.input)} · 命中 ${number(event.cached)} · 未命中 ${number(event.uncached)}`));
    return;
  }
  if (!contextPanel.querySelector(".context-totals")) setContextTotals("等待供应商用量");
  const card = node("details");
  card.dataset.request = event.request;
  card.append(node("summary", "", `请求 ${event.request} · ${number(event.estimated)} / ${number(event.limit)}`));
  const categories = Object.entries(event.categories).map(([key, value]) => {
    const budget = event.section_budgets?.[key];
    return `${tokenLabels[key]} ${number(value)}${Number.isInteger(budget) ? ` / ${number(budget)}` : ""}`;
  });
  card.append(node("p", "context-line", categories.join(" · ")));
  card.append(node("p", "context-line",
    `压缩前估算 ${number(event.before)} · 输出预留 ${number(event.output_reserve)} · 缩短 ${event.shortened.length} 条 · 移出 ${event.removed.length} 条`));
  if (event.thresholds) {
    const [soft, hard, summary] = event.thresholds.map(number);
    card.append(node("p", "context-line",
      `缩短阈值 ${soft} · 整组移出阈值 ${hard} · 摘要阈值 ${summary} · 已摘要 ${event.summarized.length} 条需求`));
  }
  if (event.emergency_shortened?.length) {
    card.append(node("p", "context-line",
      `为避免超预算，缩短了 ${event.emergency_shortened.length} 条近期工具结果，已保留回读编号。`));
  }
  if (event.source_checks?.length) {
    const stale = event.source_checks.filter((source) => source.status !== "current");
    card.append(node("p", "context-line", `校验了 ${event.source_checks.length} 条源码证据，其中 ${stale.length} 条待重读。`));
    for (const source of stale) {
      card.append(node("p", "context-line", `${source.evidence} · ${source.path} · ${source.status === "changed" ? "内容已变" : "文件不可核验"}`));
    }
  }
  if (event.blocked) card.append(node("p", "context-line exit-fail", "保留的消息超出预算，未发送此请求。"));
  contextPanel.append(card);
}

function setContextTotals(text) {
  contextPanel.querySelector(".empty")?.remove();
  const totals = contextPanel.querySelector(".context-totals");
  if (totals) totals.textContent = text;
  else contextPanel.prepend(node("p", "context-totals", text));
}

function appendEvent(event) {
  if (["context", "usage", "recall", "summary", "evidence_search"].includes(event.kind)) {
    appendContext(event);
    return;
  }
  welcome.hidden = true;
  if (event.kind === "tool") {
    const listPlaceholder = tools.querySelector(".empty");
    if (listPlaceholder) listPlaceholder.remove();
    tools.append(commandCard(event));
    events.append(commandCard(event));
    if (currentTab === "tools") element("inspector-label").textContent = `${tools.querySelectorAll("details").length} 条命令`;
    return;
  }
  const entry = node("article", "entry entry-" + event.kind);
  const label = ({
    user: "你", assistant: "Agent", notice: "运行提示", error: "运行出错",
    verification: "提交验证 · " + (verificationLabels[event.status] || event.status),
    finish: {Submitted: "本轮已提交", Cancelled: "本轮已停止"}[event.reason]
      || `本轮结束 · ${event.reason || "未知状态"}`,
  })[event.kind];
  if (!label) return;
  entry.append(node("h2", "entry-label", label));
  if (event.text) {
    if (event.html && ["assistant", "finish"].includes(event.kind)) {
      const markdown = node("div", "markdown");
      markdown.innerHTML = event.html;
      entry.append(markdown);
    } else {
      entry.append(node("p", "", event.text));
    }
  }
  if (event.kind === "verification") entry.append(node("p", "", `${event.command} · [${event.returncode}] · ${event.evidence}`));
  events.append(entry);
}

async function updateSession() {
  const session = active;
  if (!session || session.polling) return;
  session.polling = true;
  try {
    let data;
    do {
      data = await api(`/${session.id}?since=${session.cursor}`);
      if (session !== active) return;
      const atEnd = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 90;
      session.cursor = data.cursor;
      const needsDiff = data.events.some((event) => event.kind === "finish" || event.kind === "error")
        || (inProgress(session.status) && !inProgress(data.status));
      const listed = sessions.find((entry) => entry.id === session.id);
      element("title").textContent = data.title;
      if (listed && (listed.title !== data.title || listed.status !== data.status
          || listed.needs_approval !== data.needs_approval)) {
        Object.assign(listed, {title: data.title, status: data.status, needs_approval: data.needs_approval});
        renderSessions();
      }
      Object.assign(session, {
        status: data.status, pending: data.pending, turnId: data.turn_id,
        backend: data.backend, workspaceAvailable: data.workspace_available,
        contextStrategy: data.context_strategy, contextSettings: data.context_settings, archive: data.archive,
      });
      showResume(data.resume);
      if (!session.verification || session.verification.evidence !== data.verification.evidence
          || data.verification.status !== "passed") {
        session.verification = data.verification;
        showVerification(session.verification, data.backend);
      }
      if (currentTab === "context") updateContextLabel();
      for (const event of data.events) appendEvent(event);
      if (atEnd && data.events.length) timeline.scrollTop = timeline.scrollHeight;
      const model = element("model");
      model.textContent = data.model;
      model.title = data.model;
      element("calls").textContent = `${data.calls} 次调用`;
      if (needsDiff) refreshChanges();
      setConnection(true);
      renderState();
    } while (data.more);
  } catch (error) {
    setConnection(false);
    if (session === active) showError(error);
  } finally {
    session.polling = false;
  }
}

function diffCard(file) {
  const card = node("details");
  const labels = {added: "新增", deleted: "删除", modified: "修改"};
  const suffix = file.truncated ? "\n（diff 预览已截断）" : "";
  const preview = node("pre");
  for (const line of (file.diff || (file.diff === "" ? "文件标记或行尾发生变化" : "没有可显示的文本 diff")).split("\n")) {
    const className = line.startsWith("@@") ? "diff-hunk"
      : line.startsWith("+") ? "diff-add" : line.startsWith("-") ? "diff-delete" : "";
    preview.append(node("span", className, line + "\n"));
  }
  if (suffix) preview.append(document.createTextNode(suffix));
  card.append(node("summary", "", `${labels[file.status]} · ${file.path}`), preview);
  return card;
}

async function refreshChanges() {
  const session = active;
  if (!session) return;
  const request = (session.diffRequest || 0) + 1;
  session.diffRequest = request;
  try {
    const data = await api(`/${session.id}/changes`);
    if (active !== session || session.diffRequest !== request) return;
    session.verification = data.verification;
    showVerification(data.verification, session.backend);
    if (currentTab !== "changes") return;
    const expanded = new Set([...changes.querySelectorAll("details[open] summary")].map((summary) => summary.textContent));
    changes.replaceChildren();
    if (!data.files.length) changes.append(node("p", "empty", "暂无文件改动。"));
    for (const file of data.files) {
      const card = diffCard(file);
      card.open = expanded.has(card.firstElementChild.textContent);
      changes.append(card);
    }
    if (data.limited) changes.append(node("p", "empty", "部分文件超出工作台预览范围。"));
  } catch (error) {
    if (active === session) changes.replaceChildren(node("p", "empty", error.message));
  }
}

const contextStrategies = {original: "完整历史", window: "近期窗口", evidence: "证据回读"};
const formatThreshold = new Intl.NumberFormat("zh-CN", {style: "percent", maximumFractionDigits: 3});

function budgetText(config) {
  return (config.context_window - config.output_reserve).toLocaleString() + " 输入 token";
}

function thresholdText(config) {
  return [config.soft_threshold, config.hard_threshold, config.summary_threshold]
    .map((value) => formatThreshold.format(value)).join(" / ");
}

function renderContextSettings() {
  const settings = active?.contextSettings;
  element("context-settings").hidden = currentTab !== "context" || !settings;
  if (!settings || currentTab !== "context") return;
  const {effective, startup, differs_from_startup: differs} = settings;
  element("effective-budget").textContent =
    `当前预算 ${budgetText(effective)} · 输出预留 ${effective.output_reserve.toLocaleString()}`;
  element("effective-thresholds").textContent = `缩短 / 移出 / 摘要 · ${thresholdText(effective)}`;
  element("context-startup").hidden = !differs;
  element("startup-budget").textContent =
    `启动配置 · ${contextStrategies[startup.strategy]} · ${budgetText(startup)} · ${thresholdText(startup)}`;
  const running = inProgress(active.status);
  const button = element("apply-context");
  button.disabled = running || busy.has(requestKey(active, "context")) ||
    busy.has(requestKey(active, "send")) || busy.has(requestKey(active, "delete"));
  button.title = running ? "可以在本轮结束后应用" : "";
  const feedback = element("context-feedback");
  feedback.hidden = !active.contextApplied;
  feedback.textContent = active.contextApplied ? "已应用启动配置。" : "";
}

function updateContextLabel() {
  const label = element("inspector-label");
  label.textContent = contextStrategies[active?.contextStrategy] || "当前模型未启用预算";
  label.title = active?.archive || "";
}

function selectTab(tab) {
  currentTab = tab;
  for (const id of ["changes", "tools", "context"]) {
    element(id + "-tab").setAttribute("aria-pressed", id === tab);
    element(id).hidden = id !== tab;
  }
  element("refresh-changes").hidden = tab === "tools";
  if (tab === "context") updateContextLabel();
  else element("inspector-label").textContent =
    tab === "changes" ? "会话开始以来" : `${tools.querySelectorAll("details").length} 条命令`;
  renderContextSettings();
  if (tab === "changes") refreshChanges();
}

workspaceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = workspaceForm.querySelector("button");
  button.disabled = true;
  showError();
  try {
    const session = await api("", {workspace: element("workspace").value});
    listRevision++;
    sessions.push(session);
    workspaceForm.hidden = true;
    selectSession(session);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
});

element("new-session").addEventListener("click", () => {
  workspaceForm.hidden = !workspaceForm.hidden;
  if (!workspaceForm.hidden) {
    element("workspace").value = active?.workspace || defaultWorkspace;
    element("workspace").focus();
  }
});

async function sessionRequest(session, action, data, finish) {
  const key = requestKey(session, action);
  if (busy.has(key)) return;
  busy.add(key);
  showError();
  renderState();
  try {
    finish(await api(`/${session.id}/${action === "send" ? "messages" : action}`, data));
  } catch (error) {
    if (active?.id === session.id) showError(error);
  } finally {
    busy.delete(key);
    if (active?.id === session.id) updateSession();
    renderState();
  }
}

element("apply-context").addEventListener("click", () => {
  const session = active;
  if (!session?.contextSettings?.differs_from_startup || inProgress(session.status)) return;
  sessionRequest(session, "context", {}, (result) => {
    if (active?.id !== session.id) return;
    active.contextSettings = result.context_settings;
    active.contextStrategy = result.context_settings.effective.strategy;
    active.contextApplied = true;
    updateContextLabel();
  });
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const session = active;
  const message = input.value.trim();
  if (!message || busy.has(requestKey(session, "context"))
      || !["ready", "interrupted", "cancelled"].includes(session?.status)
      || session.workspaceAvailable === false) return;
  await sessionRequest(session, "send", {message}, () => {
    drafts.delete(session.id);
    if (active?.id === session.id) {
      input.value = "";
      if (["ready", "interrupted", "cancelled"].includes(active.status)) active.status = "running";
    }
  });
});

async function decide(approved) {
  const session = active;
  if (!session?.pending) return;
  const id = session.pending.id;
  await sessionRequest(session, "approval", {id, approved}, () => {
    if (active?.id === session.id && active.pending?.id === id) active.pending = null;
  });
}

element("stop-turn").addEventListener("click", () => {
  const session = active;
  if (!session?.turnId) return;
  sessionRequest(session, "stop", {turn_id: session.turnId}, () => {
    if (active?.id === session.id && active.status === "running") {
      active.status = "stopping";
      active.pending = null;
    }
  });
});

function clearSession() {
  active = undefined;
  sessionStorage.removeItem("mini-workbench-session");
  input.value = "";
  events.replaceChildren();
  changes.replaceChildren(node("p", "empty", "会话中的文件改动会显示在这里。"));
  tools.replaceChildren(node("p", "empty", "执行后的命令和退出码会显示在这里。"));
  contextPanel.replaceChildren(node("p", "empty", "模型请求的预算和实际用量会显示在这里。"));
  element("verification-state").hidden = true;
  showResume(null);
  element("calls").textContent = "0 次调用";
  element("model").textContent = "未选择模型";
  element("title").textContent = "工作台";
  element("current-workspace").textContent = "选择一个工作区开始";
  element("current-workspace").removeAttribute("title");
  element("welcome-title").textContent = "从一个工作区开始";
  element("welcome-text").textContent = "在左侧创建会话，然后告诉 Agent 你想做什么。";
  element("suggestions").hidden = true;
  welcome.hidden = false;
  element("workspace").value = defaultWorkspace;
  workspaceForm.hidden = false;
  selectTab(currentTab);
  renderSessions();
  renderState();
}

element("delete-session").addEventListener("click", () => {
  const session = active;
  if (!session || inProgress(session.status)) return;
  if (!window.confirm("永久删除这条会话的聊天、日志和证据？工作区里的文件改动会保留。")) return;
  sessionRequest(session, "delete", {}, () => {
    listRevision++;
    sessions = sessions.filter((entry) => entry.id !== session.id);
    if (active?.id === session.id) {
      const next = sessions.filter((entry) => entry.workspace === session.workspace).at(-1);
      if (next) selectSession(next);
      else clearSession();
    }
    drafts.delete(session.id);
    renderSessions();
  });
});

input.addEventListener("input", renderState);
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
element("approve").addEventListener("click", () => decide(true));
element("reject").addEventListener("click", () => decide(false));
element("changes-tab").addEventListener("click", () => selectTab("changes"));
element("resume-diff").addEventListener("click", () => selectTab("changes"));
element("tools-tab").addEventListener("click", () => selectTab("tools"));
element("context-tab").addEventListener("click", () => selectTab("context"));
element("refresh-changes").addEventListener("click", refreshChanges);
for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    renderState();
    input.focus();
  });
}

async function initialize() {
  if (refreshing) return;
  refreshing = true;
  try {
    const data = await api("");
    defaultWorkspace = data.workspace;
    sessions = data.sessions;
    const id = sessionStorage.getItem("mini-workbench-session");
    const candidates = sessions.filter((entry) => entry.workspace === defaultWorkspace);
    const session = candidates.find((entry) => entry.id === id) || candidates.at(-1);
    renderSessions();
    if (session) {
      selectSession(session);
    } else {
      element("workspace").value = defaultWorkspace;
      workspaceForm.hidden = false;
    }
    setConnection(true);
  } catch (error) {
    setConnection(false);
    showError(error);
  } finally {
    refreshing = false;
  }
}

async function refreshSessionList() {
  const revision = listRevision;
  try {
    const data = await api("");
    if (revision !== listRevision || JSON.stringify(sessions) === JSON.stringify(data.sessions)) return;
    sessions = data.sessions;
    if (active && !sessions.some((entry) => entry.id === active.id)) clearSession();
    else renderSessions();
  } catch {
    // The active session poll reports connection errors.
  }
}

initialize();
window.setInterval(() => { if (defaultWorkspace) refreshSessionList(); }, 1800);
window.setInterval(() => {
  if (active) updateSession();
  else if (!defaultWorkspace) initialize();
}, 900);
