const TOKEN_KEY = "medflow_token";

// null means "no conversation started yet" - the backend generates one on
// the first /ask call and this picks it up from the response. Reset
// whenever the employee view (re)opens, so each visit is a fresh
// conversation (no "past conversations" list exists yet).
let conversationId = null;

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

// Every authenticated call goes through this - a 401 means the token is
// missing, expired, or invalid, and in every one of those cases the right
// move is the same: drop back to the login view.
async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    clearToken();
    showLoginView();
  }
  return response;
}

// --- View switching ---

function showLoginView() {
  document.getElementById("app-header").hidden = true;
  document.getElementById("login-view").hidden = false;
  document.getElementById("employee-view").hidden = true;
  document.getElementById("manager-view").hidden = true;
}

function showEmployeeView(user) {
  setHeader(user);
  document.getElementById("login-view").hidden = true;
  document.getElementById("employee-view").hidden = false;
  document.getElementById("manager-view").hidden = true;
  conversationId = null;
}

function showManagerView(user) {
  setHeader(user);
  document.getElementById("login-view").hidden = true;
  document.getElementById("employee-view").hidden = true;
  document.getElementById("manager-view").hidden = false;
  loadApprovals();
}

function setHeader(user) {
  document.getElementById("app-header").hidden = false;
  document.getElementById("user-info").textContent = `${user.name} (${user.role})`;
}

// Asks the backend who's actually logged in - the token alone doesn't
// carry role or name, so this is the only way to know which view to show.
async function routeToCurrentUser() {
  if (!getToken()) {
    showLoginView();
    return;
  }

  const response = await apiFetch("/me");
  if (!response.ok) {
    return;
  }
  const user = await response.json();
  if (user.role === "manager") {
    showManagerView(user);
  } else {
    showEmployeeView(user);
  }
}

// --- Login ---

document.getElementById("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = document.getElementById("login-email").value;
  const password = document.getElementById("login-password").value;
  const errorEl = document.getElementById("login-error");
  errorEl.hidden = true;

  // /login expects OAuth2's password-flow form encoding, not JSON.
  const body = new URLSearchParams();
  body.set("username", email);
  body.set("password", password);

  const response = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!response.ok) {
    errorEl.textContent = "Incorrect email or password.";
    errorEl.hidden = false;
    return;
  }

  const data = await response.json();
  setToken(data.access_token);
  document.getElementById("login-form").reset();
  routeToCurrentUser();
});

document.getElementById("logout-button").addEventListener("click", () => {
  clearToken();
  conversationId = null;
  document.getElementById("chat-log").innerHTML = "";
  showLoginView();
});

// --- Employee chat ---

function appendMessage(text, className) {
  const log = document.getElementById("chat-log");
  const el = document.createElement("div");
  el.className = `message ${className}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;

  appendMessage(question, "user");
  input.value = "";
  const pending = appendMessage("Thinking...", "pending");

  const response = await apiFetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  pending.remove();

  if (!response.ok) {
    if (response.status !== 401) {
      appendMessage("Something went wrong answering that.", "agent");
    }
    return;
  }
  const data = await response.json();
  conversationId = data.conversation_id;
  appendMessage(data.answer, "agent");
});

// --- Manager approvals ---

async function loadApprovals() {
  const list = document.getElementById("approvals-list");
  const empty = document.getElementById("approvals-empty");
  list.innerHTML = "";

  const response = await apiFetch("/approvals");
  if (!response.ok) return;
  const approvals = await response.json();

  empty.hidden = approvals.length > 0;
  for (const approval of approvals) {
    list.appendChild(renderApprovalCard(approval));
  }
}

// Built with createElement/textContent throughout, not innerHTML - reason
// is free text an employee typed (extracted by the LLM from their
// question), so it must never be interpreted as HTML.
function renderApprovalCard(approval) {
  const card = document.createElement("div");
  card.className = "approval-card";

  const heading = document.createElement("h3");
  heading.textContent = `Ticket #${approval.ticket_id} → ${approval.requested_priority}`;
  card.appendChild(heading);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `Requested ${new Date(approval.requested_at).toLocaleString()}`;
  card.appendChild(meta);

  const reason = document.createElement("div");
  reason.className = "reason";
  reason.textContent = approval.reason;
  card.appendChild(reason);

  const actions = document.createElement("div");
  actions.className = "approval-actions";

  const approveButton = document.createElement("button");
  approveButton.className = "approve-button";
  approveButton.textContent = "Approve";
  approveButton.addEventListener("click", () => decide(approval.id, true, card));

  const rejectButton = document.createElement("button");
  rejectButton.className = "reject-button";
  rejectButton.textContent = "Reject";
  rejectButton.addEventListener("click", () => decide(approval.id, false, card));

  actions.appendChild(approveButton);
  actions.appendChild(rejectButton);
  card.appendChild(actions);

  return card;
}

async function decide(approvalId, approved, card) {
  const response = await apiFetch(`/approvals/${approvalId}/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
  if (response.ok) {
    card.remove();
    const list = document.getElementById("approvals-list");
    document.getElementById("approvals-empty").hidden = list.children.length > 0;
  }
}

// --- Init ---

routeToCurrentUser();
