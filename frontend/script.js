// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------
// Left blank so requests go to whatever origin is serving this page
// (the FastAPI app now serves this frontend directly, see app/main.py).
// If you run the frontend separately (e.g. a live-reload dev server),
// point this at your API instead, e.g. "http://localhost:8000".
const API_BASE_URL = "";

const SESSION_STORAGE_KEY = "ar_session_id";

// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------
let sessionId = getOrCreateSessionId();
let isSending = false;

// ------------------------------------------------------------------
// Elements
// ------------------------------------------------------------------
const chatEl = document.getElementById("chat");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const quickActionsEl = document.getElementById("quick-actions");
const statusEl = document.getElementById("status-indicator");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------
init();

function init() {
  renderWelcomeMessage();
  checkHealth();
  setInterval(checkHealth, 30000);

  composerEl.addEventListener("submit", handleSubmit);
  newChatBtn.addEventListener("click", startNewConversation);
  quickActionsEl.addEventListener("click", handleQuickAction);

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerEl.requestSubmit();
    }
  });

  inputEl.addEventListener("input", autoGrow);
}

// ------------------------------------------------------------------
// Session handling
// ------------------------------------------------------------------
function getOrCreateSessionId() {
  const existing = localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;

  const fresh = createId();
  localStorage.setItem(SESSION_STORAGE_KEY, fresh);
  return fresh;
}

function createId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "sess-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function startNewConversation() {
  sessionId = createId();
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  chatEl.innerHTML = "";
  renderWelcomeMessage();
  inputEl.focus();
}

// ------------------------------------------------------------------
// Sending messages
// ------------------------------------------------------------------
async function handleSubmit(e) {
  e.preventDefault();

  const text = inputEl.value.trim();
  if (!text || isSending) return;

  appendUserMessage(text);
  inputEl.value = "";
  autoGrow();

  const typingRow = appendTypingIndicator();
  setSending(true);

  try {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) {
      const detail = await safeErrorDetail(res);
      throw new Error(detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    typingRow.remove();
    appendAssistantMessage(data);
  } catch (err) {
    typingRow.remove();
    appendSystemMessage(
      "Couldn't reach the support agent. Make sure the backend is running, then try again."
    );
    console.error(err);
  } finally {
    setSending(false);
  }
}

async function safeErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail;
  } catch {
    return null;
  }
}

function handleQuickAction(e) {
  const btn = e.target.closest("button[data-prompt]");
  if (!btn) return;
  inputEl.value = btn.dataset.prompt;
  autoGrow();
  composerEl.requestSubmit();
}

function setSending(value) {
  isSending = value;
  sendBtn.disabled = value || inputEl.value.trim() === "";
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------
function renderWelcomeMessage() {
  appendAssistantMessage({
    answer:
      "Hi, I'm the Aster & Row support assistant. I can help with order status, shipping, returns, and product care — what can I help with today?",
    sources: [],
    handoff: false,
    blocked: false,
  });
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "row user";
  row.innerHTML = `
    <div class="bubble-col">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
  `;
  chatEl.appendChild(row);
  scrollToBottom();
}

function appendAssistantMessage({ answer, sources = [], handoff = false, blocked = false }) {
  const row = document.createElement("div");
  row.className = "row assistant" + (blocked ? " flagged" : "");

  const sourceChips = sources
    .map((s) => `<span class="source-chip">${docIcon()} ${escapeHtml(humanizeSource(s))}</span>`)
    .join("");

  const handoffNote = handoff
    ? `<span class="handoff-note">${flagIcon()} ${
        blocked ? "Flagged for a human teammate to confirm" : "Suggested: connect with a person"
      }</span>`
    : "";

  const metaRow =
    sourceChips || handoffNote
      ? `<div class="meta-row">${sourceChips}${handoffNote}</div>`
      : "";

  // Insert parsed Markdown HTML directly for assistant messages
  row.innerHTML = `
    <div class="avatar">A&amp;R</div>
    <div class="bubble-col">
      <div class="bubble">${marked.parse(answer)}</div>
      ${metaRow}
    </div>
  `;
  chatEl.appendChild(row);
  scrollToBottom();
}

function appendSystemMessage(text) {
  const row = document.createElement("div");
  row.className = "row system";
  row.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(row);
  scrollToBottom();
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "row assistant typing";
  row.innerHTML = `
    <div class="avatar">A&amp;R</div>
    <div class="bubble-col">
      <div class="bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  `;
  chatEl.appendChild(row);
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  sendBtn.disabled = isSending || inputEl.value.trim() === "";
}

// ------------------------------------------------------------------
// Health check
// ------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("bad status");
    setStatus(true);
  } catch {
    setStatus(false);
  }
}

function setStatus(online) {
  statusEl.classList.toggle("online", online);
  statusEl.classList.toggle("offline", !online);
  statusLabel.textContent = online ? "Online" : "Offline";
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
function humanizeSource(filename) {
  return filename
    .replace(/\.md$/i, "")
    .replace(/^\d+-/, "")
    .split("-")
    .join(" ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function docIcon() {
  return `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" style="vertical-align:-1px"><path d="M6 2h9l5 5v15H6V2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M14 2v6h6" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
}

function flagIcon() {
  return `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" style="vertical-align:-1px"><path d="M5 3v18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M5 4h13l-3 4 3 4H5" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>`;
}