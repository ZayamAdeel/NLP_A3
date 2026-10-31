const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("msg-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const stageBadge = document.getElementById("stage-badge");
const connStatus = document.getElementById("conn-status");

let sessionId = null;
let socket = null;
let currentAssistantBubble = null;

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function setSending(isSending) {
  sendBtn.disabled = isSending;
  input.disabled = isSending;
}

// A random id kept in this browser lets the assistant recognise a returning customer (CRM).
function userId() {
  let id = null;
  try { id = localStorage.getItem("bc_user_id"); } catch (e) { /* storage blocked */ }
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()) + Math.random().toString(16).slice(2);
    try { localStorage.setItem("bc_user_id", id); } catch (e) { /* ignore */ }
  }
  return id;
}

async function bootstrapSession() {
  chatWindow.innerHTML = "";
  const res = await fetch(`/api/session?user_id=${encodeURIComponent(userId())}`, { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  addBubble("assistant", data.greeting);
  connectSocket();
}

function connectSocket() {
  if (socket) {
    socket.close();
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat`);

  socket.onopen = () => {
    connStatus.textContent = "connected";
  };

  socket.onclose = () => {
    connStatus.textContent = "disconnected — retrying in 2s";
    setTimeout(connectSocket, 2000);
  };

  socket.onerror = () => {
    connStatus.textContent = "connection error";
  };

  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleServerEvent(msg);
  };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // textContent only: never inject HTML
  return node;
}

// Citations: one chip per source the answer was grounded in; click to read the excerpt.
function buildSources(sources) {
  const wrap = el("div", "sources");
  wrap.appendChild(el("span", "sources-label", "Sources"));
  const row = el("div", "src-row");
  wrap.appendChild(row);
  sources.forEach((s) => {
    const panel = el("div", "src-detail");
    panel.hidden = true;
    panel.appendChild(el("div", "src-path", `${s.doc_id} (relevance ${Number(s.score).toFixed(2)})`));
    panel.appendChild(el("div", "src-snippet", s.snippet));

    const chip = el("button", "src-chip", `[${s.n}] ${s.title}`);
    chip.type = "button";
    chip.setAttribute("aria-expanded", "false");
    chip.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      chip.setAttribute("aria-expanded", String(!panel.hidden));
    });
    row.appendChild(chip);
    wrap.appendChild(panel);
  });
  return wrap;
}

function buildTool(t) {
  const card = el("div", "tool-card");
  card.appendChild(el("div", "tool-head", `Tool: ${t.name} (${t.status}, ${t.ms} ms)`));
  card.appendChild(el("div", "tool-body", t.summary));
  return card;
}

function renderExtras(msg) {
  const r = msg.retrieval;
  const extra = el("div", "msg-extra");
  (msg.tools || []).forEach((t) => extra.appendChild(buildTool(t)));

  if (msg.sources && msg.sources.length) {
    extra.appendChild(buildSources(msg.sources));
  } else if (r && r.status === "no_match") {
    extra.appendChild(el("div", "src-note", "No matching documents found, so this answer is not based on the knowledge base."));
  } else if (r && r.status === "unavailable") {
    extra.appendChild(el("div", "src-note", "The knowledge base could not be reached, so this answer is not based on documents."));
  }

  const parts = [];
  if (r && r.status !== "disabled" && r.status !== "skipped") {
    parts.push(`retrieval ${r.ms} ms${r.cached ? " (cached)" : ""}`);
  }
  if (msg.timing && msg.timing.tools_ms > 0) parts.push(`tools ${msg.timing.tools_ms} ms`);
  if (msg.time_to_first_token != null) parts.push(`first token ${msg.time_to_first_token} s`);
  if (msg.tokens_per_second != null) parts.push(`${msg.tokens_per_second} tokens/s`);
  if (msg.total_time != null) parts.push(`generation ${msg.total_time} s`);
  if (parts.length) extra.appendChild(el("div", "meta", parts.join(", ")));

  chatWindow.appendChild(extra);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function handleServerEvent(msg) {
  switch (msg.type) {
    case "stage":
      stageBadge.textContent = `stage: ${msg.stage}`;
      break;
    case "token":
      if (!currentAssistantBubble) {
        currentAssistantBubble = addBubble("assistant", "");
      }
      currentAssistantBubble.textContent += msg.text;
      chatWindow.scrollTop = chatWindow.scrollHeight;
      break;
    case "blocked":
      addBubble("blocked", msg.text);
      setSending(false);
      currentAssistantBubble = null;
      break;
    case "done": {
      renderExtras(msg);
      currentAssistantBubble = null;
      setSending(false);
      break;
    }
    case "error":
      addBubble("error", `⚠ ${msg.message}`);
      currentAssistantBubble = null;
      setSending(false);
      break;
    default:
      console.warn("Unknown event", msg);
  }
}

function sendMessage() {
  const text = input.value.trim();
  if (!text || !sessionId || socket.readyState !== WebSocket.OPEN) return;

  addBubble("user", text);
  input.value = "";
  setSending(true);
  currentAssistantBubble = null;

  socket.send(JSON.stringify({ type: "message", session_id: sessionId, text }));
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

newChatBtn.addEventListener("click", async () => {
  if (sessionId) {
    await fetch(`/api/session/${sessionId}`, { method: "DELETE" });
  }
  stageBadge.textContent = "stage: greeting";
  await bootstrapSession();
});

bootstrapSession();
