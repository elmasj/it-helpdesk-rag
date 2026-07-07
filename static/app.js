// Front-end for the IT Helpdesk Knowledge Assistant.
//
// Deliberately plain JavaScript, no build step and no external libraries --
// this whole app runs from two static files (this one + style.css) served
// by Flask, with Claude's Markdown answer already rendered to safe HTML by
// the backend (see webapp.py render_answer_html()).

const chat = document.getElementById("chat");
const emptyState = document.getElementById("empty-state");
const composer = document.getElementById("composer");
const input = document.getElementById("question-input");
const sendButton = document.getElementById("send-button");
const spendValue = document.getElementById("spend-value");
const budgetFill = document.getElementById("budget-fill");
const themeToggle = document.getElementById("theme-toggle");

// Theme is applied by an inline script in <head> before first paint (to
// avoid a flash of the wrong theme); this just handles the toggle click
// and remembers the choice for next time.
themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

// ---------- Sidebar (knowledge index) ----------
const layout = document.querySelector(".layout");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarClose = document.getElementById("sidebar-close");
const sidebarOverlay = document.getElementById("sidebar-overlay");

function openSidebar() { layout.classList.add("sidebar-open"); }
function closeSidebar() { layout.classList.remove("sidebar-open"); }

sidebarToggle.addEventListener("click", openSidebar);
sidebarClose.addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);

// Clicking a topic asks a reasonable starter question about it rather than
// just being a static label -- keeps the index useful, not just decorative.
document.querySelectorAll(".sidebar-topic").forEach((button) => {
  button.addEventListener("click", () => {
    const topic = button.dataset.topic;
    const product = button.dataset.product;
    closeSidebar();
    sendQuestion(`What are common tasks or issues related to ${topic} in ${product}?`);
  });
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autoGrow);

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addUserMessage(question) {
  emptyState.style.display = "none";
  const msg = document.createElement("div");
  msg.className = "msg user";
  msg.innerHTML = `<div class="bubble">${escapeHtml(question)}</div>`;
  chat.appendChild(msg);
  scrollToBottom();
}

function addTypingIndicator() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  msg.id = "typing-msg";
  msg.innerHTML = `<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(msg);
  scrollToBottom();
  return msg;
}

function citationsHtml(citations) {
  if (!citations || citations.length === 0) return "";
  const items = citations.map((c) => `
    <div class="citation">
      <div class="citation-index">[${c.index}]</div>
      <div class="citation-body">
        <div class="citation-title"><a href="${c.source_url}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a></div>
        <div class="citation-meta">
          <span class="product-badge">${escapeHtml(c.product)}</span>
          <span>${escapeHtml(c.section || "top of doc")}</span>
        </div>
      </div>
    </div>
  `).join("");

  return `
    <details class="citations">
      <summary>Sources (${citations.length} passage${citations.length === 1 ? "" : "s"} retrieved)</summary>
      <div class="citation-list">${items}</div>
    </details>
  `;
}

function replaceTypingWithAnswer(typingMsg, data) {
  const usage = data.usage;
  typingMsg.innerHTML = `
    <div class="bubble">
      <div class="answer-body">${data.answer_html}</div>
      ${citationsHtml(data.citations)}
      <div class="msg-footer">
        ${usage.input_tokens} in / ${usage.output_tokens} out tokens
        &nbsp;&middot;&nbsp; this query: $${data.cost_usd.toFixed(4)}
        &nbsp;&middot;&nbsp; total spend: $${data.total_spend_usd.toFixed(4)}
      </div>
    </div>
  `;
  spendValue.textContent = `$${data.total_spend_usd.toFixed(4)}`;
  if (typeof data.budget_pct_used === "number") {
    budgetFill.style.width = data.budget_pct_used + "%";
  }
  scrollToBottom();
}

function replaceTypingWithError(typingMsg, message) {
  typingMsg.innerHTML = `<div class="bubble error-bubble">${escapeHtml(message)}</div>`;
  scrollToBottom();
}

async function sendQuestion(question) {
  addUserMessage(question);
  input.value = "";
  autoGrow();
  sendButton.disabled = true;

  const typingMsg = addTypingIndicator();

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      replaceTypingWithError(typingMsg, data.error || "Something went wrong.");
    } else {
      replaceTypingWithAnswer(typingMsg, data);
    }
  } catch (err) {
    replaceTypingWithError(typingMsg, "Network error -- is the Flask server still running?");
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  sendQuestion(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    sendQuestion(chip.dataset.question);
  });
});
