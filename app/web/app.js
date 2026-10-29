const api = "/api/v1";
const $ = (id) => document.getElementById(id);
const safe = (value) => String(value ?? "");

let currentMode = "direct"; // "direct" | "agent" | "plan"
window.currentDocCount = 0;

function escapeHtml(value) {
  return safe(value).replace(
    /[&<>'"]/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]
  );
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  button.dataset.label ??= button.textContent;
  button.textContent = busy ? label : button.dataset.label;
}

async function request(path, options = {}) {
  const response = await fetch(`${api}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

/* User Tracking & Email Session Management */
function getActiveEmail() {
  return (localStorage.getItem("research_user_email") || "").trim().toLowerCase();
}

function updateActiveUserDisplay(email) {
  const emailEl = $("user-display-email");
  if (emailEl) {
    emailEl.textContent = email || "Not Signed In";
    emailEl.title = email || "";
  }
}

function showLoginView() {
  const loginView = $("login-view");
  const appShell = $("app-shell");
  if (loginView) {
    loginView.style.display = "flex";
    loginView.classList.remove("hidden");
  }
  if (appShell) {
    appShell.style.display = "none";
    appShell.classList.add("hidden");
  }
  if ($("login-email")) $("login-email").value = "";
  if ($("login-error")) $("login-error").textContent = "";
  setTimeout(() => $("login-email")?.focus(), 80);
}

function hideLoginView() {
  const loginView = $("login-view");
  const appShell = $("app-shell");
  if (loginView) {
    loginView.style.display = "none";
    loginView.classList.add("hidden");
  }
  if (appShell) {
    appShell.style.display = "grid";
    appShell.classList.remove("hidden");
  }
}

function checkUserSession() {
  const email = getActiveEmail();
  if (!email) {
    showLoginView();
    renderDynamicQueries(0);
  } else {
    updateActiveUserDisplay(email);
    hideLoginView();
    loadLibrary();
    loadEvaluation();
  }
}

$("login-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const emailInput = $("login-email");
  const email = emailInput.value.trim().toLowerCase();
  const errorEl = $("login-error");
  const submitBtn = $("login-submit-btn");
  if (errorEl) errorEl.textContent = "";

  if (!email || !email.includes("@")) {
    if (errorEl) errorEl.textContent = "Please enter a valid email address.";
    return;
  }

  setBusy(submitBtn, true, "Entering workspace…");
  try {
    const data = await request("/auth/identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });

    localStorage.setItem("research_user_email", data.email);
    updateActiveUserDisplay(data.email);
    hideLoginView();
    await loadLibrary();
    loadEvaluation();
  } catch (err) {
    if (errorEl) errorEl.textContent = err.message || "Failed to identify user.";
  } finally {
    setBusy(submitBtn, false);
  }
});

$("btn-logout")?.addEventListener("click", () => {
  localStorage.removeItem("research_user_email");
  sessionStorage.clear();
  window.currentDocCount = 0;
  updateActiveUserDisplay("");
  if ($("library-documents-list")) $("library-documents-list").innerHTML = "<p class='empty'>No papers indexed yet.</p>";
  if ($("document-count")) $("document-count").textContent = "0";
  if ($("answer-section")) $("answer-section").classList.add("hidden");
  if ($("answer")) $("answer").innerHTML = "";
  if ($("question")) $("question").value = "";
  renderDynamicQueries(0);
  showLoginView();
});

/* Mode Switching & Reactive 3 Sub-Pages */
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  if ($("agent-mode")) $("agent-mode").checked = mode === "agent" || mode === "plan";

  // Toggle the 3 distinct sub-pages
  const subpages = {
    direct: $("subpage-direct"),
    agent: $("subpage-agent"),
    plan: $("subpage-plan"),
  };

  Object.entries(subpages).forEach(([m, el]) => {
    if (el) {
      if (m === mode) {
        el.classList.add("active");
        el.classList.remove("hidden");
      } else {
        el.classList.remove("active");
        el.classList.add("hidden");
      }
    }
  });

  // Update prompt placeholders, buttons, and quick suggestions
  const suggestionsContainer = $("mode-quick-suggestions");
  const questionInput = $("question");
  const askButton = $("ask-button");
  const hasDocs = window.currentDocCount > 0;

  if (mode === "direct") {
    if (questionInput) {
      questionInput.placeholder = hasDocs
        ? "Ask a direct factual question about your research collection… (e.g. What were the key conclusions of the study?)"
        : "Ask a factual research question or test academic web fallback…";
    }
    if (askButton) askButton.innerHTML = "Ask research desk <span>↵</span>";
    if (suggestionsContainer) {
      suggestionsContainer.innerHTML = hasDocs
        ? `
          <button class="prompt-chip" type="button">Summarize the key findings</button>
          <button class="prompt-chip" type="button">Compare the proposed methods</button>
          <button class="prompt-chip" type="button">What datasets were evaluated?</button>
        `
        : `
          <button class="prompt-chip" type="button">How hybrid search works</button>
          <button class="prompt-chip" type="button">2024 AHA/ASA stroke guidelines (Web Fallback)</button>
          <button class="prompt-chip" type="button">Upload &amp; indexing guide</button>
        `;
    }
  } else if (mode === "agent") {
    if (questionInput) {
      questionInput.placeholder = hasDocs
        ? "Ask the research agent to analyze, compare papers, or pick an Agent Tool below…"
        : "Ask how the agent coordinates analytical tools or test multi-tool reasoning…";
    }
    if (askButton) askButton.innerHTML = "Execute Agent Investigation <span>↵</span>";
    if (suggestionsContainer) {
      suggestionsContainer.innerHTML = hasDocs
        ? `
          <button class="prompt-chip" type="button">Extract metrics table across papers</button>
          <button class="prompt-chip" type="button">Compare methodologies and accuracy</button>
          <button class="prompt-chip" type="button">Structured summary of paper</button>
        `
        : `
          <button class="prompt-chip" type="button">How agent tools coordinate</button>
          <button class="prompt-chip" type="button">Tabular extraction explainer</button>
          <button class="prompt-chip" type="button">Cross-paper analysis overview</button>
        `;
    }
  } else if (mode === "plan") {
    if (questionInput) {
      questionInput.placeholder = hasDocs
        ? "Enter a complex, multi-faceted research inquiry to decompose and plan…"
        : "Enter a research hypothesis to test multi-step planning & decomposition…";
    }
    if (askButton) askButton.innerHTML = "Plan & Execute Analysis <span>↵</span>";
    if (suggestionsContainer) {
      suggestionsContainer.innerHTML = hasDocs
        ? `
          <button class="prompt-chip" type="button">Evaluate impact of explainable AI on accuracy</button>
          <button class="prompt-chip" type="button">Compare ML vs deep learning architectures</button>
          <button class="prompt-chip" type="button">Synthesize benchmark progression</button>
        `
        : `
          <button class="prompt-chip" type="button">Plan-and-execute paradigm</button>
          <button class="prompt-chip" type="button">Hypothesis decomposition workflow</button>
          <button class="prompt-chip" type="button">Evaluating clinical AI systems</button>
        `;
    }
  }

  // Attach click listener to prompt chips
  suggestionsContainer?.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (questionInput) {
        questionInput.value = chip.textContent;
        questionInput.focus();
      }
    });
  });

  const modeLabels = {
    direct: "DIRECT QA RESPONSE",
    agent: "AGENTIC MULTI-TOOL RESPONSE",
    plan: "PLAN-AND-EXECUTE SYNTHESIS",
  };
  const modeLabel = $("response-mode-label");
  if (modeLabel) modeLabel.textContent = modeLabels[mode] || "GROUNDED RESPONSE";
}

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

/* Dynamic Subpage Queries Rendering Based on User Documents */
function renderDynamicQueries(docCount) {
  window.currentDocCount = docCount;
  const directContainer = $("direct-queries-container");
  const agentContainer = $("agent-queries-container");
  const planContainer = $("plan-queries-container");

  if (docCount > 0) {
    if (directContainer) {
      directContainer.innerHTML = `
        <div class="subpage-section-title">Verified Inquiries on Your Indexed Papers (${docCount} documents in library):</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="What were the key diagnostic accuracy findings and conclusions across the papers?">
            <div class="query-card-header">
              <span class="query-card-icon">⚡</span>
              <span class="query-card-tag">Accuracy &amp; Diagnostics</span>
            </div>
            <p class="query-card-text">"What were the key diagnostic accuracy findings and conclusions across the papers?"</p>
          </div>
          <div class="query-card" data-prompt="Which imaging modalities (CT vs MRI) showed higher sensitivity in stroke lesion classification?">
            <div class="query-card-header">
              <span class="query-card-icon">🔍</span>
              <span class="query-card-tag">Modalities &amp; Imaging</span>
            </div>
            <p class="query-card-text">"Which imaging modalities (CT vs MRI) showed higher sensitivity in stroke lesion classification?"</p>
          </div>
          <div class="query-card" data-prompt="What machine learning architectures and feature extraction methods were evaluated?">
            <div class="query-card-header">
              <span class="query-card-icon">🤖</span>
              <span class="query-card-tag">Model Architectures</span>
            </div>
            <p class="query-card-text">"What machine learning architectures and feature extraction methods were evaluated?"</p>
          </div>
        </div>
      `;
    }

    if (agentContainer) {
      agentContainer.innerHTML = `
        <div class="subpage-section-title">Recommended Agent Investigations (${docCount} documents in library):</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="Extract a markdown comparison table of classification accuracies, models, and datasets across all indexed papers.">
            <div class="query-card-header">
              <span class="query-card-icon">📊</span>
              <span class="query-card-tag">Tabular Extraction</span>
            </div>
            <p class="query-card-text">"Extract a markdown comparison table of classification accuracies, models, and datasets across all indexed papers."</p>
          </div>
          <div class="query-card" data-prompt="Compare the methodological differences, datasets, and neural architectures between the stroke analysis studies.">
            <div class="query-card-header">
              <span class="query-card-icon">⚖️</span>
              <span class="query-card-tag">Comparative Contrast</span>
            </div>
            <p class="query-card-text">"Compare the methodological differences, datasets, and neural architectures between the stroke analysis studies."</p>
          </div>
          <div class="query-card" data-prompt="What exact patient cohort sizes, training splits, and validation metrics are reported across the papers?">
            <div class="query-card-header">
              <span class="query-card-icon">🔢</span>
              <span class="query-card-tag">Numeric Synthesis</span>
            </div>
            <p class="query-card-text">"What exact patient cohort sizes, training splits, and validation metrics are reported across the papers?"</p>
          </div>
        </div>
      `;
    }

    if (planContainer) {
      planContainer.innerHTML = `
        <div class="subpage-section-title">Recommended Plan-and-Execute Investigations (${docCount} documents in library):</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="Analyze the evolution from classical machine learning to deep CNNs in stroke imaging, comparing accuracy, limitations, and future clinical translation.">
            <div class="query-card-header">
              <span class="query-card-icon">🧠</span>
              <span class="query-card-tag">Architectural Evolution</span>
            </div>
            <p class="query-card-text">"Analyze the evolution from classical machine learning to deep CNNs in stroke imaging, comparing accuracy, limitations, and future clinical translation."</p>
          </div>
          <div class="query-card" data-prompt="Synthesize the benchmark progression for acute ischemic vs hemorrhagic stroke detection across the indexed studies.">
            <div class="query-card-header">
              <span class="query-card-icon">📈</span>
              <span class="query-card-tag">Benchmark Progression</span>
            </div>
            <p class="query-card-text">"Synthesize the benchmark progression for acute ischemic vs hemorrhagic stroke detection across the indexed studies."</p>
          </div>
          <div class="query-card" data-prompt="Decompose and evaluate how transfer learning on MRI/CT scans mitigates limited medical training sample size.">
            <div class="query-card-header">
              <span class="query-card-icon">🔬</span>
              <span class="query-card-tag">Sample Size &amp; Transfer</span>
            </div>
            <p class="query-card-text">"Decompose and evaluate how transfer learning on MRI/CT scans mitigates limited medical training sample size."</p>
          </div>
        </div>
      `;
    }
  } else {
    // 0 documents in library: onboarding & explanatory queries that teach each mode
    if (directContainer) {
      directContainer.innerHTML = `
        <div class="subpage-section-title">How Direct QA Works &amp; Exploratory Queries:</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="How does hybrid retrieval combine dense ChromaDB vectors and SQLite BM25 lexical keyword search?">
            <div class="query-card-header">
              <span class="query-card-icon">💡</span>
              <span class="query-card-tag">Hybrid Search Explainer</span>
            </div>
            <p class="query-card-text"><strong>Explore Concept:</strong> "How does hybrid retrieval combine dense ChromaDB vectors and SQLite BM25 lexical keyword search?"</p>
          </div>
          <div class="query-card" data-prompt="What are the 2024 AHA/ASA clinical guidelines for acute ischemic stroke management and IV thrombolysis?">
            <div class="query-card-header">
              <span class="query-card-icon">🌐</span>
              <span class="query-card-tag">Academic Web Search Test</span>
            </div>
            <p class="query-card-text"><strong>Try Web Fallback:</strong> "What are the 2024 AHA/ASA clinical guidelines for acute ischemic stroke management and IV thrombolysis?"</p>
          </div>
          <div class="query-card" data-prompt="What file formats (PDF, TXT, MD) can be indexed and how does chunking preserve page citations?">
            <div class="query-card-header">
              <span class="query-card-icon">📄</span>
              <span class="query-card-tag">Upload &amp; Indexing Guide</span>
            </div>
            <p class="query-card-text"><strong>Onboarding Guide:</strong> "What file formats (PDF, TXT, MD) can be indexed and how does chunking preserve page citations?"</p>
          </div>
        </div>
        <p class="card-copy" style="margin-top: 10px; font-size: 0.85rem; color: var(--color-muted);">
          ℹ️ <em>Tip: You currently have 0 documents in your private library. Upload research papers below to unlock grounded citations over your personal corpus!</em>
        </p>
      `;
    }

    if (agentContainer) {
      agentContainer.innerHTML = `
        <div class="subpage-section-title">How Autonomous Agent Mode Works &amp; Tool Capabilities:</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="Explain how the research agent selects and coordinates analytical tools like metric tables and paper comparisons.">
            <div class="query-card-header">
              <span class="query-card-icon">🤖</span>
              <span class="query-card-tag">Agent Architecture</span>
            </div>
            <p class="query-card-text"><strong>Explore Tool Orchestration:</strong> "Explain how the research agent selects and coordinates analytical tools like metric tables and paper comparisons."</p>
          </div>
          <div class="query-card" data-prompt="How does the agent cross-examine contradictory findings and extract quantitative metrics across multiple studies?">
            <div class="query-card-header">
              <span class="query-card-icon">⚖️</span>
              <span class="query-card-tag">Cross-Paper Reasoning</span>
            </div>
            <p class="query-card-text"><strong>Explore Synthesis:</strong> "How does the agent cross-examine contradictory findings and extract quantitative metrics across multiple studies?"</p>
          </div>
          <div class="query-card" data-prompt="How can I use Agent Mode once I upload multiple comparative clinical or technical papers?">
            <div class="query-card-header">
              <span class="query-card-icon">📑</span>
              <span class="query-card-tag">Multi-Paper Workflow</span>
            </div>
            <p class="query-card-text"><strong>Agent Quickstart:</strong> "How can I use Agent Mode once I upload multiple comparative clinical or technical papers?"</p>
          </div>
        </div>
        <p class="card-copy" style="margin-top: 10px; font-size: 0.85rem; color: var(--color-muted);">
          ℹ️ <em>Tip: In Agent Mode, you can click any of the 4 Agent Tool cards above or ask complex comparative questions once you ingest 1 or more papers.</em>
        </p>
      `;
    }

    if (planContainer) {
      planContainer.innerHTML = `
        <div class="subpage-section-title">How Plan &amp; Execute Works &amp; Multi-Step Decomposition:</div>
        <div class="example-queries-grid">
          <div class="query-card" data-prompt="What is the plan-and-execute paradigm and how does it decompose broad scientific inquiries into verifiable sub-questions?">
            <div class="query-card-header">
              <span class="query-card-icon">🧠</span>
              <span class="query-card-tag">Decomposition Paradigm</span>
            </div>
            <p class="query-card-text"><strong>Explore Planning:</strong> "What is the plan-and-execute paradigm and how does it decompose broad scientific inquiries into verifiable sub-questions?"</p>
          </div>
          <div class="query-card" data-prompt="How does the meta-synthesizer review evidence from each sub-hypothesis to generate an executive report?">
            <div class="query-card-header">
              <span class="query-card-icon">🔬</span>
              <span class="query-card-tag">Meta-Analysis Workflow</span>
            </div>
            <p class="query-card-text"><strong>Explore Synthesis:</strong> "How does the meta-synthesizer review evidence from each sub-hypothesis to generate an executive report?"</p>
          </div>
          <div class="query-card" data-prompt="Show an example of a multi-stage research plan for evaluating clinical AI diagnostic systems.">
            <div class="query-card-header">
              <span class="query-card-icon">📋</span>
              <span class="query-card-tag">Demonstration Plan</span>
            </div>
            <p class="query-card-text"><strong>Example Multi-Step Plan:</strong> "Show an example of a multi-stage research plan for evaluating clinical AI diagnostic systems."</p>
          </div>
        </div>
        <p class="card-copy" style="margin-top: 10px; font-size: 0.85rem; color: var(--color-muted);">
          ℹ️ <em>Tip: Plan &amp; Execute performs best for complex thesis, systematic review, and multi-faceted questions. Upload your papers to start decomposing research queries!</em>
        </p>
      `;
    }
  }

  // Re-wire card click listeners and refresh prompt chips
  wireSubpageCards();
  setMode(currentMode);
}

// Wire up all example queries and agent tool cards across all 3 sub-pages
function wireSubpageCards() {
  const questionInput = $("question");
  document.querySelectorAll(".query-card, .agent-tool-box").forEach((card) => {
    // Avoid double attaching
    if (card.dataset.wired === "true") return;
    card.dataset.wired = "true";
    card.addEventListener("click", () => {
      const prompt = card.dataset.prompt;
      if (prompt && questionInput) {
        questionInput.value = prompt;
        questionInput.focus();
        questionInput.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });
}

/* Markdown Parser */
function parseMarkdown(text) {
  if (!text) return "";
  const lines = text.split("\n");
  const htmlLines = [];
  let inTable = false;
  let tableHeaderParsed = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.startsWith("|") && line.endsWith("|")) {
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) {
        tableHeaderParsed = true;
        continue;
      }
      if (!inTable) {
        inTable = true;
        tableHeaderParsed = false;
        htmlLines.push('<div class="table-container"><table class="markdown-table"><thead><tr>');
        cells.forEach((cell) => htmlLines.push(`<th>${formatInline(cell)}</th>`));
        htmlLines.push("</tr></thead><tbody>");
      } else if (!tableHeaderParsed) {
        htmlLines.push("<tr>");
        cells.forEach((cell) => htmlLines.push(`<th>${formatInline(cell)}</th>`));
        htmlLines.push("</tr>");
      } else {
        htmlLines.push("<tr>");
        cells.forEach((cell) => htmlLines.push(`<td>${formatInline(cell)}</td>`));
        htmlLines.push("</tr>");
      }
      continue;
    } else if (inTable) {
      inTable = false;
      tableHeaderParsed = false;
      htmlLines.push("</tbody></table></div>");
    }

    if (!line) {
      htmlLines.push("<br>");
      continue;
    }

    if (line.startsWith("### ")) htmlLines.push(`<h4>${formatInline(line.slice(4))}</h4>`);
    else if (line.startsWith("## ")) htmlLines.push(`<h3>${formatInline(line.slice(3))}</h3>`);
    else if (line.startsWith("# ")) htmlLines.push(`<h2>${formatInline(line.slice(2))}</h2>`);
    else if (line.startsWith("- ") || line.startsWith("* ")) htmlLines.push(`<li>${formatInline(line.slice(2))}</li>`);
    else htmlLines.push(`<p>${formatInline(line)}</p>`);
  }

  if (inTable) {
    htmlLines.push("</tbody></table></div>");
  }

  return htmlLines.join("");
}

function formatInline(str) {
  let esc = escapeHtml(str);
  esc = esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  esc = esc.replace(/\*([^\*]+?)\*/g, "<em>$1</em>");
  esc = esc.replace(
    /\[S(\d+)\]/g,
    '<span class="citation-badge" title="Source S$1" onclick="highlightCitation(\'S$1\')">[S$1]</span>'
  );
  return esc;
}

window.highlightCitation = function (citationId) {
  const el = document.getElementById(`citation-${citationId}`);
  if (el) {
    el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("highlight");
    setTimeout(() => el.classList.remove("highlight"), 2500);
  }
};

function renderCitations(citations) {
  $("citations").innerHTML = citations.length
    ? citations
        .map(
          (source, index) =>
            `<details class="citation" id="citation-${escapeHtml(source.citation_id || 'S' + (index + 1))}">
              <summary>
                <span class="source-number">${index + 1}</span>
                <span>
                  <strong>${escapeHtml(source.filename)}</strong>
                  <small>Page ${escapeHtml(source.page_number)} · [${escapeHtml(source.citation_id || 'S' + (index + 1))}]</small>
                </span>
                <span class="expand">+</span>
              </summary>
              <p>${escapeHtml(source.excerpt)}</p>
            </details>`
        )
        .join("")
    : "<p class='empty'>No sources were returned.</p>";
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("ask-button");
  const error = $("ask-error");
  error.textContent = "";

  const userEmail = getActiveEmail();
  if (!userEmail) {
    showLoginView();
    return;
  }

  const busyLabels = {
    direct: "Searching evidence…",
    agent: "Agent coordinating tools…",
    plan: "Decomposing & synthesizing…",
  };
  setBusy(button, true, busyLabels[currentMode] || "Researching…");

  try {
    let endpoint = "/ask";
    if (currentMode === "agent") endpoint = "/agent/ask";
    else if (currentMode === "plan") endpoint = "/agent/plan";

    const result = await request(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("question").value.trim(),
        user_email: userEmail,
      }),
    });

    const planContainer = $("plan-steps-container");
    if (currentMode === "plan" && result.sub_questions && result.sub_questions.length > 0) {
      $("plan-step-count").textContent = `${result.sub_questions.length} research steps`;
      $("plan-steps-list").innerHTML = result.sub_questions
        .map((subQ, idx) => {
          const subAns = result.sub_answers && result.sub_answers[idx] ? result.sub_answers[idx] : "";
          return `<div class="plan-step-card">
            <div class="plan-step-header">
              <span class="step-num">${idx + 1}</span>
              <strong>${escapeHtml(subQ)}</strong>
            </div>
            <div class="plan-step-answer">${parseMarkdown(subAns)}</div>
          </div>`;
        })
        .join("");
      planContainer.classList.remove("hidden");
    } else {
      planContainer.classList.add("hidden");
    }

    const modeLabel = $("response-mode-label");
    if (modeLabel) {
      if (result.source_mode === "google_search_rag") {
        modeLabel.textContent = "ACADEMIC & WEB SEARCH FALLBACK";
      } else {
        const modeLabels = {
          direct: "DIRECT QA RESPONSE",
          agent: "AGENTIC MULTI-TOOL RESPONSE",
          plan: "PLAN-AND-EXECUTE SYNTHESIS",
        };
        modeLabel.textContent = modeLabels[currentMode] || "GROUNDED RESPONSE";
      }
    }

    $("answer").innerHTML = parseMarkdown(result.answer);
    $("candidate-count").textContent = `${result.retrieval_candidate_count} passages considered`;
    renderCitations(result.citations || []);
    $("answer-section").classList.remove("hidden");
    $("answer-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (problem) {
    error.textContent = problem.message;
  } finally {
    setBusy(button, false);
  }
});

$("question").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") $("ask-form").requestSubmit();
});

// Library management: Upload, List, and Delete
async function loadLibrary() {
  const target = $("library-documents-list");
  const countEl = $("document-count");
  if (!target) return;
  const userEmail = getActiveEmail();
  try {
    const url = userEmail ? `/documents?user_email=${encodeURIComponent(userEmail)}` : "/documents";
    const docs = await request(url);
    if (countEl) countEl.textContent = docs.length;
    renderDynamicQueries(docs.length);
    if (!docs.length) {
      target.innerHTML = "<p class='empty'>No papers indexed yet in your library. Upload a research PDF or text below to start querying.</p>";
      return;
    }
    target.innerHTML = docs
      .map(
        (doc) => `
        <div class="doc-item">
          <div class="doc-info">
            <span class="doc-icon">📄</span>
            <div>
              <strong>${escapeHtml(doc.filename)}</strong>
              <small>${doc.page_count} pages · ${doc.chunk_count} chunks</small>
            </div>
          </div>
          <button type="button" class="btn-delete" title="Delete from your library" onclick="deleteDocument('${escapeHtml(doc.filename)}')">✕</button>
        </div>`
      )
      .join("");
  } catch (problem) {
    target.innerHTML = `<p class="empty error">${escapeHtml(problem.message)}</p>`;
  }
}

window.deleteDocument = async function (filename) {
  if (!confirm(`Are you sure you want to remove "${filename}" from your library?`)) return;
  try {
    const userEmail = getActiveEmail();
    const url = `/documents/${encodeURIComponent(filename)}` + (userEmail ? `?user_email=${encodeURIComponent(userEmail)}` : "");
    await request(url, { method: "DELETE" });
    await loadLibrary();
  } catch (problem) {
    alert(`Could not delete: ${problem.message}`);
  }
};

$("refresh-library")?.addEventListener("click", loadLibrary);

$("upload-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("upload-button");
  const result = $("upload-result");
  const file = $("document").files[0];
  if (!file) return;
  const userEmail = getActiveEmail();
  setBusy(button, true, "Indexing source…");
  try {
    const form = new FormData();
    form.append("file", file);
    const uploadUrl = "/documents/ingest" + (userEmail ? `?user_email=${encodeURIComponent(userEmail)}` : "");
    const data = await request(uploadUrl, { method: "POST", body: form });
    result.className = "message success";
    result.textContent = `${data.filename} ready — ${data.chunk_count} passages indexed.`;
    $("document").value = "";
    await loadLibrary();
  } catch (problem) {
    result.className = "message error";
    result.textContent = problem.message;
  } finally {
    setBusy(button, false);
  }
});

$("search-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const target = $("search-results");
  target.textContent = "Searching…";
  target.className = "search-results";
  try {
    const userEmail = getActiveEmail();
    const searchUrl = `/search?query=${encodeURIComponent($("search-query").value.trim())}` + (userEmail ? `&user_email=${encodeURIComponent(userEmail)}` : "");
    const data = await request(searchUrl);
    target.innerHTML = data.results.length
      ? data.results
          .map(
            (hit) => `
          <article class="search-hit">
            <div>
              <strong>${escapeHtml(hit.filename)}</strong>
              <span>p. ${escapeHtml(hit.page_number)}</span>
            </div>
            <p>${escapeHtml(hit.text)}</p>
            <small>${escapeHtml(hit.retrieval_sources.join(" · "))}</small>
          </article>`
          )
          .join("")
      : "No matching passages found.";
  } catch (problem) {
    target.className = "search-results error";
    target.textContent = problem.message;
  }
});

async function loadEvaluation() {
  const target = $("evaluation-results");
  if (!target) return;
  try {
    const report = await request("/evaluations/latest");
    const metrics = Object.entries(report.averages)
      .map(
        ([name, value]) => `
      <div class="metric">
        <span>${escapeHtml(name.replaceAll("_", " "))}</span>
        <strong>${(Number(value) * 100).toFixed(0)}%</strong>
        <small>target ${(Number(report.targets[name]) * 100).toFixed(0)}%</small>
      </div>`
      )
      .join("");
    target.className = "evaluation-results";
    target.innerHTML = `
      <p class="evaluation-verdict ${report.passed ? "pass" : "fail"}">
        ${report.passed ? "Quality targets met" : "Review quality targets"}
      </p>
      <div class="metrics">${metrics}</div>
      <p class="report-date">${report.dataset_size} questions · ${new Date(report.created_at).toLocaleDateString()}</p>`;
  } catch (problem) {
    target.className = "evaluation-results empty";
    target.textContent = problem.message;
  }
}

$("refresh-evaluation")?.addEventListener("click", loadEvaluation);
$("theme-toggle")?.addEventListener("click", () => document.body.classList.toggle("dark"));

// Initialize subpage cards and mode
wireSubpageCards();
setMode("direct");

// Check user session
checkUserSession();
