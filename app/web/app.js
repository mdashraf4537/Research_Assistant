const api = "/api/v1";
const $ = (id) => document.getElementById(id);
const safe = (value) => String(value ?? "");

function escapeHtml(value) {
  return safe(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}

function setBusy(button, busy, label) {
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

function renderCitations(citations) {
  $("citations").innerHTML = citations.length ? citations.map((source, index) => `<details class="citation"><summary><span class="source-number">${index + 1}</span><span><strong>${escapeHtml(source.filename)}</strong><small>Page ${escapeHtml(source.page_number)}</small></span><span class="expand">+</span></summary><p>${escapeHtml(source.excerpt)}</p></details>`).join("") : "<p class='empty'>No sources were returned.</p>";
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("ask-button");
  const error = $("ask-error");
  error.textContent = "";
  setBusy(button, true, "Searching evidence…");
  try {
    const endpoint = $("agent-mode").checked ? "/agent/ask" : "/ask";
    const result = await request(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: $("question").value.trim() }) });
    $("answer").innerHTML = escapeHtml(result.answer).replace(/\n/g, "<br>");
    $("candidate-count").textContent = `${result.retrieval_candidate_count} passages considered`;
    renderCitations(result.citations);
    $("answer-section").classList.remove("hidden");
    $("answer-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (problem) { error.textContent = problem.message; } finally { setBusy(button, false); }
});

document.querySelectorAll(".prompt-chip").forEach((chip) => chip.addEventListener("click", () => { $("question").value = chip.textContent; $("question").focus(); }));
$("question").addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") $("ask-form").requestSubmit(); });

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("upload-button"); const result = $("upload-result"); const file = $("document").files[0];
  if (!file) return;
  setBusy(button, true, "Indexing source…");
  try { const form = new FormData(); form.append("file", file); const data = await request("/documents/ingest", { method: "POST", body: form }); result.className = "message success"; result.textContent = `${data.filename} is ready — ${data.chunk_count} citeable passages indexed.`; }
  catch (problem) { result.className = "message error"; result.textContent = problem.message; }
  finally { setBusy(button, false); }
});

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const target = $("search-results"); target.textContent = "Searching…"; target.className = "search-results";
  try { const data = await request(`/search?query=${encodeURIComponent($("search-query").value.trim())}`); target.innerHTML = data.results.length ? data.results.map((hit) => `<article class="search-hit"><div><strong>${escapeHtml(hit.filename)}</strong><span>p. ${escapeHtml(hit.page_number)}</span></div><p>${escapeHtml(hit.text)}</p><small>${escapeHtml(hit.retrieval_sources.join(" · "))}</small></article>`).join("") : "No matching passages found."; }
  catch (problem) { target.className = "search-results error"; target.textContent = problem.message; }
});

async function loadEvaluation() {
  const target = $("evaluation-results");
  try { const report = await request("/evaluations/latest"); const metrics = Object.entries(report.averages).map(([name, value]) => `<div class="metric"><span>${escapeHtml(name.replaceAll("_", " "))}</span><strong>${(Number(value) * 100).toFixed(0)}%</strong><small>target ${(Number(report.targets[name]) * 100).toFixed(0)}%</small></div>`).join(""); target.className = "evaluation-results"; target.innerHTML = `<p class="evaluation-verdict ${report.passed ? "pass" : "fail"}">${report.passed ? "Quality targets met" : "Review quality targets"}</p><div class="metrics">${metrics}</div><p class="report-date">${report.dataset_size} questions · ${new Date(report.created_at).toLocaleDateString()}</p>`; }
  catch (problem) { target.className = "evaluation-results empty"; target.textContent = problem.message; }
}

$("refresh-evaluation").addEventListener("click", loadEvaluation);
$("theme-toggle").addEventListener("click", () => document.body.classList.toggle("dark"));

async function checkHealth() {
  const status = $("service-status");
  try { const health = await request("/health/ready"); status.innerHTML = "<i></i> Online"; status.title = `${health.service} ${health.version}`; }
  catch { status.classList.add("offline"); status.innerHTML = "<i></i> Offline"; }
}
checkHealth(); loadEvaluation();
