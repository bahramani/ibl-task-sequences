const apiBase = "/api";
let currentPid = null;
let sessionData = null;

const qs = (id) => document.getElementById(id);

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return await res.json();
}

function renderPlot(divId, fig) {
  if (!fig || !fig.data) return;
  Plotly.react(divId, fig.data, fig.layout || {}, { responsive: true });
}

function fillTable(tableId, rows, columns) {
  const table = qs(tableId);
  table.innerHTML = "";
  if (!rows || rows.length === 0) return;
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = row[col];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function updateNeuronInfo() {
  const info = qs("neuronInfo");
  info.innerHTML = "";
  if (!sessionData) return;
  const idx = qs("neuronSelect").selectedIndex;
  if (idx < 0) return;
  const clusterId = sessionData.cluster_ids[idx];
  const region = sessionData.cluster_acronyms[idx];
  const label = sessionData.labels ? (sessionData.labels[idx] === 1 ? "Good" : "Not good") : "NA";
  ["Cluster: " + clusterId, "Region: " + region, "Quality: " + label].forEach((text) => {
    const span = document.createElement("span");
    span.textContent = text;
    info.appendChild(span);
  });
}

function getPlotOnlyGood() {
  return qs("plotOnlyGood").checked ? "true" : "false";
}

function getVariability() {
  return qs("variabilitySelect").value;
}

async function loadSession(pid) {
  currentPid = pid;
  sessionData = await fetchJSON(`${apiBase}/session?pid=${pid}`);

  fillTable(
    "sessionInfo",
    Object.entries(sessionData.info).map(([k, v]) => ({ key: k, value: v })),
    ["key", "value"]
  );
  fillTable("regionTable", sessionData.region_table, ["region", "all", "good"]);

  qs("generalStart").value = sessionData.min_time.toFixed(2);
  qs("generalEnd").value = Math.min(sessionData.min_time + 10, sessionData.max_time).toFixed(2);

  const trialSelect = qs("trialSelect");
  trialSelect.innerHTML = "";
  sessionData.trial_idx.forEach((val) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = val;
    trialSelect.appendChild(opt);
  });

  const neuronSelect = qs("neuronSelect");
  neuronSelect.innerHTML = "";
  sessionData.cluster_ids.forEach((cid, idx) => {
    const opt = document.createElement("option");
    const region = sessionData.cluster_acronyms[idx];
    const label = sessionData.labels ? (sessionData.labels[idx] === 1 ? "Good" : "Not good") : "NA";
    opt.value = cid;
    opt.textContent = `${cid} | ${region} | ${label}`;
    neuronSelect.appendChild(opt);
  });

  updateNeuronInfo();
  await updateAllPlots();
}

async function updateGeneralRaster() {
  const tStart = qs("generalStart").value;
  const tEnd = qs("generalEnd").value;
  const sort = qs("generalSort").value;
  const variability = getVariability();
  const plotOnlyGood = getPlotOnlyGood();
  const fig = await fetchJSON(
    `${apiBase}/fig/general_raster?pid=${currentPid}&t_start=${tStart}&t_end=${tEnd}&sort=${sort}&plot_only_good=${plotOnlyGood}&variability=${variability}`
  );
  renderPlot("generalRaster", fig);
}

async function updateTrialRaster() {
  const trialIdx = qs("trialSelect").value;
  const sort = qs("trialSort").value;
  const variability = getVariability();
  const plotOnlyGood = getPlotOnlyGood();
  const fig = await fetchJSON(
    `${apiBase}/fig/trial_raster?pid=${currentPid}&trial_idx=${trialIdx}&sort=${sort}&plot_only_good=${plotOnlyGood}&variability=${variability}`
  );
  renderPlot("trialRaster", fig);
}

async function updatePopulation() {
  const sort = qs("populationSort").value;
  const plotOnlyGood = getPlotOnlyGood();
  const resp = await fetchJSON(
    `${apiBase}/fig/population?pid=${currentPid}&sort=${sort}&plot_only_good=${plotOnlyGood}`
  );
  renderPlot("popPlot1", resp.figs[0]);
  renderPlot("popPlot2", resp.figs[1]);
  renderPlot("popPlot3", resp.figs[2]);
}

async function updateCoupling() {
  const plotOnlyGood = getPlotOnlyGood();
  const resp = await fetchJSON(
    `${apiBase}/fig/coupling?pid=${currentPid}&plot_only_good=${plotOnlyGood}`
  );
  renderPlot("couplingSpont", resp.spont);
  renderPlot("couplingTask", resp.task);
}

async function updateStprComparison() {
  const clusterId = qs("neuronSelect").value;
  const plotOnlyGood = getPlotOnlyGood();
  const strength = await fetchJSON(
    `${apiBase}/fig/stpr_strength?pid=${currentPid}&cluster_id=${clusterId}&plot_only_good=${plotOnlyGood}`
  );
  const delay = await fetchJSON(
    `${apiBase}/fig/stpr_delay?pid=${currentPid}&cluster_id=${clusterId}&plot_only_good=${plotOnlyGood}`
  );
  renderPlot("stprStrength", strength);
  renderPlot("stprDelay", delay);
}

async function updateSingleNeuron() {
  const clusterId = qs("neuronSelect").value;
  const stim = await fetchJSON(
    `${apiBase}/fig/single_stim?pid=${currentPid}&cluster_id=${clusterId}`
  );
  const move = await fetchJSON(
    `${apiBase}/fig/single_move?pid=${currentPid}&cluster_id=${clusterId}`
  );
  const feedback = await fetchJSON(
    `${apiBase}/fig/single_feedback?pid=${currentPid}&cluster_id=${clusterId}`
  );
  const stprTask = await fetchJSON(
    `${apiBase}/fig/stpr_curve?pid=${currentPid}&cluster_id=${clusterId}&mode=task`
  );
  const stprSpont = await fetchJSON(
    `${apiBase}/fig/stpr_curve?pid=${currentPid}&cluster_id=${clusterId}&mode=spont`
  );
  renderPlot("singleStim", stim);
  renderPlot("singleMove", move);
  renderPlot("singleFeedback", feedback);
  renderPlot("stprTaskCurve", stprTask);
  renderPlot("stprSpontCurve", stprSpont);
}

async function updateAllPlots() {
  await updateGeneralRaster();
  await updateTrialRaster();
  await updatePopulation();
  await updateCoupling();
  await updateStprComparison();
  await updateSingleNeuron();
}

function bindEvents() {
  qs("pidSelect").addEventListener("change", (e) => loadSession(e.target.value));
  qs("plotOnlyGood").addEventListener("change", () => updateAllPlots());
  qs("variabilitySelect").addEventListener("change", () => {
    updateGeneralRaster();
    updateTrialRaster();
  });
  qs("generalUpdate").addEventListener("click", updateGeneralRaster);
  qs("trialUpdate").addEventListener("click", updateTrialRaster);
  qs("populationUpdate").addEventListener("click", updatePopulation);
  qs("generalShiftBtn").addEventListener("click", () => {
    const shiftVal = parseFloat(qs("generalShift").value || "0");
    const start = parseFloat(qs("generalStart").value);
    const end = parseFloat(qs("generalEnd").value);
    qs("generalStart").value = (start + shiftVal).toFixed(2);
    qs("generalEnd").value = (end + shiftVal).toFixed(2);
    updateGeneralRaster();
  });
  qs("neuronSelect").addEventListener("change", async () => {
    updateNeuronInfo();
    await updateStprComparison();
    await updateSingleNeuron();
  });
}

async function init() {
  const resp = await fetchJSON(`${apiBase}/pids`);
  const pidSelect = qs("pidSelect");
  pidSelect.innerHTML = "";
  resp.pids.forEach((pid) => {
    const opt = document.createElement("option");
    opt.value = pid;
    opt.textContent = pid;
    pidSelect.appendChild(opt);
  });
  bindEvents();
  if (resp.pids.length > 0) {
    await loadSession(resp.pids[0]);
  }
}

init().catch((err) => console.error(err));
