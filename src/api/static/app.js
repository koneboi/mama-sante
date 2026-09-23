"use strict";

const form = document.getElementById("triage-form");
const submit = document.getElementById("submit");
const banner = document.getElementById("banner");
const results = document.getElementById("results");
const riskBar = document.getElementById("risk-bar");
const riskValue = document.getElementById("risk-value");
const triageLabel = document.getElementById("triage-label");
const explain = document.getElementById("explain");
const thresholdValue = document.getElementById("threshold-value");
const thresholdNote = document.getElementById("threshold-note");
const factors = document.getElementById("factors");
const centerCard = document.getElementById("center-card");
const centerLevel = document.getElementById("center-level");
const centerName = document.getElementById("center-name");
const centerMeta = document.getElementById("center-meta");
const centerCost = document.getElementById("center-cost");
const centerMap = document.getElementById("center-map");

const LEVEL_COPY = {
  wait: ["Wait / annual re-screen", "wait"],
  local_scan: ["Schedule local ultrasound scan", "local"],
  regional_imaging: ["Refer to regional imaging (mammo)", "regional"],
  urgent_biopsy: ["Urgent referral — biopsy needed", "urgent"],
};

function note(msg, kind) {
  banner.textContent = msg;
  banner.className = kind || "";
}

function field(name) {
  const el = form.elements[name];
  if (!el) return "0";
  if (el.type === "checkbox") return el.checked ? "1" : "0";
  return el.value || "0";
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submit.disabled = true;
  results.hidden = true;
  centerCard.hidden = true;
  note("Screening patient…", "info");
  document.documentElement.scrollTop = document.body.scrollTop;

  const payload = {
    age: Number(field("age")),
    family_history: Number(field("family_history")),
    palpability: Number(field("palpability")),
    prior_biopsy: Number(field("prior_biopsy")),
    symptom_duration_months: Number(field("symptom_duration_months")),
    rural_flag: Number(field("rural_flag")),
    wealth_quartile: Number(field("wealth_quartile")),
    region: field("region"),
  };

  const file = form.elements.image.files[0];
  let suspicious = false;
  if (file) {
    try {
      const fd = new FormData();
      fd.append("image", file);
      const up = await fetch("/classify", { method: "POST", body: fd });
      const img = await up.json();
      if (img.suspicious) {
        suspicious = true;
        note("Image rejected by the quality gate (not a readable ultrasound), using risk-only triage", "warn");
      }
      if (img.class) payload.image_result = { confidence: img.confidence, above_floor: img.above_floor };
      if (img.error) note("Image analysis failed, using risk-only triage", "warn");
    } catch (err) {
      note("Image upload failed, using risk-only triage", "warn");
    }
  }

  const r = await postJSON("/triage", payload);
  submit.disabled = false;

  const [label, levelClass] = LEVEL_COPY[r.triage] || [r.triage, "wait"];
  triageLabel.textContent = label;
  triageLabel.className = "badge " + levelClass;

  const score = Math.round(r.risk_score * 100);
  riskValue.textContent = score;
  riskBar.style.width = score + "%";
  explain.textContent = suspicious
    ? "Risk-only referral — uploaded image was rejected by the quality gate."
    : (payload.image_result ? "Combined risk model + ultrasound image analysis." : "Risk-only referral (no image submitted).");

  thresholdValue.textContent = r.threshold_strategy === "group"
    ? r.risk_threshold.toFixed(2)
    : r.risk_threshold.toFixed(3);
  thresholdNote.textContent = r.threshold_strategy === "group"
    ? "(group-adjusted for region & wealth)"
    : "";
  thresholdValue.closest("p").hidden = false;

  factors.innerHTML = "";
  if (r.factors && r.factors.length) {
    const head = document.createElement("li");
    head.className = "factors-head";
    head.textContent = "Main drivers of this risk score";
    factors.appendChild(head);
    for (const f of r.factors.slice(0, 4)) {
      const li = document.createElement("li");
      const sign = f.contribution >= 0 ? "+" : "−";
      li.textContent = `${f.feature.replace(/_/g, " ")} · ${sign}${Math.abs(f.contribution).toFixed(3)}`;
      factors.appendChild(li);
    }
  }

  if (r.center) {
    const c = r.center;
    centerLevel.textContent = c.level;
    centerName.textContent = c.name;
    centerMeta.textContent =
      `${c.distance_km.toFixed(0)} km · ${Math.round(c.travel_h * 60)} min drive`;
    centerCost.textContent =
      `Estimated out-of-pocket: ${Math.round(c.out_of_pocket).toLocaleString()} FCFA`;
    centerMap.href = `https://www.openstreetmap.org/?mlat=${c.lat}&mlon=${c.lon}#map=8/${c.lat}/${c.lon}`;
    centerCard.hidden = false;
  }

  results.hidden = false;
  note("");
});