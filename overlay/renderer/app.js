/**
 * app.js — DeepShield Overlay Renderer
 *
 * Connects to Python agent via WebSocket (proxied through Electron main).
 * Receives TrustEvent and AlertEvent payloads and updates the UI.
 *
 * UI Components managed:
 *  - Shield badge (color + pulse)
 *  - Trust score number + bar
 *  - Risk label
 *  - Signal grid cards (debug panel)
 *  - Sparkline chart (trust history)
 *  - Alert modal (on HIGH risk)
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────
const $shield      = document.getElementById("shield-icon");
const $riskLabel   = document.getElementById("risk-label");
const $trustScore  = document.getElementById("trust-score");
const $statusText  = document.getElementById("status-text");
const $scoreBar    = document.getElementById("score-bar");
const $debugPanel  = document.getElementById("debug-panel");
const $debugToggle = document.getElementById("debug-toggle");
const $signalGrid  = document.getElementById("signal-grid");
const $sparkline   = document.getElementById("sparkline");
const $sessionId   = document.getElementById("session-id");
const $badge       = document.querySelector(".badge-container");
const $iconCheck   = document.getElementById("icon-check");
const $iconExclaim = document.getElementById("icon-exclaim");

// Alert modal
const $alertModal    = document.getElementById("alert-modal");
const $alertReason   = document.getElementById("alert-reason");
const $alertSignals  = document.getElementById("alert-signals");
const $alertDismiss  = document.getElementById("alert-dismiss");
const $alertCd       = document.getElementById("alert-countdown");

// ── State ─────────────────────────────────────────────────────────────────
let debugOpen      = false;
let scoreHistory   = [];   // [{ts, score}]
let alertTimer     = null;
let alertCdInterval = null;
const SPARKLINE_POINTS = 60;  // ~30s at 2Hz

const SIGNAL_LABELS = {
  blink:                "Blink",
  head_pose:            "Head Pose",
  gaze:                 "Gaze",
  micro_expr:           "Micro-Expr",
  compression_artifact: "Compression",
  temporal_jitter:      "Temporal",
  texture_freq:         "Texture",
};

// ── Risk → style mapping ──────────────────────────────────────────────────
function riskClass(score) {
  if (score >= 0.80) return "ok";
  if (score >= 0.55) return "warn";
  return "danger";
}

function riskLevel(score) {
  if (score >= 0.80) return "LOW";
  if (score >= 0.55) return "MEDIUM";
  return "HIGH";
}

function riskLabelText(level) {
  if (level === "LOW")    return "VERIFIED";
  if (level === "MEDIUM") return "UNCERTAIN";
  return "DEEPFAKE";
}

// ── Main trust update handler ─────────────────────────────────────────────
function applyTrustUpdate(data) {
  if (!data.trust_score && data.type === "connected") {
    $statusText.textContent = "Connected — analyzing...";
    $badge.classList.remove("disconnected");
    return;
  }
  if (data.type === "disconnected") {
    $statusText.textContent = "Reconnecting to agent...";
    $badge.classList.add("disconnected");
    $trustScore.textContent = "–––";
    return;
  }

  const score    = data.trust_score ?? 0;
  const level    = data.risk_level  ?? riskLevel(score);
  const cls      = riskClass(score);
  const pct      = Math.round(score * 100);

  // Update shield
  $shield.className = `shield-icon ${cls}`;

  // Update icon glyph
  if (cls === "ok") {
    $iconCheck.classList.remove("hidden");
    $iconExclaim.classList.add("hidden");
  } else {
    $iconCheck.classList.add("hidden");
    $iconExclaim.classList.remove("hidden");
  }

  // Update score
  $trustScore.textContent = `${pct}%`;
  $trustScore.className   = `trust-score ${cls}`;

  // Update risk label
  $riskLabel.textContent = riskLabelText(level);
  $riskLabel.className   = `risk-label ${cls}`;

  // Update bar
  $scoreBar.style.width      = `${pct}%`;
  $scoreBar.style.background = `var(--clr-${cls === "ok" ? "low" : cls === "warn" ? "medium" : "high"})`;

  // Status text
  if (data.alert_reason) {
    $statusText.textContent = `⚠ ${data.alert_reason}`;
  } else {
    const signal_count = Object.keys(data.contributing_signals || {}).length;
    $statusText.textContent = `${signal_count} signals · ${level} risk`;
  }

  // Session ID
  if (data.session_id) {
    $sessionId.textContent = `session: ${data.session_id.slice(0, 8)}…`;
  }

  // Sparkline history
  scoreHistory.push({ ts: data.timestamp ?? Date.now() / 1000, score });
  if (scoreHistory.length > SPARKLINE_POINTS) scoreHistory.shift();
  drawSparkline();

  // Signal grid
  if (data.contributing_signals) {
    updateSignalGrid(data.contributing_signals);
  }
}

// ── Signal grid ───────────────────────────────────────────────────────────
function updateSignalGrid(signals) {
  const names = Object.keys(SIGNAL_LABELS);
  $signalGrid.innerHTML = "";

  names.forEach(name => {
    if (!(name in signals)) return;
    const score = signals[name];
    const cls   = riskClass(score);
    const pct   = Math.round(score * 100);

    const card = document.createElement("div");
    card.className = `signal-card ${cls}`;
    card.innerHTML = `
      <div class="signal-name">${SIGNAL_LABELS[name]}</div>
      <div class="signal-score-row">
        <span class="signal-score ${cls}">${pct}%</span>
        <div class="signal-bar-track">
          <div class="signal-bar ${cls}" style="width:${pct}%"></div>
        </div>
      </div>
    `;
    $signalGrid.appendChild(card);
  });
}

// ── Sparkline canvas ──────────────────────────────────────────────────────
function drawSparkline() {
  const canvas = $sparkline;
  const ctx    = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (scoreHistory.length < 2) return;

  const scores = scoreHistory.map(p => p.score);
  const minS = 0, maxS = 1;
  const pts = scores.map((s, i) => ({
    x: (i / (SPARKLINE_POINTS - 1)) * W,
    y: H - ((s - minS) / (maxS - minS)) * (H - 4) - 2,
  }));

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0.0, "rgba(0, 229, 160, 0.35)");
  grad.addColorStop(1.0, "rgba(0, 229, 160, 0.02)");

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) {
    const cp_x = (pts[i - 1].x + pts[i].x) / 2;
    ctx.bezierCurveTo(cp_x, pts[i - 1].y, cp_x, pts[i].y, pts[i].x, pts[i].y);
  }
  // Close path for fill
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke line
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) {
    const cp_x = (pts[i - 1].x + pts[i].x) / 2;
    ctx.bezierCurveTo(cp_x, pts[i - 1].y, cp_x, pts[i].y, pts[i].x, pts[i].y);
  }
  ctx.strokeStyle = "rgba(0, 229, 160, 0.8)";
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // Threshold lines
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = "rgba(251, 191, 36, 0.3)";
  ctx.lineWidth   = 1;
  const medY = H - 0.55 * (H - 4) - 2;
  ctx.beginPath(); ctx.moveTo(0, medY); ctx.lineTo(W, medY); ctx.stroke();

  const lowY = H - 0.80 * (H - 4) - 2;
  ctx.strokeStyle = "rgba(0, 229, 160, 0.3)";
  ctx.beginPath(); ctx.moveTo(0, lowY); ctx.lineTo(W, lowY); ctx.stroke();
  ctx.setLineDash([]);
}

// ── Alert modal ───────────────────────────────────────────────────────────
function showAlert(data) {
  clearAlertTimer();

  $alertReason.textContent  = `Primary trigger: ${(data.primary_trigger || "").replace(/_/g, " ")}`;
  $alertModal.classList.remove("hidden");

  // Build signal tags
  $alertSignals.innerHTML = "";
  if (data.primary_trigger) {
    const tag = document.createElement("span");
    tag.className   = "alert-signal-tag trigger";
    tag.textContent = (data.primary_trigger).replace(/_/g, " ");
    $alertSignals.appendChild(tag);
  }

  // Auto-dismiss countdown
  let countdown = 8;
  $alertCd.textContent = `Auto-dismiss in ${countdown}s`;
  alertCdInterval = setInterval(() => {
    countdown--;
    $alertCd.textContent = countdown > 0 ? `Auto-dismiss in ${countdown}s` : "";
    if (countdown <= 0) dismissAlert();
  }, 1000);

  alertTimer = setTimeout(dismissAlert, 8000);
}

function dismissAlert() {
  clearAlertTimer();
  $alertModal.classList.add("hidden");
}

function clearAlertTimer() {
  if (alertTimer)     { clearTimeout(alertTimer);  alertTimer = null; }
  if (alertCdInterval){ clearInterval(alertCdInterval); alertCdInterval = null; }
}

// ── Debug panel toggle ────────────────────────────────────────────────────
$debugToggle.addEventListener("click", () => {
  debugOpen = !debugOpen;
  $debugPanel.classList.toggle("hidden", !debugOpen);
  if (window.deepshield) window.deepshield.toggleDebug();
});

$alertDismiss.addEventListener("click", dismissAlert);

// ── Wire up IPC from Electron main ────────────────────────────────────────
if (window.deepshield) {
  window.deepshield.onTrustUpdate(applyTrustUpdate);
  window.deepshield.onAlert(showAlert);
} else {
  // Fallback: direct WebSocket for testing outside Electron
  console.warn("[DeepShield] Running without Electron preload — connecting directly");
  const ws = new WebSocket("ws://127.0.0.1:8765/ws");
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === "ALERT") showAlert(data);
      else applyTrustUpdate(data);
    } catch {}
  };
  ws.onopen  = () => applyTrustUpdate({ type: "connected" });
  ws.onclose = () => applyTrustUpdate({ type: "disconnected" });
}

// ── Init ──────────────────────────────────────────────────────────────────
$statusText.textContent = "Connecting to agent...";
