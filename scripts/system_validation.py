"""
scripts/system_validation.py
================================
Audit-grade system validation harness — v2.
Runs the DeepShield pipeline as a subprocess, parses its FUSION output
in real-time, and records FULLY ISOLATED per-scenario statistics.

Key guarantees (v2):
  - Logs are auto-cleared before every run (Task 1)
  - Each scenario switch resets ALL buffers and counters (Task 2)
  - Every frame is strictly tagged to one scenario (Task 3)
  - Report uses only per-scenario data, never mixed (Task 4)
  - Clear runtime prints every 30 frames (Task 5)
  - Scenarios with < 50 frames are marked INVALID (Task 6)

Usage:
  python scripts/reset_logs.py          # optional explicit clear
  python scripts/system_validation.py   # always auto-clears first

Scenario keys (type + ENTER):
  1 = REAL_WEBCAM
  2 = REPLAY
  3 = DEEPFAKE_VIDEO
  4 = OBS_ATTACK
  q = stop and generate report
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR      = Path("logs")
REPORT_PATH  = LOG_DIR / "system_validation_report.txt"
RAW_CSV_PATH = LOG_DIR / "system_validation_raw.csv"
SYS_LOG_PATH = LOG_DIR / "system_validation.log"

# ── Scenario config ───────────────────────────────────────────────────────────
SCENARIO_KEYS = {
    "1": "REAL_WEBCAM",
    "2": "REPLAY",
    "3": "DEEPFAKE_VIDEO",
    "4": "OBS_ATTACK",
}

MIN_FRAMES = 50          # below this → INVALID verdict

PASS_CRITERIA = {
    "REAL_WEBCAM":    {"df_max": 0.50, "safe_pct_min": 80, "spike_thresh": 0.70},
    "REPLAY":         {"cnn_high_pct_min": 60, "risk_within_sec": 10},
    "DEEPFAKE_VIDEO": {"df_avg_min": 0.50, "non_safe_pct_min": 50},
    "OBS_ATTACK":     {"safe_sustained_max_sec": 5},
}

# ── Regex ─────────────────────────────────────────────────────────────────────
_FUSION_RE = re.compile(
    r"FUSION:\s+gru=(?P<gru>[0-9.]+)\s+"
    r"cnn=(?P<cnn>[0-9.]+)\s+"
    r"df=(?P<df>[0-9.]+)\s+"
    r"(?:vg=[0-9.]+\s+vd=[0-9.]+\s+)?"          # optional variance fields
    r"score=(?P<score>[0-9.]+)\s+"
    r"smooth=(?P<smooth>[0-9.]+)\s+"
    r"status=(?P<status>\w+)\s+"
    r"reason='?(?P<reason>[^'\"]+)'?"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 999.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def _std(vals: list[float]) -> float:
    return _variance(vals) ** 0.5 if len(vals) >= 2 else 0.0


def _reset_logs_silent() -> None:
    """Delete all three log files silently (Task 1 — auto-reset)."""
    LOG_DIR.mkdir(exist_ok=True)
    for f in (RAW_CSV_PATH, REPORT_PATH, SYS_LOG_PATH):
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass


# ── Scenario state (fully isolated per switch) ────────────────────────────────

class ScenarioState:
    """
    All data for ONE scenario window.
    Created fresh on every scenario switch — no carryover. (Task 2)
    """

    def __init__(self, name: str):
        self.name      = name
        self.start_t   = time.monotonic()
        self.frames:   list[dict] = []       # raw frame rows

        # Consistency buffers (Task 2 — reset per scenario)
        self._gru_buf: deque[float] = deque(maxlen=25)
        self._df_buf:  deque[float] = deque(maxlen=25)

    def add_frame(self, t: float, gru: float, cnn: float, df: float,
                  score: float, smooth: float, status: str, reason: str) -> None:
        """Append one strictly-tagged frame. (Task 3)"""
        elapsed = round(t - self.start_t, 3)
        self.frames.append({
            "scenario": self.name,     # always tagged (Task 3)
            "elapsed":  elapsed,
            "gru":      gru,
            "cnn":      cnn,
            "df":       df,
            "score":    score,
            "smooth":   smooth,
            "status":   status,
            "reason":   reason,
        })
        self._gru_buf.append(gru)
        self._df_buf.append(df)

    # ── Statistics (Task 4 — only this scenario's data) ──────────────────────

    def stats(self) -> dict:
        n = len(self.frames)
        if n == 0:
            return {}

        grus    = [f["gru"]   for f in self.frames]
        cnns    = [f["cnn"]   for f in self.frames]
        dfs     = [f["df"]    for f in self.frames]
        scores  = [f["score"] for f in self.frames]
        statuses = [f["status"] for f in self.frames]

        status_counts: dict[str, int] = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        status_pct = {k: round(100 * v / n, 1) for k, v in status_counts.items()}

        # Time to first WARNING or HIGH_RISK
        t2risk = None
        for f in self.frames:
            if f["status"] in ("WARNING", "HIGH_RISK"):
                t2risk = f["elapsed"]
                break

        # Longest consecutive SAFE run in approximate seconds
        # (each FUSION tick ≈ 0.3 s)
        max_safe_run = cur = 0
        for f in self.frames:
            if f["status"] == "SAFE":
                cur += 1
                max_safe_run = max(max_safe_run, cur)
            else:
                cur = 0
        max_safe_sec = round(max_safe_run * 0.3, 1)

        spike_thresh = PASS_CRITERIA.get(self.name, {}).get("spike_thresh", 0.70)
        spikes = sum(1 for d in dfs if d > spike_thresh)

        return {
            "n":              n,
            "duration_sec":   round(self.frames[-1]["elapsed"], 1),
            "gru_avg":        round(sum(grus)   / n, 4),
            "cnn_avg":        round(sum(cnns)   / n, 4),
            "cnn_max":        round(max(cnns),   4),
            "df_avg":         round(sum(dfs)    / n, 4),
            "df_min":         round(min(dfs),   4),
            "df_max":         round(max(dfs),   4),
            "df_std":         round(_std(dfs),  4),
            "score_avg":      round(sum(scores) / n, 4),
            "status_pct":     status_pct,
            "time_to_risk_s": t2risk,
            "max_safe_run_s": max_safe_sec,
            "df_spikes":      spikes,
        }

    # ── Pass/fail ─────────────────────────────────────────────────────────────

    def verdict(self) -> tuple[str, list[str]]:
        n = len(self.frames)
        if n < MIN_FRAMES:
            return "INVALID", [f"only {n} frames collected (min {MIN_FRAMES} required)"]

        st    = self.stats()
        crit  = PASS_CRITERIA.get(self.name, {})
        fails = []

        if self.name == "REAL_WEBCAM":
            if st["df_max"] > crit["df_max"]:
                fails.append(f"df_max={st['df_max']:.3f} > {crit['df_max']} (domain mismatch)")
            if st["df_spikes"] > 0:
                fails.append(f"{st['df_spikes']} spikes above {crit['spike_thresh']:.2f}")
            safe_pct = st["status_pct"].get("SAFE", 0)
            if safe_pct < crit["safe_pct_min"]:
                fails.append(f"SAFE%={safe_pct:.1f} < {crit['safe_pct_min']}% required")

        elif self.name == "REPLAY":
            cnn_high = sum(1 for f in self.frames if f["cnn"] > 0.60)
            cnn_high_pct = 100 * cnn_high / n
            if cnn_high_pct < crit["cnn_high_pct_min"]:
                fails.append(f"CNN>0.60 frames: {cnn_high_pct:.0f}% < {crit['cnn_high_pct_min']}%")
            if st["time_to_risk_s"] is None:
                fails.append("never reached WARNING/HIGH_RISK")
            elif st["time_to_risk_s"] > crit["risk_within_sec"]:
                fails.append(f"time_to_risk={st['time_to_risk_s']}s > {crit['risk_within_sec']}s")

        elif self.name == "DEEPFAKE_VIDEO":
            if st["df_avg"] < crit["df_avg_min"]:
                fails.append(f"df_avg={st['df_avg']:.3f} < {crit['df_avg_min']}")
            non_safe_pct = 100 - st["status_pct"].get("SAFE", 100)
            if non_safe_pct < crit["non_safe_pct_min"]:
                fails.append(f"Non-SAFE%={non_safe_pct:.0f}% < {crit['non_safe_pct_min']}%")

        elif self.name == "OBS_ATTACK":
            if st["max_safe_run_s"] > crit["safe_sustained_max_sec"]:
                fails.append(
                    f"CRITICAL: sustained SAFE {st['max_safe_run_s']}s "
                    f"> {crit['safe_sustained_max_sec']}s"
                )

        return ("PASS" if not fails else "FAIL"), fails


# ── Background threads ────────────────────────────────────────────────────────

def reader_thread(proc: subprocess.Popen, queue: list, stop: threading.Event) -> None:
    for raw in iter(proc.stdout.readline, b""):
        if stop.is_set():
            break
        line = raw.decode("utf-8", errors="replace").rstrip()
        m = _FUSION_RE.search(line)
        if m:
            queue.append({
                "t":      time.monotonic(),
                "gru":    float(m.group("gru")),
                "cnn":    float(m.group("cnn")),
                "df":     float(m.group("df")),
                "score":  float(m.group("score")),
                "smooth": float(m.group("smooth")),
                "status": m.group("status"),
                "reason": m.group("reason"),
            })


def input_thread(scenario_signal: list, stop: threading.Event) -> None:
    """Blocking stdin reader — sets scenario_signal[0] on key press."""
    print(
        "\n[VALIDATION] Scenario keys:\n"
        "  1 = REAL_WEBCAM      (sit naturally, 60–90 sec)\n"
        "  2 = REPLAY           (hold phone/laptop to cam, 30 sec)\n"
        "  3 = DEEPFAKE_VIDEO   (play deepfake video, 30 sec)\n"
        "  4 = OBS_ATTACK       (OBS virtual cam ON, 60 sec)\n"
        "  q = finish + report\n"
    )
    while not stop.is_set():
        try:
            key = input("scenario> ").strip().lower()
        except EOFError:
            break
        if key in SCENARIO_KEYS:
            scenario_signal[0] = SCENARIO_KEYS[key]
        elif key in ("q", "quit", "done"):
            stop.set()
            break
        else:
            print(f"  Unknown key '{key}'. Use 1–4 or q.")


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(states: list[ScenarioState], timestamp: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("DEEPSHIELD SYSTEM VALIDATION REPORT")
    lines.append(f"Generated : {timestamp}")
    lines.append(f"Scenarios : {len(states)}")
    lines.append("=" * 70)

    lines.append(
        f"\n{'Scenario':<18} {'N':>5} {'df_avg':>7} {'df_max':>7} "
        f"{'cnn_avg':>7} {'SAFE%':>6} {'WARN%':>6} {'RISK%':>6} "
        f"{'t2risk':>7}  Verdict"
    )
    lines.append("-" * 92)

    all_verdicts: list[tuple[str, list[str]]] = []
    detail_lines: list[str] = []

    for state in states:
        n = len(state.frames)
        v, fails = state.verdict()
        all_verdicts.append((v, fails, state.name))

        if n < MIN_FRAMES:
            lines.append(f"  {state.name:<16}  {n:>5}  INVALID (< {MIN_FRAMES} frames)")
            detail_lines.append(f"\n{'─'*60}")
            detail_lines.append(f"SCENARIO: {state.name}  →  INVALID")
            detail_lines.append(f"  Frames collected: {n}  (need ≥ {MIN_FRAMES})")
            continue

        st = state.stats()
        safe_pct = st["status_pct"].get("SAFE",      0)
        warn_pct = st["status_pct"].get("WARNING",   0)
        risk_pct = st["status_pct"].get("HIGH_RISK", 0)
        t2r = f"{st['time_to_risk_s']:.1f}s" if st["time_to_risk_s"] is not None else "---"

        lines.append(
            f"  {state.name:<16} {n:>5} {st['df_avg']:>7.3f} {st['df_max']:>7.3f} "
            f"{st['cnn_avg']:>7.3f} {safe_pct:>6.1f} {warn_pct:>6.1f} {risk_pct:>6.1f} "
            f"{t2r:>7}  {v}"
        )

        detail_lines.append(f"\n{'─'*60}")
        detail_lines.append(f"SCENARIO: {state.name}  →  {v}")
        detail_lines.append(f"{'─'*60}")
        detail_lines.append(f"  Frames / duration    : {n} / {st['duration_sec']}s")
        detail_lines.append(f"  GRU avg              : {st['gru_avg']:.4f}")
        detail_lines.append(f"  CNN avg / max        : {st['cnn_avg']:.4f} / {st['cnn_max']:.4f}")
        detail_lines.append(f"  DF  avg              : {st['df_avg']:.4f}")
        detail_lines.append(f"  DF  min / max / std  : {st['df_min']:.4f} / {st['df_max']:.4f} / {st['df_std']:.4f}")
        detail_lines.append(f"  DF  spikes (>0.7)    : {st['df_spikes']}")
        detail_lines.append(f"  Score avg            : {st['score_avg']:.4f}")
        detail_lines.append(f"  SAFE%  / WARN%  / RISK%  : {safe_pct:.1f}% / {warn_pct:.1f}% / {risk_pct:.1f}%")
        detail_lines.append(f"  Max sustained SAFE   : {st['max_safe_run_s']}s")
        detail_lines.append(f"  Time to first risk   : {t2r}")
        if fails:
            detail_lines.append(f"  FAIL REASONS:")
            for f in fails:
                detail_lines.append(f"    ✗ {f}")

    lines.extend(detail_lines)

    # Fusion rule coverage (Task 4 — each scenario separately)
    lines.append(f"\n{'─'*60}")
    lines.append("FUSION RULE COVERAGE (per scenario)")
    lines.append(f"{'─'*60}")
    for state in states:
        if not state.frames:
            continue
        lines.append(f"  [{state.name}]")
        rule_counts: dict[str, int] = {}
        for f in state.frames:
            rule_counts[f["reason"]] = rule_counts.get(f["reason"], 0) + 1
        for reason, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / len(state.frames)
            lines.append(f"    {reason:<42}: {cnt:>4}  ({pct:.1f}%)")

    # Final verdict
    lines.append(f"\n{'='*70}")
    non_invalid = [(v, f, n) for v, f, n in all_verdicts if v != "INVALID"]
    if not non_invalid:
        lines.append("  FINAL VERDICT: NO VALID SCENARIOS (run again with ≥50 frames each)")
    elif all(v == "PASS" for v, _, _ in non_invalid):
        lines.append("  FINAL VERDICT: SYSTEM SAFE FOR DEMO")
        lines.append("  All tested scenarios passed.")
    else:
        lines.append("  FINAL VERDICT: CRITICAL GAP FOUND")
        for v, fails, name in all_verdicts:
            if v == "FAIL":
                lines.append(f"\n  [{name}]")
                for f in fails:
                    lines.append(f"    CRITICAL: {f}")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_csv(states: list[ScenarioState]) -> None:
    """Write all frames from all scenarios to one CSV. (Task 3 — each row tagged)"""
    LOG_DIR.mkdir(exist_ok=True)
    with open(RAW_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "scenario", "elapsed", "gru", "cnn", "df",
            "score", "smooth", "status", "reason",
        ])
        writer.writeheader()
        for state in states:
            writer.writerows(state.frames)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepShield system validation harness v2")
    p.add_argument("--device", type=int, default=0,
                   help="Webcam device index (0=webcam, 1/2=OBS)")
    p.add_argument("--python", default="venv310/Scripts/python.exe",
                   help="Python interpreter inside venv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Task 1: Auto-reset logs before every run ──────────────────────────────
    _reset_logs_silent()
    print("=" * 60)
    print("DEEPSHIELD SYSTEM VALIDATION HARNESS  v2")
    print("=" * 60)
    print("Logs cleared — starting fresh run.")
    print(f"Device: {args.device}  |  Launching pipeline...\n")

    # ── Launch pipeline ───────────────────────────────────────────────────────
    cmd = [args.python, "-m", "agent.main", "--debug-ui",
           f"--device={args.device}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0)
    print(f"[VALIDATION] Pipeline PID={proc.pid}\n")

    # ── Shared state ──────────────────────────────────────────────────────────
    queue:           list[dict] = []
    scenario_signal: list[str]  = ["REAL_WEBCAM"]   # mutable slot for thread comms
    stop_event = threading.Event()

    rt = threading.Thread(target=reader_thread,
                          args=(proc, queue, stop_event), daemon=True)
    it = threading.Thread(target=input_thread,
                          args=(scenario_signal, stop_event), daemon=True)
    rt.start()
    it.start()

    # ── Per-scenario tracking ─────────────────────────────────────────────────
    completed_states: list[ScenarioState] = []     # finished scenarios
    current_name  = scenario_signal[0]
    current_state = ScenarioState(current_name)     # Task 2: fresh state

    print(f"[SCENARIO SWITCH] → {current_name}")

    try:
        while not stop_event.is_set():
            # Drain parse queue
            while queue:
                row = queue.pop(0)

                # ── Task 2: Scenario switch → complete isolation ───────────────
                new_name = scenario_signal[0]
                if new_name != current_name:
                    n_collected = len(current_state.frames)
                    print(
                        f"\n[SCENARIO SWITCH] {current_name} ended "
                        f"({n_collected} frames collected)"
                    )
                    if n_collected >= MIN_FRAMES:
                        completed_states.append(current_state)
                    else:
                        print(
                            f"  [WARN] {current_name} had only {n_collected} frames "
                            f"(< {MIN_FRAMES}) — marking INVALID and discarding"
                        )
                        # Still keep it so report shows INVALID
                        completed_states.append(current_state)

                    # Task 2: Completely fresh state — zero carryover
                    current_name  = new_name
                    current_state = ScenarioState(current_name)
                    print(f"[SCENARIO SWITCH] → {current_name}\n")

                # ── Task 3: Tag frame to current scenario and record ───────────
                current_state.add_frame(
                    t=row["t"], gru=row["gru"], cnn=row["cnn"], df=row["df"],
                    score=row["score"], smooth=row["smooth"],
                    status=row["status"], reason=row["reason"],
                )

                # ── Task 5: Debug print every 30 frames ───────────────────────
                n = len(current_state.frames)
                if n % 30 == 0:
                    print(
                        f"[COLLECTING][SCENARIO={current_name[:12]:12s}] "
                        f"frames={n:4d}  "
                        f"gru={row['gru']:.3f}  cnn={row['cnn']:.3f}  "
                        f"df={row['df']:.3f}  status={row['status']}"
                    )

            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n[VALIDATION] Interrupted.")
        stop_event.set()

    # ── Finalise last scenario ────────────────────────────────────────────────
    if current_state.frames:
        completed_states.append(current_state)

    # Shutdown pipeline
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    # ── Generate outputs ──────────────────────────────────────────────────────
    if not completed_states:
        print("\n[VALIDATION] No data collected. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_csv(completed_states)
    print(f"[VALIDATION] Raw CSV  → {RAW_CSV_PATH}")

    report_text = build_report(completed_states, timestamp)
    print("\n" + report_text)

    LOG_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n[VALIDATION] Report   → {REPORT_PATH}")

    # Exit code: 0 = all PASS/INVALID, 1 = at least one FAIL
    has_fail = any(
        s.verdict()[0] == "FAIL"
        for s in completed_states
        if len(s.frames) >= MIN_FRAMES
    )
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
