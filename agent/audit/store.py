"""
audit/store.py — Audit & Telemetry Layer

Writes immutable session records to a local SQLite database.
Called on AlertEvent and on session end.

Privacy note: we store hashed face embeddings (SHA-256), NOT raw video.
The audit trail can be exported as a signed JSON package for enterprise SIEM.

Schema:
  sessions table  — one row per session (start/end/peak_risk)
  alerts table    — one row per alert fired during a session
  score_trace     — time-series of trust scores (stored as JSON blob)
"""

from __future__ import annotations
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from agent.events import AlertEvent, TrustEvent
from agent.event_bus import bus

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "audit.db"


class AuditStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None
        self._score_trace: list = []
        self._peak_risk = "LOW"
        self._alert_count = 0
        self._session_id: Optional[str] = None
        self._start_time = time.time()

    async def initialize(self, session_id: str) -> None:
        self._session_id = session_id
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._create_tables()
        await self._insert_session()

        bus.subscribe(AlertEvent, self._on_alert)
        bus.subscribe(TrustEvent, self._on_trust_event)
        logger.info(f"AuditStore initialized (db={self.db_path})")

    async def _create_tables(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                peak_risk TEXT,
                alert_count INTEGER,
                score_trace TEXT,
                face_embedding_hash TEXT,
                policy_snapshot TEXT
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                trust_score REAL,
                primary_trigger TEXT,
                recommended_action TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        await self._db.commit()

    async def _insert_session(self) -> None:
        await self._db.execute("""
            INSERT OR IGNORE INTO sessions
            (session_id, start_time, end_time, peak_risk, alert_count, score_trace, face_embedding_hash, policy_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self._session_id,
            self._start_time,
            self._start_time,
            "LOW",
            0,
            "[]",
            "",
            "{}",
        ))
        await self._db.commit()

    async def _on_alert(self, event: AlertEvent) -> None:
        self._alert_count += 1
        self._peak_risk = "HIGH"

        await self._db.execute("""
            INSERT INTO alerts (session_id, timestamp, trust_score, primary_trigger, recommended_action)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.session_id,
            event.timestamp,
            round(event.trust_score, 4),
            event.primary_trigger,
            event.recommended_action,
        ))
        await self._db.commit()
        logger.info(f"Alert logged: trigger={event.primary_trigger}, score={event.trust_score:.3f}")

    async def _on_trust_event(self, event: TrustEvent) -> None:
        # Update peak risk
        risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if risk_rank.get(event.risk_level, 0) > risk_rank.get(self._peak_risk, 0):
            self._peak_risk = event.risk_level

        # Append to score trace (keep last 3600 entries ≈ 30min at 2Hz)
        self._score_trace.append((round(event.timestamp, 2), round(event.trust_score, 4)))
        if len(self._score_trace) > 3600:
            self._score_trace = self._score_trace[-3600:]

    async def close_session(self, face_embedding: Optional[bytes] = None) -> str:
        """Finalize session record. Returns session_id."""
        end_time = time.time()

        emb_hash = ""
        if face_embedding:
            emb_hash = hashlib.sha256(face_embedding).hexdigest()

        await self._db.execute("""
            UPDATE sessions
            SET end_time=?, peak_risk=?, alert_count=?, score_trace=?, face_embedding_hash=?
            WHERE session_id=?
        """, (
            end_time,
            self._peak_risk,
            self._alert_count,
            json.dumps(self._score_trace),
            emb_hash,
            self._session_id,
        ))
        await self._db.commit()
        await self._db.close()
        logger.info(f"Session closed: {self._session_id}, peak_risk={self._peak_risk}, alerts={self._alert_count}")
        return self._session_id

    async def get_session_summary(self) -> dict:
        """Return current session stats for API endpoint."""
        return {
            "session_id": self._session_id,
            "start_time": self._start_time,
            "duration_sec": round(time.time() - self._start_time, 1),
            "peak_risk": self._peak_risk,
            "alert_count": self._alert_count,
            "recent_scores": self._score_trace[-20:],
        }
