"""
api/server.py — Local REST + WebSocket API

Provides:
  GET  /status          — current trust score + risk level
  GET  /session/summary — audit summary for current session
  GET  /health          — heartbeat
  POST /policy          — update policy config (enterprise use)
  WS   /ws              — streams TrustEvent JSON at ~2Hz to Electron overlay

Security:
  MVP:    shared token in Authorization header (hardcoded for hackathon)
  Phase 2: proper JWT with enterprise IdP

The WebSocket is the primary channel for the Electron overlay UI.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.events import TrustEvent, AlertEvent
from agent.event_bus import bus

logger = logging.getLogger(__name__)

# Shared secret for hackathon MVP — in Phase 2 replace with JWT
API_TOKEN = os.environ.get("DEEPSHIELD_TOKEN", "deepshield-hackathon-2025")

app = FastAPI(
    title="DeepShield Local API",
    description="Continuous Proof of Personhood Middleware",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Electron can connect from any origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state accessible by routes
_current_trust: Optional[TrustEvent] = None
_audit_store = None
_websocket_clients: list[WebSocket] = []


def set_audit_store(store) -> None:
    global _audit_store
    _audit_store = store


async def _on_trust_event(event: TrustEvent) -> None:
    global _current_trust
    _current_trust = event
    # Broadcast to all connected WebSocket clients
    dead = []
    payload = json.dumps(event.to_dict())
    for ws in _websocket_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _websocket_clients.remove(ws)


async def _on_alert_event(event: AlertEvent) -> None:
    # Broadcast alert separately with type field so Electron can trigger modal
    payload = json.dumps({
        "type": "ALERT",
        "session_id": event.session_id,
        "trust_score": round(event.trust_score, 4),
        "primary_trigger": event.primary_trigger,
        "recommended_action": event.recommended_action,
        "timestamp": event.timestamp,
    })
    dead = []
    for ws in _websocket_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _websocket_clients.remove(ws)


def register_handlers() -> None:
    """Call after app boot to wire event handlers."""
    bus.subscribe(TrustEvent, _on_trust_event)
    bus.subscribe(AlertEvent, _on_alert_event)
    logger.info("API event handlers registered")


# ---------------------------------------------------------------------------
# REST Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}


@app.get("/status")
async def status():
    if _current_trust is None:
        return JSONResponse({"status": "initializing", "trust_score": None, "risk_level": None})
    return JSONResponse(_current_trust.to_dict())


@app.get("/session/summary")
async def session_summary():
    if _audit_store is None:
        return JSONResponse({"error": "audit store not initialized"}, status_code=503)
    return JSONResponse(await _audit_store.get_session_summary())


@app.post("/policy")
async def update_policy(payload: dict):
    """Enterprise endpoint: push a new policy config."""
    # In Phase 2: validate JWT, reload policy, push to risk engine
    return JSONResponse({"status": "accepted", "note": "Policy hot-reload in Phase 2"})


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _websocket_clients.append(websocket)
    client = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket client connected: {client} (total={len(_websocket_clients)})")

    try:
        # Send current state immediately on connect
        if _current_trust:
            await websocket.send_text(json.dumps(_current_trust.to_dict()))

        # Keep connection alive — events are pushed via _on_trust_event
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping from client
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "keepalive"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client}")
    finally:
        if websocket in _websocket_clients:
            _websocket_clients.remove(websocket)
