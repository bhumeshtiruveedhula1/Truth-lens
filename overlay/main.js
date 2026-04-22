/**
 * main.js — Electron Main Process
 *
 * Creates an always-on-top transparent overlay window.
 * Connects to Python agent WebSocket at ws://127.0.0.1:8765/ws
 * and forwards events to the renderer via IPC.
 *
 * Window is click-through by default so it doesn't block the call UI.
 * User can click the shield badge to toggle the debug panel.
 */

const { app, BrowserWindow, ipcMain, screen } = require("electron");
const path    = require("path");
const WebSocket = require("ws"); // bundled with Electron runtime via Node

const WS_URL     = "ws://127.0.0.1:8765/ws";
const RECONNECT_MS = 2000;
const APP_VERSION  = "1.0.0-hackathon";

let mainWindow = null;
let ws         = null;
let debugMode  = false;

// ---------------------------------------------------------------------------
// Window creation
// ---------------------------------------------------------------------------

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  // Overlay window: bottom-right corner, 320×180 compact badge size
  mainWindow = new BrowserWindow({
    width:  340,
    height: 180,
    x:      width  - 360,
    y:      height - 200,
    frame:         false,
    transparent:   true,
    alwaysOnTop:   true,
    skipTaskbar:   true,
    resizable:     false,
    hasShadow:     false,
    // Click-through: mouse events pass through the transparent areas
    // User can still click on opaque UI elements
    webPreferences: {
      preload:           path.join(__dirname, "preload.js"),
      contextIsolation:  true,
      nodeIntegration:   false,
    },
  });

  mainWindow.setAlwaysOnTop(true, "screen-saver");
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  // Allow click-through on transparent areas but capture clicks on UI
  mainWindow.setIgnoreMouseEvents(false);

  mainWindow.on("closed", () => { mainWindow = null; });
}

// ---------------------------------------------------------------------------
// WebSocket connection to Python agent
// ---------------------------------------------------------------------------

function connectWebSocket() {
  if (!mainWindow) return;

  ws = new WebSocket(WS_URL);

  ws.on("open", () => {
    console.log("[DeepShield] Connected to Python agent");
    if (mainWindow) mainWindow.webContents.send("trust-update", { type: "connected" });
    // Keepalive ping every 25s
    const ping = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send("ping");
      else clearInterval(ping);
    }, 25000);
  });

  ws.on("message", (raw) => {
    try {
      const data = JSON.parse(raw.toString());
      if (!mainWindow) return;
      if (data.type === "ALERT") {
        mainWindow.webContents.send("alert", data);
      } else if (data.type !== "keepalive" && data.type !== "connected") {
        mainWindow.webContents.send("trust-update", data);
      }
    } catch (e) { /* non-JSON message, ignore */ }
  });

  ws.on("close", () => {
    console.log("[DeepShield] WebSocket closed — reconnecting...");
    if (mainWindow) mainWindow.webContents.send("trust-update", { type: "disconnected" });
    setTimeout(connectWebSocket, RECONNECT_MS);
  });

  ws.on("error", (err) => {
    console.error("[DeepShield] WS error:", err.message);
    ws.terminate();
  });
}

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

ipcMain.on("toggle-debug", () => {
  debugMode = !debugMode;
  if (!mainWindow) return;
  // Resize window to show/hide debug panel
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  if (debugMode) {
    mainWindow.setSize(340, 420);
    mainWindow.setPosition(sw - 360, sh - 440);
  } else {
    mainWindow.setSize(340, 180);
    mainWindow.setPosition(sw - 360, sh - 200);
  }
});

ipcMain.handle("get-version", () => APP_VERSION);

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(() => {
  createWindow();
  // Small delay to let Python agent start up
  setTimeout(connectWebSocket, 1500);
});

app.on("window-all-closed", () => {
  if (ws) ws.terminate();
  app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
