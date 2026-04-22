/**
 * preload.js — Electron preload script
 * 
 * Exposes a safe IPC bridge from renderer to main process.
 * contextBridge ensures renderer cannot escape to Node.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("deepshield", {
  onTrustUpdate: (cb) => ipcRenderer.on("trust-update", (_, data) => cb(data)),
  onAlert:       (cb) => ipcRenderer.on("alert",        (_, data) => cb(data)),
  toggleDebug:   ()   => ipcRenderer.send("toggle-debug"),
  getVersion:    ()   => ipcRenderer.invoke("get-version"),
});
