from __future__ import annotations

from pygrabber.dshow_graph import FilterGraph


SUSPICIOUS_KEYWORDS = (
    "OBS",
    "Virtual",
    "ManyCam",
    "Snap Camera",
)


def get_camera_devices() -> list[str]:
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return list(devices)


def detect_suspicious_devices(devices: list[str]) -> dict:
    flagged_devices = [
        device
        for device in devices
        if any(keyword.lower() in device.lower() for keyword in SUSPICIOUS_KEYWORDS)
    ]

    return {
        "devices": devices,
        "suspicious": bool(flagged_devices),
        "flagged_devices": flagged_devices,
    }
