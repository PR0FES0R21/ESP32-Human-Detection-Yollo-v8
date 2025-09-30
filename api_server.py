"""Compatibility wrapper for esp32_cam.api.server."""

from esp32_cam.api.server import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()

