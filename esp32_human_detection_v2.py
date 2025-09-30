"""Compatibility wrapper for esp32_cam.detection.system."""

from esp32_cam.detection.system import ESP32HumanDetectionSystem, main

__all__ = ["ESP32HumanDetectionSystem", "main"]

if __name__ == "__main__":
    main()

