"""Compatibility wrapper for esp32_cam.detection.optimized."""

from esp32_cam.detection.optimized import OptimizedHumanDetector, main

__all__ = ["OptimizedHumanDetector", "main"]

if __name__ == "__main__":
    main()

