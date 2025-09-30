"""Detection pipelines for the ESP32-CAM."""

from .system import ESP32HumanDetectionSystem
from .optimized import OptimizedHumanDetector

__all__ = ["ESP32HumanDetectionSystem", "OptimizedHumanDetector"]

