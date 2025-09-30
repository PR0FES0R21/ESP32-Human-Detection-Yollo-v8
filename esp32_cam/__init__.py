"""Core package for ESP32-CAM human detection utilities."""

from .config import Config
from .detection.system import ESP32HumanDetectionSystem
from .detection.optimized import OptimizedHumanDetector

__all__ = ["Config", "ESP32HumanDetectionSystem", "OptimizedHumanDetector"]

