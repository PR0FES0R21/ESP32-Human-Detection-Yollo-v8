from setuptools import setup, find_packages

setup(
    name="esp32-human-detection",
    version="1.0.0",
    description="Real-time human detection system for ESP32-CAM with smart recording",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.23.0",
    ],
    python_requires=">=3.8",
)
