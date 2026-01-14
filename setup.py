from setuptools import setup, find_packages

setup(
    name="smart-reframe",
    version="0.1.0",
    description="Intelligent video reframing library (Landscape to Portrait) using AI.",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "tqdm",
        "mediapipe>=0.10.0",
        "librosa>=0.9.0",
        "filterpy>=1.4.5",
        "scipy>=1.7.0",
        "yt-dlp",  # Often useful for sourcing, though strictly core might not need it if reading local files. keeping for helper utils.
        "lap"      # For tracking
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
