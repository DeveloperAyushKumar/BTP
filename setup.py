from setuptools import setup, find_packages

setup(
    name="ecovideo-studio",
    version="1.0.0",
    description="Energy-efficient video generation from images using Stable Video Diffusion on Apple Silicon",
    author="Aayush",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "diffusers==0.30.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "streamlit>=1.28.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "psutil>=5.9.0",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "numpy==1.26.4",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ecovideo=frontend.app:main",
        ],
    },
)
