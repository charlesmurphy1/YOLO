from setuptools import find_packages, setup

setup(
    name="yolo",
    version="0.1.0",
    description="Official Implementation of YOLO",
    author="Hao-Tang, Tsui",
    author_email="henrytsui000@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="LICENSE",
    python_requires=">=3.12",
    packages=find_packages(where=".", include=["yolo*"]),
    package_data={"yolo": ["**/*.yaml"]},
    include_package_data=True,
    install_requires=[
        "einops>=0.8.1",
        "faster-coco-eval>=1.6.5",
        "graphviz>=0.20.3",
        "hydra-core>=1.3.2",
        "lightning>=2.5.1",
        "loguru>=0.7.3",
        "numpy>=2.2.4",
        "omegaconf>=2.3.0",
        "opencv-python>=4.11.0.86",
        "pillow>=11.1.0",
        "pyarrow>=19.0.1",
        "pycocotools>=2.0.8",
        "requests>=2.32.3",
        "rich>=13.9.4",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "wandb>=0.19.8",
    ],
    extras_require={
        "dev": [
            "ipykernel>=6.29.5",
            "pyhectiqlab>=3.1.13",
        ]
    },
    entry_points={
        "console_scripts": [
            "yolo=yolo.lazy:main",
        ],
    },
)
