"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="nidn",
    version="0.1.1",
    description="A package for inverse material design of nanostructures using neural networks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esa/nidn",
    author="ESA Advanced Concepts Team",
    author_email="pablo.gomez@esa.int",
    include_package_data=True,
    install_requires=[
        "dotmap>=1.3.24",
        "loguru>=0.5.3",
        "matplotlib>=3.3.3",
        "numpy>=1.20.0",
        "pandas>=1.3.1",
        "scipy>=1.6.0",
        "tqdm>=4.56.1",
        "toml>=0.10.2",
        "torch>=1.9",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
    ],
    packages=[
        "nidn",
        "nidn.fdtd",
        "nidn.fdtd_integration",
        "nidn.materials",
        "nidn.plots",
        "nidn.tests",
        "nidn.training",
        "nidn.training.losses",
        "nidn.training.model",
        "nidn.training.utils",
        "nidn.trcwa",
        "nidn.trcwa.utils",
        "nidn.utils",
    ],
    python_requires=">=3.8, <4",
    project_urls={
        "Source": "https://github.com/esa/nidn/",
    },
)
