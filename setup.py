"""
setup.py — ARCA build script.

Handles C++ extension (pybind11) with graceful fallback.
Prefer `pip install -e .` which uses pyproject.toml.
For C++ build: `pip install -e ".[cpp]" --no-build-isolation`
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os


class OptionalBuildExt(build_ext):
    """Build C++ extension but don't fail the whole install if it can't compile."""

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f"\n[ARCA] ⚠ C++ extension build failed: {e}")
            print("[ARCA] Falling back to pure-Python simulation (all functionality still works).\n")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
            print(f"[ARCA] ✓ C++ extension '{ext.name}' built successfully.")
        except Exception as e:
            print(f"[ARCA] ⚠ Could not build {ext.name}: {e}")
            print("[ARCA] Pure-Python fallback will be used.\n")


def get_ext_modules():
    try:
        import pybind11
        ext = Extension(
            "arca._cpp_sim",
            sources=["arca/cpp_ext/sim_engine.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-march=native", "-fvisibility=hidden"],
        )
        return [ext]
    except ImportError:
        print("[ARCA] pybind11 not found — skipping C++ extension.")
        return []


try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "ARCA — Autonomous Reinforcement Cyber Agent"


setup(
    name="arca-agent",
    version="0.1.0",
    author="Dipayan Dasgupta",
    author_email="ce24b059@smail.iitm.ac.in",
    description="Local RL-powered Autonomous Cyber Agent with LangGraph orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dipayandasgupta/arca",
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "scripts*"]),
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": OptionalBuildExt},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "gymnasium>=0.29",
        "stable-baselines3>=2.2",
        "torch>=2.0",
        "networkx>=3.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.0",
        "rich>=13.0",
        "typer>=0.12",
        "matplotlib>=3.8",
        "plotly>=5.20",
        "pandas>=2.0",
        "httpx>=0.27",
        "langchain>=0.2",
        "langchain-community>=0.2",
        "langgraph>=0.1",
        "langchain-core>=0.2",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov", "black", "ruff", "mypy"],
        "cpp": ["pybind11>=2.11"],
        "viz": ["dash>=2.16", "dash-cytoscape>=1.0"],
        "llm": ["ollama>=0.2"],
        "all": ["pybind11>=2.11", "dash>=2.16", "ollama>=0.2"],
    },
    entry_points={
        "console_scripts": [
            "arca=arca.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    keywords="reinforcement-learning cybersecurity pentesting autonomous-agent langgraph pybind11",
    include_package_data=True,
    package_data={"arca": ["configs/*.yaml", "data/*.json"]},
    zip_safe=False,
)