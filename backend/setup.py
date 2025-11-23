# setup.py
# TCMVE — Truth-Convergent Metaphysical Verification Engine
# @ECKHART_DIESTEL | DE | 2025-11-15 02:43 PM CET
# Install: pip install -e .

from setuptools import setup, find_packages
from pathlib import Path

# Load README for long description
README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="tcmve",
    version="1.0.0",
    description="Truth from Being — Zero-domain LLM verification via Thomistic metaphysics",
    long_description=README,
    long_description_content_type="text/markdown",
    author="@ECKHART_DIESTEL",
    author_email="eckhart.diestel@example.com",
    url="https://github.com/ediestel/tcmve",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-groq>=0.1.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "paper": [
            "matplotlib>=3.7",
            "pandas>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tcmve=tcmve:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Religion :: Philosophy",
    ],
    python_requires=">=3.10",
    keywords="llm verification thomism metaphysics truth convergence ai safety",
    project_urls={
        "Paper": "https://github.com/ediestel/tcmve/blob/main/main.pdf",
        "Documentation": "https://github.com/ediestel/tcmve/blob/main/README.md",
        "Source": "https://github.com/ediestel/tcmve",
        "Tracker": "https://github.com/ediestel/tcmve/issues",
    },
)

def main():
    """Entry point for `tcmve` command."""
    from tcmve import TCMVE
    tcmve = TCMVE()
    demo_query = "IV furosemide dose in acute HF for 40 mg oral daily?"
    result = tcmve.run(demo_query)
    print(result["tlpo_markup"])
