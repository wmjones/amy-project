from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# Read requirements from requirements.txt
def read_requirements(filename):
    with open(os.path.join(this_directory, filename)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="file-organizer-ai",
    version="1.0.0",
    author="File Organizer Team",
    author_email="team@file-organizer.ai",
    description="An intelligent file organization system using Claude AI to analyze and categorize documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/file-organizer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/file-organizer/issues",
        "Documentation": "https://github.com/yourusername/file-organizer/blob/main/docs/",
        "Source Code": "https://github.com/yourusername/file-organizer",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Topic :: Office/Business",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-mock>=3.12.0",
            "black>=24.3.0",
            "flake8>=7.0.0",
            "mypy>=1.9.0",
            "pre-commit>=3.5.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
            "opencv-python>=4.9.0",
            "pdf2image>=1.16.3",
        ],
        "dropbox": [
            "dropbox>=11.36.2",
        ],
        "all": [
            "pytesseract>=0.3.10",
            "opencv-python>=4.9.0",
            "pdf2image>=1.16.3",
            "dropbox>=11.36.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "file-organizer=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.json", "config/*.example"],
    },
    keywords="file organization ai claude document management automation",
)
