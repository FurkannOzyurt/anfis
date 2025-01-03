from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anfis",
    version="0.3.1",
    description="Python Adaptive Neuro Fuzzy Inference System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twmeggs/anfis",
    author="Tim Meggs",
    author_email="twmeggs@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-fuzzy",
    ],
    keywords=[
        "anfis",
        "fuzzy logic",
        "neural networks",
        "adaptive systems",
        "machine learning",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.6",
)

