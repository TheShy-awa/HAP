from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hap-kv",
    version="0.1.0",
    author="TheShy-awa",
    author_email="3781102822@qq.com",
    description="Hidden-state Aware Pruning for KV Cache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheShy-awa/HAP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
    ],
)
