import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kg",
    version="0.0.1",
    author="Anonymized",
    description="UMCR: Unified Medical Concept Representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["kg"],
    python_requires=">=3.8",
)
