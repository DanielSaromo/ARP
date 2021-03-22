import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arpkeras",
    version="1.0.0",
    author="Daniel Saromo",
    author_email="danielsaromo@gmail.com",
    description="Auto-Rotating Perceptron implementation for Keras.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielSaromo/ARP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
