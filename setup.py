import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="madre",
    version="0.0.1",
    author="Pedro Córdoba & Mario",
    author_email="pedrocorglez@gmail.com",
    description="A simple arithmetic package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pedrocg42/madre",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
