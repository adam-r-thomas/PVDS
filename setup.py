from setuptools import setup, find_packages

setup(
    name="evapsim",
    version="0.0.6",
    description="Evaporation simulator for thermal, ebeam, sputter",
    author="Adam R Thomas",
    classifiers=["Programming Language :: Python :: 3.10"],
    long_description=open("README.md").read(),
    install_requires=[
        "numpy", "numba", "PyQt5", "matplotlib", "pandas"
    ],
    license="AGPL-3.0",
    packages=["evapsim"]
    )
