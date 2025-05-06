from setuptools import setup, find_packages

setup(
    name="SolverEmulation",
    version="0.1.0",
    packages=find_packages(),
    description="Atmospheric modeling with graph neural networks",
    author="DeepCloud Team",
    install_requires=[
        "torch",
        "numpy",
        "xarray",
        "h5py",
    ],
) 