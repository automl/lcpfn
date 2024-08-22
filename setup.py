from setuptools import setup, find_packages

setup(
    name="lcpfn",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lcpfn=lcpfn.model:main",
        ],
    },
)
