#!/usr/bin/env python3

from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]

setup(
    name='mmanalysis',
    install_requires=parse_requirements('requirements.txt'),
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mmanalysis": ["data/*"],   # include all files in mmanalysis/data/
    },
)
