#!/usr/bin/env python3

from setuptools import setup

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]

setup(
    install_requires=parse_requirements('requirements.txt')
)
