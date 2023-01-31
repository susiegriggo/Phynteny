#!/usr/bin/env python3
# encoding: utf-8
# copy this from metacoag 
from setuptools import setup, find_packages 

def readme(): 
    with open("README.md", "r") as fh:
        long_desc = fh.read() 
    return long_desc 

def get_version(): 
    with open("VERSION", "r") as f: 
        v = f.readline().strip()
        return v 

def main(): 
    setup(name='Phynteny',
        version=get_version(),
        long_description=readme(),
        description='Synteny-based bacteriophage annotation',
        url='https://github.com/susiegriggo/Phynteny',
        author='Susanna Grigson',
        scripts=['phynteny.py'],
        packages=find_packages(),
        entry_points={"console_scripts" : ["phynteny=phynteny.__main__:main"]},
        include_package_data=True, 
        author_email='susie.grigson@flinders.edu.au',
        license='MIT'
        )


if __name__ == "__main__":
    main()
    print(find_packages()) 
