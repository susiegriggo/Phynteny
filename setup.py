#!/usr/bin/env python3

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
package_data = {"phynteny": ["phynteny/*"]}

data_files = [(".", ["LICENSE", "README.md"])]

setuptools.setup(
    name="Phynteny",
    version="0",
    zip_safe=True,
    author="Susanna Grigson",
    author_email="susie.grigson@gmail.com",
    description="Phynteny: Synteny-based prediction of bacteriophage genes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/susiegriggo/Phynteny",
    license="MIT",
    packages=packages,
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    scripts=["phynteny.py"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    install_requires=[ #TODO
        "biopython",
        "tensorflow",
        "pickle",
        "pandas",
        "numpy",
    ],
    python_requires=">=3.7",
)