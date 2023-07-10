#!/usr/bin/env python3

import setuptools
import glob
import os


def is_mac():
    version = os.uname().version
    sysname = os.uname().sysname

    return sysname == "Darwin" and "ARM64" in version


with open("README.md", "r") as fh:
    long_description = fh.read()


if is_mac():
    install_requires = [
        "biopython>=1.79",
        "pickle5",
        "scikit-learn==1.2.2",
        'numpy==1.24',
        "pandas",
        "click",
        "joblib",
        "loguru",
        "tensorflow-macos",
    ]

else:
    install_requires = [
        "biopython>=1.79",
        "pickle5",
        "scikit-learn==1.2.2",
        "numpy==1.21",
        "pandas",
        "click",
        "joblib",
        "loguru",
        "tensorflow==2.9.0",
    ]


packages = setuptools.find_packages()
print(packages)
package_data = {"phynteny_utils": ["phynteny_utils/*"]}

model_files = glob.glob("phynteny_utils/model/*")
data_files = [
    (".", ["LICENSE", "README.md"]),
    (
        "data",
        [
            "phynteny_utils/phrog_annotation_info/integer_category.pkl",
            "phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv",
            "phynteny_utils/phrog_annotation_info/phrog_integer.pkl",
            "phynteny_utils/phrog_annotation_info/confidence_kde.pkl",
        ]
        + model_files,
    ),
]

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
    scripts=["phynteny"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires=">=3.7",
)
