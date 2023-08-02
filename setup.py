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

def get_version():
    with open("VERSION", "r") as f:
        return f.readline().strip()


if is_mac():
    install_requires = [
        "biopython>=1.79",
        "pickle5",
        "scikit-learn<=1.2.2",
        'numpy',
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
        "scikit-learn<=1.2.2",
        "numpy",
        "pandas",
        "click",
        "joblib",
        "loguru",
        "tensorflow==2.9.1",
    ]


packages = setuptools.find_packages()
print(packages)
package_data = {
	"phynteny_utils": ["phynteny_utils/*", "phynteny_utils/models/*", "phynteny_utils/phrog_annotation_info/*"],
	"train_phynteny": ["train_phynteny/*"]
}

model_files = glob.glob("phynteny_utils/models/*")
data_files = [(".", ["LICENSE", "README.md"])]

setuptools.setup(
    name="phynteny",
    version=get_version(),
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
    entry_points={
        "console_scripts": [
            "generate_training_data=train_phynteny.generate_training_data:main",
            "train_model=train_phynteny.train_phyntenty:main",
            "compute_confidence=train_phynteny.compute_confidence:main",
            "install_models=phynteny_utils.install_models:main"
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires="<3.11",
)
