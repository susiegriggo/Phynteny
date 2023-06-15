#!/usr/bin/env python3

import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
print(packages) 
package_data = {"phynteny_utils": ["phynteny_utils/*"]}

model_files = glob.glob('phynteny_utils/model/*')
data_files = [(".", ["LICENSE", "README.md"]), ('data', ['phynteny_utils/phrog_annotation_info/integer_category.pkl', 'phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv', 'phynteny_utils/phrog_annotation_info/phrog_integer.pkl',  'phynteny_utils/phrog_annotation_info/confidence_kde.pkl']  + model_files)]

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
    scripts=["phynteny", "scripts/generate_training_data.py","scripts/train_crossvalidation.py" ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    install_requires=[  # TODO
        "biopython>=1.79",
        "pickle5",
        "scikit-learn",
        'numpy==1.21',
        "pandas",
        "click", 
        "joblib",
        "tensorflow==2.9.0"
    ],
    python_requires=">=3.7",
)

#may need to add nvidia-tensorrt==7.2.3.4 
