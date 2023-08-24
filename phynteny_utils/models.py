"""
Module for handling models.

Based on code from https://github.com/gbouras13/pharokka/blob/master/bin/databases.py
"""

import hashlib
#!/usr/bin/env python3
import os
import re
import shutil
import subprocess as sp
import sys
import tarfile
from pathlib import Path

import pkg_resources
import requests
from alive_progress import alive_bar

PHYNTENY_MODEL_NAMES = [
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_0.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_1.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_2.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_3.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_4.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_5.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_6.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_7.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_8.best_val_loss.h5",
    "grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_9.best_val_loss.h5",
]

VERSION_DICTIONARY = {
    "0.1.11": {
        "md5": "364c1eab8cda872a8757c1187a5c2b53",
        "db_url": "https://zenodo.org/record/8198288/files/phynteny_models_v0.1.11.tar.gz",
        "dir_name": "phynteny_models_zenodo",
    }
}


def instantiate_install(db_dir):
    """
    Begin model install

    :param: path to install the models
    """
    instantiate_dir(db_dir)
    downloaded_flag = check_db_installation(db_dir)
    if downloaded_flag == True:
        print("All phynteny models have already been downloaded and checked.")
    else:
        print("Some models are missing.")
        get_model_zenodo(db_dir)


def instantiate_dir(db_dir):
    """
    Create directory to download models

    :param db_dir: path to the model directory
    """

    if os.path.isdir(db_dir) == False:
        os.mkdir(db_dir)


def check_db_installation(db_dir):
    """
    Check that models have been installed

    :param db_dir: path to the models directory
    """

    downloaded_flag = True

    for file_name in PHYNTENY_MODEL_NAMES:
        path = os.path.join(db_dir, file_name)
        if os.path.isfile(path) == False:
            print("Phynteny models are missing.")
            downloaded_flag = False
            break

    return downloaded_flag


# def get_model_url():
#     """
#     Get the current url of the model
#     """

#     url_path = pkg_resources.resource_filename("phynteny_utils", "current_models.txt")

#     with open(url_path, "r") as file:
#         url = file.readline().strip()
#     return url


"""
lots of this code from the marvellous bakta https://github.com/oschwengers/bakta, db.py specifically
"""


def download(db_url: str, tarball_path: Path):
    try:
        with tarball_path.open("wb") as fh_out, requests.get(
            db_url, stream=True
        ) as resp:
            total_length = resp.headers.get("content-length")
            if total_length is not None:  # content length header is set
                total_length = int(total_length)
            with alive_bar(total=total_length, scale="SI") as bar:
                for data in resp.iter_content(chunk_size=1024 * 1024):
                    fh_out.write(data)
                    bar(count=len(data))
    except IOError:
        print(
            f"ERROR: Could not download file from Zenodo! url={db_url}, path={tarball_path}"
        )
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download from {db_url}. "
        )


def calc_md5_sum(tarball_path: Path, buffer_size: int = 1024 * 1024) -> str:
    """
    calculates the md5 for a tarball
    """
    md5 = hashlib.md5()
    with tarball_path.open("rb") as fh:
        data = fh.read(buffer_size)
        while data:
            md5.update(data)
            data = fh.read(buffer_size)
    return md5.hexdigest()


def remove_directory(dir_path):
    """
    removes directory if it exists
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def untar(tarball_path: Path, output_path: Path):
    """
    untars a tarball and saves in the output_path
    """
    try:
        with tarball_path.open("rb") as fh_in, tarfile.open(
            fileobj=fh_in, mode="r:gz"
        ) as tar_file:
            tar_file.extractall(path=str(output_path))

        tarpath = os.path.join(output_path, VERSION_DICTIONARY["0.1.11"]["dir_name"])

        # Get a list of all files in the source directory
        files_to_move = [
            f for f in os.listdir(tarpath) if os.path.isfile(os.path.join(tarpath, f))
        ]

        # Move each file to the destination directory
        for file_name in files_to_move:
            source_path = os.path.join(tarpath, file_name)
            destination_path = os.path.join(output_path, file_name)
            shutil.move(source_path, destination_path)
        # remove the directort
        remove_directory(tarpath)

    except OSError:
        print(f"Could not extract {tarball_path} to {output_path}")
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download from {db_url}. "
        )


def get_model_zenodo(db_dir):
    """
    Download the phynteny model using the zenodo url

    :param db_dir: directory to install the models
    """

    db_url = VERSION_DICTIONARY["0.1.11"]["db_url"]
    requiredmd5 = VERSION_DICTIONARY["0.1.11"]["md5"]

    tarball = re.split("/", db_url)[-1]
    tarball_path = Path(f"{db_dir}/{tarball}")

    # download the tarball
    print(f"Downloading Phynteny Models from {db_url}.")
    download(db_url, tarball_path)

    # check md5
    md5_sum = calc_md5_sum(tarball_path)

    if md5_sum == requiredmd5:
        print(f"Phynteny Models tarball download OK: {md5_sum}")
    else:
        print(f"Error: corrupt file! MD5 should be '{requiredmd5}' but is '{md5_sum}'")
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download from {db_url}. "
        )

    print(f"Extracting Phynteny Models tarball: file={tarball_path}, output={db_dir}")

    untar(tarball_path, db_dir)
    tarball_path.unlink()

    print(f"Done.")
