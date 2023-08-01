"""
Module for handling models.

Based on code from https://github.com/gbouras13/pharokka/blob/master/bin/databases.py
"""

#!/usr/bin/env python3
import os
import sys
import subprocess as sp
import pkg_resources
import re

PHYNTENY_MODEL_NAMES = ['grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_0.best_val_loss.h5',
    'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_1.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_2.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_3.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_4.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_5.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_6.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_7.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_8.best_val_loss.h5',
'grid_search_model.m_400.b_256.lr_0.0001.dr_0.1.l_2.a_tanh.o_rmsprop.rep_9.best_val_loss.h5'
]


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


def get_model_url():
    """
    Get the current url of the model
    """

    url_path = pkg_resources.resource_filename("phynteny_utils", "current_models.txt")

    with open(url_path, 'r') as file:
        url = file.readline().strip()
    return url


def get_model_zenodo(db_dir):
    """
    Download the phynteny model using the zenodo url

    :param db_dir: directory to install the models
    """
    print("Downloading Phynteny models")
    url = get_model_url()
    tarball = re.split('/',url)[-1]

    try:
        # remvoe the directory
        sp.call(["rm", "-rf", os.path.join(db_dir)])
        # make db dir
        sp.call(["mkdir", "-p", os.path.join(db_dir)])
        # download the tarball
        sp.call(["curl", url, "-o", os.path.join(db_dir,tarball)])
        # untar tarball into model directory
        sp.call(["tar", "-xzf", os.path.join(db_dir, tarball), "-C", db_dir, "--strip-components=1"])
        # remove tarball
        sp.call(["rm","-f", os.path.join(db_dir,tarball)])
    except:
        sys.stderr.write("Error: Phynteny model install failed. \n Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main \n downloading from " + url)
        return 0