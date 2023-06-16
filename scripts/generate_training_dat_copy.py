import pickle
import re
import click
from phynteny_utils import handle_genbank
from phynteny_utils import format_data
from sklearn.model_selection import train_test_split
import pkg_resources
import numpy as np

@click.command()
@click.option(
    "--prefix",
    "-a",
    default='data',
    type=str,
    help="Prefix for the output files",
)


def say_hi(b):
    print(b)

def say_hello():
    print('hello')

def main(a):
    print('doing')
    say_hello()
    say_hi(a)

if __name__ == "__main__":
    main()

