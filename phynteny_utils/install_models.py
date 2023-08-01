#!/usr/bin/env python3
from phynteny_utils import models
import click
import pkg_resources

@click.command()
@click.option('-o', '--outfile', type = click.Path(exists=True), help = "Path to install Phynteny models", default = None)

def main(outfile):

    if outfile == None:
        print('Downloading Phynteny models to the default location')
        db_dir = pkg_resources.resource_filename("phynteny_utils", "models")
        #TODO reorganise the models to also consider the confidence pickle object

    else:
        print('Downloading Phynteny models to ' + outfile)
        db_dir = outfile

    models.instantiate_install(db_dir)

if __name__ == "__main__":
    main()
