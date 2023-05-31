"""
Python script to convert phynteny generate file to a table which describes the increase in annotation
"""

# imports
from phynteny_utils import handle_genbank
from phynteny_utils import format_data
import click
import sys
import pkg_resources

@click.command()
@click.argument("infile", type=click.Path(exists=True))
@click.option(
    "-o",
    "--outfile",
    type=click.Path(),
    help="where to write the output genbank file",
)

def main(infile, outfile):
    # get the absolute paths to phrog annotation files, model and thresholds
    print('STARTING', flush=True)
    phrog_categories = pkg_resources.resource_filename('phynteny_utils', 'phrog_annotation_info/phrog_integer.pkl')
    category_names = pkg_resources.resource_filename('phynteny_utils', 'phrog_annotation_info/integer_category.pkl')
    phrog_categories = format_data.get_dict(phrog_categories)
    category_names = format_data.get_dict(category_names)

    # reorganise the data
    print('reorganising base files', flush=True)
    phrog_categories = dict(zip([str(i) for i in list(phrog_categories.keys())], list(phrog_categories.values())))
    category_names[None] = 'unknown function'

    # get entries in the genbank file
    print('Reading the genbank file', flush=True)
    gb_dict = handle_genbank.get_genbank(infile)
    
    if not gb_dict:
        click.echo("Error: no sequences found in genbank file")
        sys.exit()
    keys = list(gb_dict.keys())
    

    print('looping through the genbank file', flush=True)
    with open(outfile, 'w') as f:
        f.write("start\tend\tstrand\tphrog_id\tphrog_category\tphynteny_category\tphage\n")

         
        for k in keys:
            #print(k, flush=True)
            cds = [f for f in gb_dict.get(k).features if f.type == 'CDS']

            # extract the features for the cds
            start = [c.location.start for c in cds]
            end = [c.location.end for c in cds]
            strand = [c.strand for c in cds]
            
            phrog = []
            
            for c in cds: 
                if 'phrog' in c.qualifiers.keys(): 
                    phrog.append(c.qualifiers.get('phrog')[0])
                else: 
                    phrog.append('No_PHROG')
             
            phynteny = [c.qualifiers.get('phynteny')[0] for c in cds]
            known_category = [category_names.get(phrog_categories.get(p)) for p in phrog]
            
            for i in range(len(cds)):
                f.write(f"{start[i]}\t{end[i]}\t{strand[i]}\t{phrog[i]}\t{known_category[i]}\t{phynteny[i]}\t{k}\n")
             
            
    print('saving', flush=True)

if __name__ == "__main__":
    main()
