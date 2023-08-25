import click
from Bio import SeqIO


@click.command()
@click.argument("genbank_file", type=click.Path(exists=True))
@click.argument("gff_file", type=click.Path())
def genbank_to_gff(genbank_file, gff_file):
    # Parse GenBank file
    records = SeqIO.parse(genbank_file, "genbank")

    with open(gff_file, "w") as gff:
        for record in records:
            for feature in record.features:
                if feature.type != "source":  # Exclude the "source" feature
                    strand = "+" if feature.location.strand >= 0 else "-"
                    gff.write(
                        f"{record.id}\t.\t{feature.type}\t{feature.location.start}\t{feature.location.end}\t.\t{strand}\t.\t"
                    )

                    # Write the attributes (qualifiers) of the feature
                    qualifiers = []
                    for qualifier in feature.qualifiers:
                        qualifiers.append(
                            f"{qualifier}={feature.qualifiers[qualifier][0]}"
                        )
                    gff.write(";".join(qualifiers))
                    gff.write("\n")


if __name__ == "__main__":
    genbank_to_gff()
