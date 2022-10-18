"""
Calculate the crc64 checksum of a fasta sequence.
"""

import os
import sys
import argparse
from crc64iso.crc64iso import crc64

__author__ = 'Rob Edwards'

def stream_fasta(fastafile, whole_id=True):
    """
    Stream a fasta file, one read at a time. Saves memory!
    :param fastafile: The fasta file to stream
    :type fastafile: str
    :param whole_id: Whether to return the whole id (default) or just up to the first white space
    :type whole_id:bool
    :return:A single read
    :rtype:str, str
    """

    try:
        if fastafile.endswith('.gz'):
            f = gzip.open(fastafile, 'rt')
        elif fastafile.endswith('.lrz'):
            f = subprocess.Popen(['/usr/bin/lrunzip', '-q', '-d', '-f', '-o-', fastafile], stdout=subprocess.PIPE).stdout
        else:
            f = open(fastafile, 'r')
    except IOError as e:
        sys.stderr.write(str(e) + "\n")
        sys.stderr.write("Message: \n" + str(e.message) + "\n")
        sys.exit("Unable to open file " + fastafile)

    posn = 0
    while f:
        # first line should start with >
        idline = f.readline()
        if not idline:
            break
        if not idline.startswith('>'):
            sys.exit("Do not have a fasta file at: {}".format(idline))
        if not whole_id:
            idline = idline.split(" ")[0]
        idline = idline.strip().replace('>', '', 1)
        posn = f.tell()
        line = f.readline()
        seq = ""
        while not line.startswith('>'):
            seq += line.strip()
            posn = f.tell()
            line = f.readline()
            if not line:
                break
        f.seek(posn)
        yield idline, seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('-f', help='fasta file', required=True)
    parser.add_argument('-o', help='output file (default -)', default=sys.stdout)
    parser.add_argument('-v', help='verbose output', action='store_true')
    args = parser.parse_args()

    for seqid, seq in stream_fasta(args.f):
        print(f"{seqid}\t{crc64(seq)}")
