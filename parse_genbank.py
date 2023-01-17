"""
Script to parse genbank output
"""

from genbank.file import File

gbank = File('test_data/pharokka.gbk')

print(gbank)

#parse in the various features etc
#think about the data structure used to store each prophage


