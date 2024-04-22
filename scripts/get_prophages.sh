#!/bin/bash
""" 
Get set of a prophages from genbank files 
""" 

prophage_ids="/scratch/user/grig0076/poznan_phynteny/prophage_dataset_gLM/phynteny_data_dereplicated_names.txt"
prophages='/scratch/user/grig0076/phispy_phrogs/GCA/'
out_path='/home/grig0076/scratch/poznan_phynteny/prophage_dataset_gLM/phynteny_data_dereplicated_allgenbank_29112023/'
dash='/'
gbk='.gbk'

for s in `cat "$prophage_ids"`;
do echo $s

# break down the id into its constitutents 
lvl1="${s:4:3}" 
lvl2="${s:7:3}"
lvl3="${s:10:3}"

# get the path to the genbank file 
genbank_directory="${prophages}${lvl1}${dash}${lvl2}${dash}${lvl3}"
genbank_file=`ls "$genbank_directory"`
genbank_path="${genbank_directory}${dash}${genbank_file}"

# get the name of the locus 
locus="${s:14}"
locus="${locus::-6}"
p="${s: -4}"
locus="${locus}${p}"

echo $genbank_path 
echo $locus

# get the name of the output 
out="${out_path}${locus}${gbk}" 


# go amd get the prophage 
zcat "$genbank_path" | awk -v locus="$locus" '
  BEGIN { RS="//" }
  $0 ~ "LOCUS[[:space:]]+" locus {
    print $0
  }
;
'> "$out" 

done
