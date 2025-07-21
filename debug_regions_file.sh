#!/bin/bash

ATLAS="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac_correlated.l3.bed"

echo "=== REGIONS FILE DEBUG ==="

echo "Atlas file first few lines:"
head -5 "$ATLAS"

echo -e "\nGenerated regions file:"
tail -n+2 "$ATLAS" | awk -v OFS="\t" '{start=$4-10; end=$5+10; if (start<0) start=0; print $1,start,end}' | head -10

echo -e "\nSpecific region for chr1:102506-102516:"
tail -n+2 "$ATLAS" | awk -v OFS="\t" '{start=$4-10; end=$5+10; if (start<0) start=0; print $1,start,end}' | grep "chr1" | grep -E "(10249|10250|10251|10252)"