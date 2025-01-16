#!/bin/bash

INPUT_DIR=$1      # Directory containing merged pat files
PATDIR=$2         # Output directory
MARKERBED=$3      # Marker bed file
TMPDIR=/users/zetzioni/sharedscratch/atlas/tmp
TYPE2BAM=$4       # Type to bam mapping file

if [ ! -e $MARKERBED ]; then
    echo "Error: $MARKERBED does not exist!" 2>&1
    exit 1
fi

mkdir -p $PATDIR
mkdir -p $TMPDIR

# Process each cell type from the mapping file
cut -f 1 $TYPE2BAM | sort -u | while read celltype; do
    outfile="$PATDIR/${celltype}.pat.gz"
    pat_file="${INPUT_DIR}/${celltype}_merged.pat.gz"
    
    # Process if output doesn't exist
    if [ ! -e "$outfile" ]; then
        if [ ! -e "$pat_file" ]; then
            echo "Warning: Input file $pat_file does not exist!" 2>&1
            continue
        fi
        
        tabix -R <(tail -n+2 $MARKERBED | awk -v OFS="\t" '{start=$4-2; end=$5+2; if (start<0) start=0; print $1,start,end}') "$pat_file" | \
            sort -k1,1V -k2,2n -k3,3 | \
            bgzip -c > "$outfile"
            
        tabix -s 1 -b 2 -e 2 -C "$outfile"
    fi
done

rm -rf $TMPDIR