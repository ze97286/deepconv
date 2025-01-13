#!/bin/bash

MIXEDPATS_DIR=$1    # Directory with mixed pat files
MARKERBED=$2        # Atlas markers bed file
OUTPUT_DIR=$3       # Where to store filtered pats

if [ ! -e $MARKERBED ]; then
    echo "Error: $MARKERBED does not exist!" 2>&1
    exit 1
fi

mkdir -p $OUTPUT_DIR

for pat_file in $MIXEDPATS_DIR/*_0.1_*_0.9_*.pat.gz; do
    if [ ! -e "$pat_file.tbi" ]; then
        tabix -s 1 -b 2 -e 2 -C "$pat_file"
    fi
done

# Process each mixed pat file
for pat_file in $MIXEDPATS_DIR/*.pat.gz; do
    outname=$(basename $pat_file)
    outfile="$OUTPUT_DIR/$outname"
    
    # Process if output doesn't exist
    if [ ! -e "$outfile" ]; then
        # Extract patterns near marker regions
        tabix -R <(tail -n+2 $MARKERBED | \
            awk '{start=$4-100; end=$5+100; if (start<0) start=0; print $1, start, end}' OFS="\t") \
            "$pat_file" | \
            sort -k1,1V -k2,2n -k3,3 | \
            bgzip -c > "$outfile"
        tabix -s 1 -b 2 -e 2 -C "$outfile"
    fi
done