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
    echo "Processing cell type: $celltype"
    outfile="$PATDIR/${celltype}.pat.gz"
    pat_file="${INPUT_DIR}/${celltype}_merged.pat.gz"
    
    # Process if output doesn't exist
    if [ ! -e "$outfile" ]; then
        if [ ! -e "$pat_file" ]; then
            echo "Warning: Input file $pat_file does not exist!" 2>&1
            continue
        fi
        
        echo "Creating regions file for tabix..."
        regions_file=$TMPDIR/${celltype}_regions.bed
        tail -n+2 $MARKERBED | awk -v OFS="\t" '{start=$4-2; end=$5-4; if (start<0) start=0; print $1,start,end}' > $regions_file
        
        echo "Running tabix for $celltype..."
        tabix -R $regions_file "$pat_file" | \
            sort -k1,1V -k2,2n -k3,3 | \
            bgzip -c > "$outfile"
            
        tabix_exit=$?
        if [ $tabix_exit -ne 0 ]; then
            echo "Error: tabix command failed for $celltype" 2>&1
            continue
        fi
        
        echo "Indexing output file..."
        tabix -s 1 -b 2 -e 2 -C "$outfile"
        
        # Clean up
        rm $regions_file
    else
        echo "Output file already exists: $outfile"
    fi
done

rm -rf $TMPDIR