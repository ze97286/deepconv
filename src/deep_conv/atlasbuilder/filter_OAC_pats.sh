#!/bin/bash

INPUT_DIR=$1      # Directory containing pat files
PATDIR=$2         # Output directory
MARKERBED=$3      # Marker bed file
TMPDIR=/users/zetzioni/sharedscratch/atlas/tmp

if [ ! -e "$MARKERBED" ]; then
    echo "Error: $MARKERBED does not exist!" 2>&1
    exit 1
fi

mkdir -p "$PATDIR"
mkdir -p "$TMPDIR"

# Create regions file once since it's the same for all files
echo "Creating regions file for tabix..."
regions_file="$TMPDIR/regions.bed"
tail -n+2 "$MARKERBED" | awk -v OFS="\t" '{start=$4-10; end=$5+10; if (start<0) start=0; print $1,start,end}' > "$regions_file"

# Re-index input files if needed
echo "Checking and re-indexing input files if needed..."
for pat_file in "$INPUT_DIR"/*.pat.gz; do
    if [ ! -e "$pat_file.tbi" ] && [ ! -e "$pat_file.csi" ]; then
        echo "Indexing $pat_file..."
        tabix -s 1 -b 2 -e 2 -C "$pat_file"
    elif [ "$pat_file.csi" -ot "$pat_file" ] || [ "$pat_file.tbi" -ot "$pat_file" ]; then
        echo "Re-indexing stale index for $pat_file..."
        tabix -f -s 1 -b 2 -e 2 -C "$pat_file"
    fi
done

# Process each pat file in input directory
for pat_file in "$INPUT_DIR"/*.pat.gz; do
    # Get basename for output file
    basename=$(basename "$pat_file")
    outfile="$PATDIR/$basename"
    
    # Check if output exists and is both non-empty and indexed
    if [ ! -e "$outfile" ] || [ ! -e "$outfile.tbi" ] || [ ! -s "$outfile" ]; then
        echo "Processing file: $basename"
        
        echo "Running tabix..."
        tabix -R "$regions_file" "$pat_file" | \
            sort -k1,1V -k2,2n -k3,3 | \
            bgzip -c > "$outfile"
            
        tabix_exit=$?
        if [ $tabix_exit -ne 0 ]; then
            echo "Error: tabix command failed for $basename" 2>&1
            continue
        fi
        
        echo "Indexing output file..."
        tabix -s 1 -b 2 -e 2 -C "$outfile"
    else
        echo "Skipping $basename - output already exists and is indexed"
    fi
done

# Clean up
rm -rf "$TMPDIR"