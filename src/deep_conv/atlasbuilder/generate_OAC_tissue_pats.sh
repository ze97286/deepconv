#!/bin/bash

PATDIR=$1
BAM_FILES=($2)
THREADS=16

mkdir -p "$PATDIR"
cd "$PATDIR"

echo "Processing these BAM files:"
parallel -j "$THREADS" '
  # Get the target pat file name
  outfile='"$PATDIR"'/{/.}.pat.gz
  if [ ! -e "$outfile" ]; then
    echo "Processing {}"
    /users/zetzioni/sharedscratch/wgbs_tools/wgbstools bam2pat \
      --no_beta --include_flags 67 --exclude_flags 3980 --clip 10 \
      -@ 2 \
      -o '"$PATDIR"' --genome hg38 {}; 
    zcat '"$PATDIR"'/{/.}.pat.gz | sed "y/TC/CT/" | bgzip -c > '"$PATDIR"'/{/.}.wgbs.pat.gz; 
    mv '"$PATDIR"'/{/.}.wgbs.pat.gz '"$PATDIR"'/{/.}.pat.gz; 
    tabix -f -s 1 -b 2 -e 2 -C '"$PATDIR"'/{/.}.pat.gz; 
  else
    echo "Skipping {} - output already exists"
  fi
' ::: "${BAM_FILES[@]}"