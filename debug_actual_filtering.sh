#!/bin/bash

# Reproduce the exact filtering command being used
INPUT_DIR="/users/zetzioni/sharedscratch/loyfer_atlas/OAC/AB/cfDNA"
PATDIR="/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac_correlated.l3/AB/cfDNA"
MARKERBED="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac_correlated.l3.bed"
TMPDIR="/tmp/debug_filter"

mkdir -p "$TMPDIR"

echo "=== REPRODUCING ACTUAL FILTERING ==="

# Create regions file exactly as the script does
echo "Creating regions file..."
regions_file="$TMPDIR/regions.bed"
tail -n+2 "$MARKERBED" | awk -v OFS="\t" '{start=$4-10; end=$5+10; if (start<0) start=0; print $1,start,end}' > "$regions_file"

echo "Regions file content (first 10 lines):"
head -10 "$regions_file"

# Test with specific pat file
pat_file="$INPUT_DIR/069-001_ScrBsl_plasma_md.pat.gz"
echo -e "\nTesting tabix extraction on: $(basename "$pat_file")"

echo "Direct tabix for region chr1:102496-102526:"
tabix "$pat_file" chr1:102496-102526 2>/dev/null | wc -l

echo -e "\nTabix with -R regions file for our test region:"
# Extract just the line for our test region
grep "102496" "$regions_file" > "$TMPDIR/test_region.bed"
echo "Test region file contains:"
cat "$TMPDIR/test_region.bed"

echo -e "\nTabix -R result:"
tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null | wc -l
tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null

# Clean up
rm -rf "$TMPDIR"