#!/bin/bash

# Test the complete pipeline from the script
INPUT_DIR="/users/zetzioni/sharedscratch/loyfer_atlas/OAC/AB/cfDNA"
MARKERBED="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac_correlated.l3.bed"
TMPDIR="/tmp/debug_full"

mkdir -p "$TMPDIR"

echo "=== TESTING FULL PIPELINE ==="

# Create regions file
regions_file="$TMPDIR/regions.bed"
tail -n+2 "$MARKERBED" | awk -v OFS="\t" '{start=$4-10; end=$5+10; if (start<0) start=0; print $1,start,end}' > "$regions_file"

# Test with specific pat file
pat_file="$INPUT_DIR/069-001_ScrBsl_plasma_md.pat.gz"
outfile="$TMPDIR/test_output.pat.gz"

echo "Testing complete pipeline on: $(basename "$pat_file")"

# Extract just our test region for focused testing
grep "102496" "$regions_file" > "$TMPDIR/test_region.bed"

echo -e "\nStep 1: Raw tabix extraction (should be ~47 patterns)"
raw_count=$(tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null | wc -l)
echo "Raw tabix count: $raw_count"

echo -e "\nStep 2: After sort (should be same count)"
sorted_count=$(tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null | sort -k1,1V -k2,2n -k3,3 | wc -l)
echo "After sort count: $sorted_count"

echo -e "\nStep 3: After bgzip (testing without bgzip first)"
tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null | sort -k1,1V -k2,2n -k3,3 > "$TMPDIR/test_unsorted.pat"
final_count=$(cat "$TMPDIR/test_unsorted.pat" | wc -l)
echo "Final count (uncompressed): $final_count"

echo -e "\nStep 4: Full pipeline with bgzip"
tabix -R "$TMPDIR/test_region.bed" "$pat_file" 2>/dev/null | \
    sort -k1,1V -k2,2n -k3,3 | \
    bgzip -c > "$outfile"

# Index it  
tabix -s 1 -b 2 -e 2 -C "$outfile"

# Check final result
echo "Final result after full pipeline:"
final_result_count=$(tabix "$outfile" chr1:102496-102526 2>/dev/null | wc -l)
echo "Final result count: $final_result_count"

if [ $final_result_count -lt 40 ]; then
    echo -e "\n!!! PIPELINE PROBLEM DETECTED !!!"
    echo "Expected ~47, got $final_result_count"
    echo "Showing final result patterns:"
    tabix "$outfile" chr1:102496-102526 2>/dev/null
fi

# Clean up
rm -rf "$TMPDIR"