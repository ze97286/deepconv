#!/bin/bash

# Debug the tabix extraction for the specific region
ATLAS="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac_correlated.l3.bed"
PAT_FILE="/users/zetzioni/sharedscratch/loyfer_atlas/OAC/AB/cfDNA/069-001_ScrBsl_plasma_md.pat.gz"

echo "=== DEBUG TABIX EXTRACTION ==="

# Create test region for chr1:102506-102516
echo "Test region (CpG positions 102496-102526):"
echo -e "chr1\t102496\t102526" > test_region.bed
cat test_region.bed

echo -e "\nExtracting patterns with tabix:"
tabix -R test_region.bed "$PAT_FILE" | head -20

echo -e "\nAll patterns in CpG range 102502-102514:"
tabix "$PAT_FILE" chr1:102502-102514

echo -e "\nDone."
rm test_region.bed