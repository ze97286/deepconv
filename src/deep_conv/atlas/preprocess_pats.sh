#!/bin/bash

# ./preprocess_pats.sh 4 /users/zetzioni/sharedscratch/atlas/taps_atlas_class.csv /users/zetzioni/sharedscratch/atlas/pats/ 32

# Arguments
min_cpgs=$1
class_file=$2
output_dir=$3
threads=${4:-32}  # Default to 32 threads

mkdir -p $output_dir/tmp

# Create task list for GNU parallel
while IFS=, read -r pat_file group; do
    [[ $pat_file == "name" ]] && continue
    pat_file=$(echo $pat_file | sed 's/\.beta/.pat.gz/' | sed 's/beta/pat\/Tissue/' | sed 's/\.calls\.all//')
    for chr in {1..22}; do
        echo "$pat_file,$group,chr$chr"
    done
done < $class_file > $output_dir/tmp/tasks.txt

# Process task
process_file() {
    IFS=, read pat_file group chr <<< "$1"
    awk -v chr="$chr" -v group="$group" -v min_cpgs=$min_cpgs '
        BEGIN { FS="\t"; OFS="\t" }
        $1 == chr {
            ct = gsub(/[CT]/, "&", $3)
            if (ct >= min_cpgs) {
                pos = $2
                n = split($3, chars, "")
                ct_count = 0
                for (i=1; i<=n; i++) {
                    if (chars[i] ~ /[CT]/) {
                        ct_count++
                        if (ct_count >= min_cpgs)
                            print chr, pos+i-1, group, $4
                    }
                }
            }
        }
    ' <(zcat "$pat_file")
}
export -f process_file

# Run all tasks in parallel
cat $output_dir/tmp/tasks.txt | parallel -j $threads --progress \
    "process_file {} > $output_dir/tmp/index.{#}.txt"

# Merge results per chromosome
for chr in {1..22}; do
    echo "Merging chromosome $chr..."
    find $output_dir/tmp -name "index.*.txt" -exec grep -h "^chr$chr" {} \; | \
        sort -k1,1 -k2,2n -k3,3 | \
        gzip > $output_dir/coverage_index_chr${chr}.txt.gz
done

# Cleanup
rm -rf $output_dir/tmp

echo "Done!"