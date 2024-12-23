#!/bin/bash

# src/deep_conv/atlas/preprocess_pats.sh 4 /users/zetzioni/sharedscratch/atlas/taps_atlas_class.csv /users/zetzioni/sharedscratch/atlas/pats/ 32

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
    echo "Input line: $1" >&2
    echo "Resolved: pat_file=$pat_file, group=$group, chr=$chr" >&2

    awk -v chr="$chr" -v group="$group" -v min_cpgs=$min_cpgs '
        BEGIN { FS="\t"; OFS="\t" }
        $1 == chr {
            pos = $2
            n = split($3, chars, "")
            for (i=1; i<=n; i++) {
                ct_count = 0
                for (j=i; j<=n; j++) {
                    if (chars[j] ~ /[CT]/) {
                        ct_count++
                    }
                }
                if (ct_count >= min_cpgs)
                    print chr, pos+i-1, group, $4
            }
        }
    ' <(zcat "$pat_file") > "$output_dir/tmp/${group}_chr${chr}_index.txt"

    if [[ -f "$output_dir/tmp/${group}_${chr}_index.txt" ]]; then
        echo "File written successfully: $output_dir/tmp/${group}_${chr}_index.txt" >&2
    else
        echo "Failed to write file for: $pat_file $group $chr" >&2
    fi
}
export output_dir="${output_dir%/}"
export -f process_file

# Run all tasks in parallel
cat $output_dir/tmp/tasks.txt | parallel -j $threads --progress process_file {}

# Cleanup temporary task list
rm $output_dir/tmp/tasks.txt

echo "Index generation complete! All files are saved with the format: <group>_chr<chr>_index.txt in $output_dir/tmp"