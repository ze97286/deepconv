# /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_pats.sh /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/files /users/zetzioni/sharedscratch/atlas/dmr_by_read.blood+gi+tum.250.l4.bed /users/zetzioni/sharedscratch/atlas/type2bam.tsv
PATDIR=$1
MARKERBED=$2 
TMPDIR=/users/zetzioni/sharedscratch/atlas/tmp
TYPE2BAM=$3

if [ ! -e $MARKERBED ]; then
  echo "Error: $MARKERBED does not exist!" 2>&1
  exit 1
fi

mkdir -p $PATDIR
mkdir -p $TMPDIR

# Get input files and process them
cut -f 2 $TYPE2BAM | while read infile; do
    # Transform path for reading
    pat_file=$(echo $infile | \
        sed 's:/mnt/lustre/users/ramess/TAPSbeta_tissue_map/Results/1.3.1/Alignments/:/mnt/lustre/users/bschuster/Tissue_map/Results/1.2/pat/:' | \
        sed 's/_md.bam/.pat.gz/')
    
    # Get just the filename for output
    outname=$(basename $pat_file)
    outfile="$PATDIR/$outname"
    
    # Process if output doesn't exist
    if [ ! -e "$outfile" ]; then
        tabix -R <(tail -n+2 $MARKERBED | awk '{start=$4-100; end=$5+100; if (start<0) start=0; print $1, start, end}' OFS="\t") "$pat_file" | \
            sort -k1,1V -k2,2n -k3,3 | \
            bgzip -c > "$outfile"
        tabix -s 1 -b 2 -e 2 -C "$outfile"
        
    fi
done

rm -rf $TMPDIR