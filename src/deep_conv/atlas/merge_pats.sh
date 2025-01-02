# /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/merge_pats.sh /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/files /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/celltypes /users/zetzioni/sharedscratch/atlas/type2bam.tsv
 
SOURCEDIR=$1
PATDIR=$2
MAPFILE=$3

mkdir -p $PATDIR

# Process each cell type separately
cut -f 1 $MAPFILE | sort -u | while read celltype; do
    if [ ! -e "$PATDIR/${celltype}.pat.gz" ]; then
        zcat $(fgrep -w "$celltype" $MAPFILE | cut -f 2 | \
            sed "s:.*/:$SOURCEDIR/:" | \
            sed 's/_md.bam/.pat.gz/') | \
            sort -k1,1V -k2,2n -k3,3 | \
            perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | \
            bgzip -c > "$PATDIR/${celltype}.pat.gz"
        tabix -s 1 -b 2 -e 2 -C "$PATDIR/${celltype}.pat.gz"
    fi
done