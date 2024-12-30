# /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/merge_pats.sh /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/files /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/celltypes
 
SOURCEDIR=$1
PATDIR=$2
BASEDIR=/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/Benchmark
MAPFILE=$BASEDIR/type2bam.tsv

mkdir -p $PATDIR

# Get unique cell types and properly quote them
celltypes=$(cut -f 1 $MAPFILE | sort -u | sed "s/.*/'&'/")

parallel -j $THREADS "if [ ! -e $PATDIR/{/.}.pat.gz ]; then 
    zcat \$(fgrep -w {} $MAPFILE | cut -f 2 | 
    sed 's:.*/:$SOURCEDIR/:' | 
    sed 's/_md.bam/.pat.gz/') | 
    sort -k1,1V -k2,2n -k3,3 | 
    perl -n deduplicate_pat.pl | 
    bgzip -c > $PATDIR/{/.}.pat.gz; 
    tabix -s 1 -b 2 -e 2 -C $PATDIR/{/.}.pat.gz; 
fi" ::: $celltypes