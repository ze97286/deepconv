# ./generate_pats /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song_pattools/files /users/zetzioni/sharedscratch/atlas/dmr_by_read.blood+gi+tum.250.l4.bed
PATDIR=$1
MARKERBED=$2 
TMPDIR=/users/zetzioni/sharedscratch/atlas/tmp
BASEDIR=/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/Benchmark

if [ ! -e $MARKERBED ]; then
  echo "Error: $MARKERBED does not exist!" 2>&1
  exit 1
fi

mkdir -p $PATDIR
mkdir -p $TMPDIR

parallel -j $THREADS "if [ ! -e $PATDIR/{/} ]; then tabix -R <(tail -n+2 $MARKERBED | cut -f 1,4,5) {} | sort -k1,1V -k2,2n -k3,3 | bgzip -c > $PATDIR/{/}; tabix -s 1 -b 2 -e 2 -C $PATDIR/{/}; fi" ::: `cut -f 2 $BASEDIR/type2bam.tsv | sed 's:/mnt/lustre/users/ramess/TAPSbeta_tissue_map/Results/1.3.1/Alignments/:/mnt/lustre/users/bschuster/Tissue_map/Results/1.2/pat:' | sed 's/_md.bam/.pat.gz/'`

rm -rf $TMPDIR