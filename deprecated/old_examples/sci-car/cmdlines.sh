#######################
######### README ######
# Do not execute this file in one go. Best is to execute each command separately in a separate window and make sure all goes well
# You will need to change SCRIPT and DDIR variables below.
# You should first download the pre-processed data generated from raw Sci-CAR data, available at: schema.csail.mit.edu
#######################

DDIR=/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/processed/
SCRIPT=/afs/csail.mit.edu/u/r/rsingh/work/schema/examples/sci-car/scicar-process2.py


### read raw data and make Scanpy files ######################
# You do NOT need to run this. Its output is available at schema.csail.mit.edu

# $SCRIPT --mode raw_data_read 




### generate peak <-> gene (radial basis function) features ################
# NOTE: Pre-generated features (i.e. the output of this command) are available at the aforementioned site.
# The cmd takes a while to run; for each gene, it looks at all peaks when computing the feature values...

oldd=$PWD; cd $DDIR
$SCRIPT --mode produce_gene2fpeak --outsfx 20191218-1615 --njobs 36  --infile ${DDIR}/adata1x.h5ad
cd $oldd




### do Schema runs #######################
# We do Schema runs for the entire dataset ('0:12000' below), as well as for each quartile of genes ranked by expression variability
# After processing and filtering for sparsity, the dataset has a little under 12K genes

oldd=$PWD; cd $DDIR
for x in 0:12000 0:3000 3000:6000 6000:9000 9000:12000  #0:12000 corresponds to entire dataset; rest are quartiles
do
        minhvg=$(echo $x| cut -d: -f1);
        maxhvg=$(echo $x | cut -d: -f2);

	$SCRIPT --infile ${DDIR}/adata1x.h5ad --mode schema_gene2fpeak --outsfx 20200211-1600 --njobs 4 --extra adata_norm_style=2 gene2fpeak_file=adata1_M-produce_gene2fpeak_S-0_mtx_20191218-1615.csv min_hvgrank=$minhvg max_hvgrank=$maxhvg fpeak_cols_to_drop=fpeak_rbf_500 

done
cd $oldd



### measure the clustering of highly variable genes (HVGs) within topologically associating domains (TADs) ######
#
# with prep_significance_testing_data=1, each call to $SCRIPT produces a file of 1001 rows, of the following format:
#    - the first row corresponds to actual ('Orig') data while the remaining 1000 rows correspond to randomly shuffled instances
#    - each row has the format <desc>,<n>,<pair2_freq>,c_1,...,c_15,f_1,...,f_15
#        - desc = Orig|Random
#        - n = number of genes that were within a TAD. We limit our analysis to these, excluding genes that lie outside a TAD.
#              The "Random" instances shuffle n genes across k TADs, with n and k as determined from "Orig"
#        - pair2_freq = fraction of gene-pairs (denominator is n*(n-1)/2) that share a TAD
#        - c_i (i=1,...,15) = number of TADs that contain exactly 'i' genes
#        - f_i (i=1,...,15) = fraction of genes (denominator is n) in TADs that contain exactly 'i' genes

oldd=$PWD; cd $DDIR
for x in 0:3000 3000:6000 6000:9000 9000:12000  #0:3000 is top quartile
do
        minhvg=$(echo $x| cut -d: -f1);
        maxhvg=$(echo $x | cut -d: -f2);

	$SCRIPT --infile ${DDIR}/adata1x.h5ad --mode compute_hvg_tad_dispersion --outsfx t11 --extra min_hvgrank=$minhvg max_hvgrank=$maxhvg tad_locations_file=hg19_A549_TAD.bed prep_significance_testing_data=1

done
cd $oldd
