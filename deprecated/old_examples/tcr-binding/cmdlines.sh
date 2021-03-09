#######################
######### README ######
# Do not execute this file in one go. Best is to execute each command separately in a separate window and make sure all goes well
# You will need to change SCRIPT and DDIR variables below.
# You should first download the pre-processed data generated from raw Slide-seq data, available at: schema.csail.mit.edu
#######################

#DDIR=/afs/csail.mit.edu/u/r/rsingh/work/schema/data/tcr-binding/processed/
#SCRIPT=/afs/csail.mit.edu/u/r/rsingh/work/schema/examples/tcr-binding/tcrbinding-process1.py

DDIR=/afs/csail.mit.edu/u/r/rsingh/work/schema/data/p2/tcr-binding/processed/
SCRIPT=/afs/csail.mit.edu/u/r/rsingh/work/schema/public/schema/examples/tcr-binding/tcrbinding-process1.py



### read raw data and make a HDF5 file ######################
# You do NOT need to run this. Its output is available at schema.csail.mit.edu

# $SCRIPT --mode raw_data_read 




### do Schema run to identify location-wise preferences in TCR alpha (tra) and TCR beta (trb) chains ####################
# Here, we're doing a 3-modality integration: a) CDR3 sequence + b) epitope binding-specificity - c) cell-surface protein markers.
#   The last one (c) above is put in with a -0.25 wt as a batch-effect correction [corresponding wt of (b) is +1.0]
#   Can also be run as a 2-modality problem by changing to "--mode compute_2_modality_columnwise_preference" below
# The output file produced has two columns: <location>,<score>.  Location is 0-indexed, and score indicates how likely is
#   it that the location displays low variablity. In the paper, we show 1-score, to indicate which locations are more variable.
# Separate runs for alpha and beta chains
#

oldd=$PWD; cd $DDIR
for c in tra trb
  do
       $SCRIPT --infile ${DDIR}/vdj_binarized_alldonors.h5 --mode compute_3_modality_columnwise_preference --outsfx _mode3_location_${c}_v1 --style 0 --extra chain=$c w_surface_markers=-0.25 > std1-mode3-${c}-wneg-style0.out 2>&1
done
cd $oldd





### do Schema run to identify amino acid selection pressure in TCR alpha (tra) and TCR beta (trb) chains ####################
# Here, we're doing a 3-modality integration: a) CDR3 sequence + b) epitope binding-specificity - c) cell-surface protein markers.
#   The last one (c) above is put in with a -0.25 wt as a batch-effect correction [corresponding wt of (b) is +1.0]
#   Can also be run as a 2-modality problem by changing to "--mode compute_2_modality_columnwise_preference" below
# The output file produced has two columns: <amino-acid-code>,<score>.  The score indicates how likely is
#   it that the amino acid is under selection pressure.

oldd=$PWD; cd $DDIR
$SCRIPT --infile ${DDIR}/vdj_binarized_alldonors.h5 --mode compute_3_modality_selection_pressure --outsfx _mode3_aa_v2 --extra kmer_type=std w_surface_markers=-0.25 > std2-mode3-${c}-wneg.out 2>&1
cd $oldd


