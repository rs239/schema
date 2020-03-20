#######################
######### README ######
# Do not execute this file in one go. Best is to execute each command separately in a separate window and make sure all goes well
# You will need to change SCRIPT and DDIR variables below.
# You should first download the pre-processed data generated from raw Slide-seq data, available at: schema.csail.mit.edu
#######################

DDIR=/afs/csail.mit.edu/u/r/rsingh/work/schema/data/slideseq/processed/
SCRIPT=/afs/csail.mit.edu/u/r/rsingh/work/schema/examples/slide-seq/slideseq-process1.py



### read raw data and make Scanpy files ######################
# $SCRIPT --mode raw_data_read # You do NOT need to run this. Its output is available at schema.csail.mit.edu



### do Schema runs ####################

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode schema_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outpfx ${DDIR}/kd_fit-on-all_kdbw-45_${pid} --extra kd_fit_granule_only=0 kd_bw=45 ; done




### do CCA runs (2nd method, involving 2 steps) #######################

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode cca2step_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outpfx ${DDIR}/ccakd_fit-on-all_kdbw-45_${pid} --extra kd_fit_granule_only=0 kd_bw=45 ; done




### Generate gene-rankings per puck

oldd=$PWD;
cd $DDIR

for f in kd_fit-on-all_kdbw-45_180430_?_func_output.pkl cca2stepkd_fit-on-all_kdbw-45_180430_?_CCA2STEP_output.pkl
do
  sfx=$(echo $f | perl -pe 's/(180430_.).*$/\1/')
  pid=$(echo $f | perl -pe 's/^.*(180430_.).*$/\1/')
  typ=$(echo $f | awk '/CCA/ {print "cca"} !/CCA/ {print "schema"}')

  $SCRIPT --mode generate_multipuck_gene_ranks --outpfx per_puck_${sfx}  --extra data_type=$typ pkl_file_glob=./$f
done
cd $oldd




### Generate consensus gene-rankings by aggregating scores across 3 pucks.
###   same usage as above except we now specify globs instead of one file at a time 

oldd=$PWD;
cd $DDIR

for f in "kd_fit-on-all_kdbw-45_180430_?_func_output.pkl" "cca2stepkd_fit-on-all_kdbw-45_180430_?_CCA2STEP_output.pkl"
do
  sfx=$(echo $f | perl -pe 's/(180430_.).*$/\1/')
  pid=$(echo $f | perl -pe 's/^.*(180430_.).*$/\1/')
  typ=$(echo $f | awk '/CCA/ {print "cca"} !/CCA/ {print "schema"}')

  $SCRIPT --mode generate_multipuck_gene_ranks --outpfx per_puck_${sfx}  --extra data_type=$typ pkl_file_glob="./$f"
done
cd $oldd
