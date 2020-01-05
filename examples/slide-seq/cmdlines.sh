DDIR=/afs/csail.mit.edu/u/r/rsingh/work/schema/data/slideseq/processed/
SCRIPT=/afs/csail.mit.edu/u/r/rsingh/work/schema/examples/slide-seq/slideseq-process1.py

### read raw data and make Scanpy files ###########################

$SCRIPT --mode raw_data_read



### do Schema runs ####################

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode schema_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outsfx ${DDIR}/kd_${pid}; done

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode schema_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outsfx ${DDIR}/kd_fit-on-all_kdbw-45_${pid} --extra kd_fit_granule_only=0 kd_bw=45 ; done



### do CCA runs #######################

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode cca_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outsfx ${DDIR}/ccakd_${pid}; done

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT --mode cca_kd_granule_cells --infile ${DDIR}/puck_${pid}.h5ad --outsfx ${DDIR}/ccakd_fit-on-all_kdbw-45_${pid} --extra kd_fit_granule_only=0 kd_bw=45 ; done



#### do Schema-CCA comparison plots ################

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT  --mode cca_schema_comparison_plot --outsfx ${DDIR}/schema-cca-cmp_${pid} --extra cca_pkl_file=${DDIR}/ccakd_${pid}_CCA_output.pkl schema_pkl_file=${DDIR}/kd_${pid}_func_output.pkl; done

for pid in 180430_1 180430_5 180430_6; do echo $pid; $SCRIPT  --mode cca_schema_comparison_plot --outsfx ${DDIR}/kd_fit-on-all_kdbw-45_schema-cca-cmp_${pid} --extra cca_pkl_file=${DDIR}/ccakd_fit-on-all_kdbw-45_${pid}_CCA_output.pkl schema_pkl_file=${DDIR}/kd_fit-on-all_kdbw-45_${pid}_func_output.pkl; done