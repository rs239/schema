#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, copy, multiprocessing, traceback
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict
from tqdm import tqdm



def read_raw_files_and_write_h5_files(outdir):
    def f_atac1(w):
        try:
            v = w["chr"]
            assert ("_" not in v) and (v in ['X','Y'] or int(v)<=22) 
            return True
        except:
            return False

    import utils
    adata1 = utils.SciCar.loadData('/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car',
                                [('rna' ,'gene','GSM3271040', lambda v: v["cell_name"]=="A549", lambda v: v["gene_type"]=="protein_coding"),
                                 ('atac','peak','GSM3271041', lambda v: v["group"][:4]=="A549", f_atac1),
                                ],
                                   "/afs/csail.mit.edu/u/r/rsingh/work/refdata/hg19_mapped.tsv")
    adata1.write("{0}/adata1x.h5ad".format(outdir))

    
def mprun_geneXfpeak_mtx(adata, n_jobs=8):
    peak_func_list = [a[0] for a in SciCar.fpeak_list_all]
    print (adata.uns.keys(), adata.uns["atac.var"].head(2))
    chr_mapping = SciCar.getChrMapping(adata)


    nPeaks = adata.uns["atac.X"].shape[1]
    l = [list(a) for a in np.array_split(range(nPeaks), 5*n_jobs)]
    pool =  multiprocessing.Pool(processes = n_jobs)

    lx = pool.map(functools.partial(SciCar.computeGeneByFpeakMatrix, 
                                    adata, peak_func_list, chr_mapping, normalize_distwt= True),
                  l)


    g2p = None
    for m, _ in lx:
        if g2p is None:
            g2p = m
        else:
            g2p += m

    g2p =  g2p * (1e5/nPeaks)

    dx = pd.DataFrame(g2p, index=None)
    dx.columns = [a[1] for a in SciCar.fpeak_list_all]
    dx["gene"] = list(adata.var.index)
    dx["ensembl_id"] = list(adata.var.ensembl_id)               
    return dx




def f_helper_mprun_schemawts_2(args):
    ax, dz, dir1, outsfx, min_corr, maxwt, strand, chromosome, adata_norm_style = args

    if adata_norm_style == 1:
        ax_l2norm = np.sqrt(np.sum(ax**2, axis=1))
        ax = ax.copy() / (1e-12 + ax_l2norm[:,None])
        
    elif adata_norm_style == 2:
        ax_l2norm = np.sqrt(np.sum(ax**2, axis=1))
        ax = np.sort(ax.copy(), axis=1) / (1e-12 + ax_l2norm[:,None])
        print("Flag 231.10 ")
        print(ax[-10:,-10:])
        print(np.sum(ax**2,axis=1)[-10:])
        
    dz_cols = dz.columns[:-2]
    dz_vals = dz.values[:,:-2]
        
    vstd = np.std(dz_vals.astype(float), axis=0)
    print("Flag 231.0275 ", vstd.shape, dz_vals.shape, flush=True)
    dz_vals = dz_vals.copy() / (1e-12 + vstd)
            
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        import schema_qp
    except:
        from schema import schema_qp

    sqp = schema_qp.SchemaQP(min_corr, maxwt, params= {"dist_npairs": 1000000}, mode="scale")
    try:
        dz1 = sqp.fit_transform(dz_vals, [ax], ['feature_vector'], [1])
        
        print("Flag 231.030 ", min_corr, maxwt, dz_vals.shape, ax.shape, flush=True)
    
        wtsx = np.sqrt(np.maximum(sqp._wts/np.sum(sqp._wts), 0))
    except:
        print ("ERROR: schema failed for ", min_corr, maxwt, strand, chromosome)
        wtsx = 1e12*np.ones(dz_vals.shape[1])
    
    wdf = pd.Series(wtsx, index=dz_cols).sort_values(ascending=False).reset_index().rename(columns={"index": "fdist",0: "wt"})
    wdf.to_csv("{0}/adata1_sqp_wts_mincorr{1}_maxw{2}_strand{4}_chr{5}_adatanorm{8}_{7}.csv".format(dir1, min_corr, maxwt, 1, strand, chromosome, 5, outsfx, adata_norm_style), index=False)


    
def mprun_schemawts_2(adata1, dz, dir1, outsfx, adata_norm_style, do_dataset_split=False, n_jobs=4):
    
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        import schema_qp
    except:
        from schema import schema_qp
    
    pool =  multiprocessing.Pool(processes = n_jobs)
    try:
        ax = np.copy(adata1.X.todense().T)
    except:
        ax = adata1.X.T.copy()

    nGenes = adata1.var.shape[0]
    chrx = adata1.var["chr"].apply(lambda s: s.replace("chr",""))
    
    lx = []

    # one-time export of data for ridge-regression comparison
    # if False:
    #     dfax = pd.DataFrame(ax.T)
    #     dfax.columns = list(adata1.var_names)
        
    #     dz_vals = dz[dz.ensembl_id.isin(adata1.var["ensembl_id"])].values.copy()[:,:-2]
    #     vstd = np.std(dz_vals.astype(float), axis=0)
    #     dfdz = pd.DataFrame(dz_vals/ (1e-12 + vstd))
    #     dfdz.columns = list(dz.columns)[:-2]
    #     dfax.to_csv("saved_data_{0}_gexp.csv".format(outsfx), index=False)
    #     adata1.var.to_csv("saved_data_{0}_aux.csv".format(outsfx), index=False)
    #     dfdz.to_csv("saved_data_{0}_features.csv".format(outsfx), index=False)
    

    dz2 = dz
    for strand in ["both", "plus", "minus"]:
        if strand != "both" and do_dataset_split==False: continue
        
        gidx_strand = np.full(nGenes, True)
        if strand == "plus":  gidx_strand = np.where(adata1.var["strand"]=="+", True, False)
        if strand == "minus": gidx_strand = np.where(adata1.var["strand"]=="-", True, False)

        for chromosome in ["all", "1--8","9--16","17--23"]:
            if chromosome != "all" and do_dataset_split==False: continue

            gidx_chr = np.full(nGenes, True)
            if chromosome=="1--8":   gidx_chr = chrx.isin("1,2,3,4,5,6,7,8".split(","))
            if chromosome=="9--16":  gidx_chr = chrx.isin("9,10,11,12,13,14,15,16".split(","))
            if chromosome=="17--23": gidx_chr = chrx.isin("17,18,19,20,21,22,X,Y".split(","))

            for mc in [0.20, 0.5, 0.9]:
                for mw in [10,5,3]:

                    gidx = gidx_strand & gidx_chr
                    print("Flag 3312.040 ", gidx.shape, np.sum(gidx), ax.shape, dz2.shape, flush=True)
                    lx.append((ax[gidx,:], dz2[dz2.ensembl_id.isin(adata1.var["ensembl_id"][gidx])], dir1, outsfx,
                               mc, mw, strand, chromosome, adata_norm_style))
                    print("Flag 3312.050 ", np.sum(gidx), lx[-1][2:], flush=True)

    try:
        pool.map(f_helper_mprun_schemawts_2, lx)

    finally:
        pool.close()
        pool.join()




def  getGeneDistances(adata1):
    chrInt = adata1.var["chr"].apply(lambda s: s.replace("chr","").replace("X","231").replace("Y","232")).astype(int).values
    diff_chr = np.subtract.outer(chrInt, chrInt)!=0

    tss_adj = adata1.var['tss_adj'].values
    chrdist = np.abs(np.subtract.outer(tss_adj, tss_adj))

    import numpy.ma as ma
    gene_dist = ma.masked_where( diff_chr, chrdist).filled(1e15)
    nGenes = gene_dist.shape[0]
    l0 = list(range(nGenes))
    gene_dist[l0, l0] = 1e15
    return gene_dist


def  getGeneStrandMatch(adata1):
    chrInt = adata1.var["chr"].apply(lambda s: s.replace("chr","").replace("X","231").replace("Y","232")).astype(int).values
    diff_chr = np.subtract.outer(chrInt, chrInt)!=0

    strand_adj = np.where(adata1.var['strand'].values=="+",1,0)
    strandM = (np.subtract.outer(strand_adj, strand_adj) == 0).astype(int)

    import numpy.ma as ma
    gene_sign = ma.masked_where( diff_chr, strandM).filled(1e15)
    nGenes = gene_sign.shape[0]
    l0 = list(range(nGenes))
    gene_sign[l0, l0] = 1e15
    return gene_sign


def rankGenesByVariability(adata1):
    import matplotlib
    matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
    import scanpy as sc 
    nGenes = adata1.shape[1]
    print ("Flag 7868.200 ",   nGenes, adata1.X.shape, adata1.shape, scipy.sparse.isspmatrix(adata1.X))
    v = np.full(nGenes, nGenes)
    for i in range(500,nGenes-500,250):
        print ("Flag 7868.205 ",   i)
        
        v0 = sc.pp.highly_variable_genes(adata1, n_top_genes=i,inplace=False).highly_variable
        print ("Flag 7868.210 ",   i, np.sum(v0), len(v0))
        v = np.where((v==nGenes) & v0, i, v).copy()
    return v


def findNonVariableGenes(adata1, gene_window):
    nGenes = adata1.shape[1]

    v = np.full(nGenes, nGenes)
    for i in range(250,nGenes-500,250):
        try:
            hv = sc.pp.highly_variable_genes(adata1, n_top_genes=i,inplace=False).highly_variable
        except:
            print ("Flag 7867.500 ", i)
            traceback.print_exc(file=sys.stdout)
            continue

        v = np.where((v==nGenes) & hv, i, v).copy()


    w = (v  >= gene_window[0]) & (v < gene_window[1])
    print ("Flag 7867.200 ",  np.sum(w), gene_window, nGenes)
    print ("Flag 7867.300 ",  pd.Series(v).value_counts())

    return w



def random_gene_list(nGenes, gene_cnt, num_random_samples=1000):
    L  = []
    for i in range(num_random_samples):
        b = np.zeros(nGenes, dtype=bool)
        b[np.random.choice(nGenes, gene_cnt, replace=False)] = True
        L.append(b)

    return L
    

def  getGene2TADmap(adata1, tad):
    l = []
    nGenes = adata1.var.shape[0]
    for i in tqdm(range(nGenes)):
        gchr = adata1.var["chr"][i]
        gtss = adata1.var["tss_adj"][i]
        idx1 = (tad.chr==gchr) & (tad["start"] <= gtss) & (tad["end"] > gtss)
        #if i%100==0: print ("Flag 3231.10 ", i)
        if idx1.sum() > 0:
            l.append( np.argmax(idx1.values))
        else:
            l.append(np.NaN)
    g2tad = np.array(l)
    gene_sharedtad = np.subtract.outer( g2tad, g2tad)==0
    return g2tad, gene_sharedtad



def f_in_tad_cnt( g2tad, gene_set):
    return np.sum(~np.isnan(g2tad[gene_set]))


def f_mode_hvg_tad_dispersion_helper(g2tad, gene_sharedtad,  gene_set):
    tad2genes = defaultdict(list)
    for i, t in enumerate(g2tad):
        if not np.isnan(t) and gene_set[i]:
            tad2genes[t].append(i)
    return [(k,v,len(v)) for k,v in tad2genes.items()]


    
#################################################################################

if __name__ == "__main__":
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        from utils import SciCar
    except:
        import schema
        from schema.utils import SciCar

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details")
    parser.add_argument("--outdir", help="output directory (can set to '.')", type=str, default=".")
    parser.add_argument("--outsfx", help="suffix to use when producing output files")
    parser.add_argument("--style", help="mode-specific interpretation", type=int, default=-1)
    parser.add_argument("--infile", help="input .h5ad file. Default is Sci-Car hs A549")
    parser.add_argument("--njobs", help="number of parallel cores to use", type=int, default=24)

    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    
    args = parser.parse_args()
    assert args.mode is not None
    if args.mode !="raw_data_read": assert args.outsfx is not None

    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}

    if args.mode== "raw_data_read":
        read_raw_files_and_write_h5_files( args.outdir)



    if args.infile is None:
        args.infile = "/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/processed/adata1x.h5ad"


    adata1 = SciCar.loadAnnData(args.infile)
    adata1 = SciCar.preprocessAnnData(adata1, True, 5, 3, 5)

    if args.mode=="produce_gene2fpeak":
        if args.style < 0: args.style = 0
        dx = mprun_geneXfpeak_mtx(adata1, 36)
        dx.to_csv("{0}/adata1_M-{1}_S-{2}_mtx_{3}.csv".format( args.outdir, args.mode, args.style, args.outsfx), index=False)
        
        
    if args.mode=="schema_gene2fpeak":
        #  --mode schema_gene2fpeak --outsfx 20191218-1615 --njobs 5 --extra gene2fpeak_file=adata1_M-produce_gene2fpeak_S-0_mtx_20191218-1615.csv

        if args.style < 0: args.style = 0

        assert "gene2fpeak_file" in extra_args
        dz = pd.read_csv(extra_args["gene2fpeak_file"])

        if "fpeak_cols_to_drop" in extra_args:
            dz = dz.drop(columns = extra_args["fpeak_cols_to_drop"].split(","))

        adata1.var["gene_variability_ranking"] = rankGenesByVariability(adata1)
        
        adata1.X = adata1.X.todense()
        
        min_hvgrank, max_hvgrank = int(extra_args.get("min_hvgrank",0)), int(extra_args.get("max_hvgrank",1000000))
        adata1 = adata1[:, ((adata1.var["gene_variability_ranking"] >= min_hvgrank) & 
                            (adata1.var["gene_variability_ranking"] < max_hvgrank))]

        dz = dz[dz.ensembl_id.isin(adata1.var["ensembl_id"])].reset_index(drop=True)

        assert np.sum(dz.ensembl_id.values != adata1.var["ensembl_id"].values)==0

        adata_norm_style = int(extra_args.get("adata_norm_style",0))
        do_dataset_split = int(extra_args.get("do_dataset_split",0)) > 0.5
        
        mprun_schemawts_2(adata1, dz, args.outdir, "M-{0}_S-{1}_minhvg-{2}_maxhvg-{3}_{4}".format(args.mode, args.style, min_hvgrank, max_hvgrank, args.outsfx), adata_norm_style, do_dataset_split, args.njobs)



    if args.mode == "compute_hvg_tad_dispersion": 
        import scanpy as sc

        min_hvgrank, max_hvgrank = int(extra_args.get("min_hvgrank",0)), int(extra_args.get("max_hvgrank",1000000))
        gene_window = [min_hvgrank, max_hvgrank]

        outsfx = args.outsfx

        gdist_matrix = getGeneDistances(adata1)
        print ("Flag 54123.20 ", gdist_matrix.shape) #, describe(gdist_matrix))

        gsign_matrix = getGeneStrandMatch(adata1)
        print ("Flag 54123.21 ", gsign_matrix.shape) #, describe(gdist_matrix))

        tad = pd.read_csv(extra_args.get("tad_locations_file", "hg19_A549_TAD.bed"), delimiter="\t", header=None)
        tad.columns = ["chr","start","end", "x1","x2"]
        g2tad, gene_sharedtad = getGene2TADmap(adata1, tad)

        nGenes = adata1.shape[1]
        
        # v = np.full(nGenes, nGenes)
        # for i in tqdm(range(500,nGenes-500,500)):
        #     v0 = sc.pp.highly_variable_genes(adata1, n_top_genes=i,inplace=False).highly_variable
        #     v = np.where((v==nGenes) & v0, i, v).copy()
        # adata1.var["gene_variability_ranking"] = v

        adata1.var["gene_variability_ranking"] = rankGenesByVariability(adata1)
            
        window_genes = ((adata1.var["gene_variability_ranking"].values >= gene_window[0]) &  
                        (adata1.var["gene_variability_ranking"].values <   gene_window[1])) 

        n_window_genes = np.sum(window_genes)

        print ("Flag 54123.30 ", gene_window, len(window_genes), np.sum(window_genes))

        num_samples = 1000

        in_tad_cnt = f_in_tad_cnt(g2tad, window_genes)
        tad_total_cnt = np.sum(~np.isnan(g2tad))

        rl = random_gene_list(tad_total_cnt, in_tad_cnt, num_random_samples = num_samples)

        # we don't really care about genes outside TADs in this analysis
        window_genes [np.isnan(g2tad)] = False
        n_window_genes = np.sum(window_genes)
        lx0 = [window_genes]
        print ("Flag 54123.32 ",  len(window_genes), np.sum(window_genes), len(lx0))
        for rlx in rl:
            ax = np.full( nGenes, False)
            ax[~np.isnan(g2tad)] = rlx #[rlx] = True
            lx0.append(ax)
        print ("Flag 54123.33 ",  len(window_genes), np.sum(window_genes), len(lx0), len(lx0[1]), np.sum(lx0[1]))
            
        lx =  [(g2tad, gene_sharedtad, a) for a in lx0]


        print ("Flag 54123.40 ", len(lx), len(lx[1][2]), np.sum(lx[1][2]))
        
        n_jobs = 36
        print ("Flag 54123.435 ", nGenes, n_window_genes, gdist_matrix.shape, adata1.shape, len(lx))


        pool =  multiprocessing.Pool(processes = n_jobs)
        ly = pool.starmap(f_mode_hvg_tad_dispersion_helper, lx)

        outfile = "adata1_tad_membership_genewindow{0}-{1}_{2}".format(min_hvgrank, max_hvgrank, outsfx)
        outfh = open(outfile, 'w')

        prep_significance_testing_data = int(extra_args.get("prep_significance_testing_data",1)) > 0.5
        
        if not prep_significance_testing_data:
            random_samples_cnts = defaultdict(int)
            random_samples_N = 0
            for z in ly[1:]:
                for _, _, tadcnt in z: 
                    random_samples_cnts[tadcnt] += 1
                random_samples_N += 1.0

            for k,v in sorted(random_samples_cnts.items(), key = lambda a: a[0]):
                outfh.write("Random,{0},{1}\n".format(k, v/random_samples_N))

            for tadidx, tad_members, tadcnt in sorted(ly[0], key=lambda a: -a[2]):
                outfh.write("Orig,{0},{1},{2}\n".format( tad["x1"][tadidx], tadcnt, ",".join([adata1.var["Symbol"][t] for t in tad_members])))

        else:
            for i , z in enumerate(ly):
                #print ("Flag 54123.50 ", i, z)
                tadcnt2freq = defaultdict(int)
                numpairs_shared_tad = 0.0
                n = 0.0
                for _, _, tadcnt in z: 
                    tadcnt2freq[tadcnt] += 1
                    numpairs_shared_tad += tadcnt*(tadcnt-1)/2.0
                    n += tadcnt
                numpairs_all = n*(n-1)/2.0
                s = "Random" if i>0 else "Orig"
                s += ",{0},{1}".format(n, numpairs_shared_tad/numpairs_all)
                s += ",".join([""]+["{0}".format(tadcnt2freq[j]) for j in range(1,16)])
                s += ",".join([""]+["{0}".format(tadcnt2freq[j]*j/n) for j in range(1,16)])
                #print("Flag 54123.56 ", s)
                outfh.write(s + "\n")
            

        outfh.close()


