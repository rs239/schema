#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, copy
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict




def write_h5_files():
    def f_atac1(w):
        try:
            v = w["chr"]
            assert ("_" not in v) and (v in ['X','Y'] or int(v)<=22) 
            return True
        except:
            return False

    def f_atac2(w):
        try:
            v = w["chr"]
            assert v[:3]=="chr" and ("_" not in v) and (v[3] in ['X','Y'] or int(v[3:])<=22) 
            return True
        except:
            return False

    adata1 = utils.SciCar.loadData('/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car',
                                [('rna' ,'gene','GSM3271040', lambda v: v["cell_name"]=="A549", lambda v: v["gene_type"]=="protein_coding"),
                                 ('atac','peak','GSM3271041', lambda v: v["group"][:4]=="A549", f_atac1),
                                ],
                                   "/afs/csail.mit.edu/u/r/rsingh/work/refdata/hg19_mapped.tsv")
    adata2 = utils.SciCar.loadData('/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car',
                                [('rna' ,'gene','GSM3271044', None, lambda v: v["gene_type"]=="protein_coding"),
                                 ('atac','peak','GSM3271045', None, f_atac2),
                                ],
                                "/afs/csail.mit.edu/u/r/rsingh/work/refdata/mm10_mapped.tsv")
    adata1.write("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
    adata2.write("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata2.h5ad")

    
def mprun_geneXfpeak_mtx(adata, n_jobs=8, style=1):
    peak_func_list = [a[0] for a in SciCar.fpeak_list_all]
    print (adata.uns.keys(), adata.uns["atac.var"].head(2))
    chr_mapping = SciCar.getChrMapping(adata)

    assert style in [1,2,3]

    nPeaks = adata.uns["atac.X"].shape[1]
    l = [list(a) for a in np.array_split(range(nPeaks), 5*n_jobs)]
    #l = [list(a) for a in np.array_split(range(1000), 5*n_jobs)]
    #print("Flag 3343.100 ", l)
    pool =  multiprocessing.Pool(processes = n_jobs)

    if style == 1:
        lx = pool.map(functools.partial(SciCar.computeGeneByFpeakMatrix, 
                                        adata, peak_func_list, chr_mapping, normalize_distwt= True),
                      l)

    elif style == 2:
        lx = pool.map(functools.partial(SciCar.computeGeneByFpeakMatrix, 
                                        adata, peak_func_list, chr_mapping, normalize_distwt= False),
                      l)

    elif style == 3:
        lx = pool.map(functools.partial(SciCar.computeGeneByFpeakMatrix, 
                                        adata, peak_func_list, chr_mapping, normalize_distwt= False, booleanPeakCounts = True),
                      l)


    # # lx = []
    # # for z in l:
    # #     lx.append(SciCar.computeGeneByFpeakMatrix(adata, peak_func_list, chr_mapping, z))


    if style == 1:
        g2p = None
        for m, _ in lx:
            if g2p is None:
                g2p = m
            else:
                g2p += m

        g2p =  g2p * (1e5/nPeaks)

    elif style==2 or style==3:
        g2p = None
        g2p_wts = None
        for m, m_wts in lx:
            if g2p is None:
                g2p = m
                g2p_wts = m_wts
            else:
                g2p += m
                g2p_wts += m_wts
        g2p =  (g2p / g2p_wts) * (1e5)
        

    dx = pd.DataFrame(g2p, index=None)
    dx.columns = [a[1] for a in SciCar.fpeak_list_all]
    dx["gene"] = list(adata.var.index)
    dx["ensembl_id"] = list(adata.var.ensembl_id)               
    return dx



def f_helper_mprun_schemawts_1(args):
    ax, dz, dir1, outsfx, min_corr, maxwt, use_first_col = args
    if use_first_col:
        dz_cols = dz.columns[:-2]
        dz_vals = dz.values[:,:-2]
    else:
        dz_cols = dz.columns[1:-2]
        dz_vals = dz.values[:,1:-2]

    import schema_qp
    sqp = schema_qp.SchemaQP(min_corr, maxwt, params= {"dist_npairs": 1000000}, mode="scale")
    dz1 = sqp.fit_transform(dz_vals, [ax], ['feature_vector'], [1])

    wdf = pd.Series(sqp._wts, index=dz_cols).sort_values(ascending=False).reset_index().rename(columns={"index": "fdist",0: "wt"})
    wdf.to_csv("{0}/adata1_sqp_wts_mincorr{1}_maxw{2}_usefirstcol{3}_{4}.csv".format(dir1, min_corr, maxwt, 1 if use_first_col else 0, outsfx), index=False)


def f_helper_mprun_schemawts_2(args):
    ax, dz, dir1, outsfx, min_corr, maxwt, use_first_col, strand, chromosome, gene2fpeak_norm_style = args
    if use_first_col:
        dz_cols = dz.columns[:-2]
        dz_vals = dz.values[:,:-2]
    else:
        dz_cols = dz.columns[1:-2]
        dz_vals = dz.values[:,1:-2]
        
    if int(gene2fpeak_norm_style)==1:
        vstd = np.std(dz_vals.astype(float), axis=0)
        print("Flag 231.020 ", vstd.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vstd)
        
    if int(gene2fpeak_norm_style)==2:
        vstd = np.std(dz_vals.astype(float), axis=0)
        print("Flag 231.022 ", vstd.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vstd)
        
        vl2norm = np.sqrt(np.sum(np.power(dz_vals.astype(float),2), axis=1))
        print("Flag 231.024 ", vl2norm.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vl2norm[:,None])
        
    if int(gene2fpeak_norm_style)==3:
        vl2norm = np.sqrt(np.sum(np.power(dz_vals.astype(float),2), axis=1))
        print("Flag 231.025 ", vl2norm.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vl2norm[:,None])
        
    import schema_qp
    sqp = schema_qp.SchemaQP(min_corr, maxwt, params= {"dist_npairs": 1000000}, mode="scale")
    try:
        dz1 = sqp.fit_transform(dz_vals, [ax], ['feature_vector'], [1])
        
        print("Flag 231.030 ", min_corr, maxwt, dz_vals.shape, ax.shape, flush=True)
    
        wtsx = np.sqrt(np.maximum(sqp._wts/np.sum(sqp._wts), 0))
    except:
        print ("ERROR: schema failed for ", min_corr, maxwt, use_first_col, strand, chromosome, gene2fpeak_norm_style)
        wtsx = 1e12*np.ones(dz_vals.shape[1])
    
    wdf = pd.Series(wtsx, index=dz_cols).sort_values(ascending=False).reset_index().rename(columns={"index": "fdist",0: "wt"})
    wdf.to_csv("{0}/adata1_sqp_wts_mincorr{1}_maxw{2}_usefirstcol{3}_strand{4}_chr{5}_norm-gene2fpeak{6}_{7}.csv".format(dir1, min_corr, maxwt, (1 if use_first_col else 0), strand, chromosome,  gene2fpeak_norm_style, outsfx), index=False)


    
def mprun_schemawts_1(adata1, dz, dir1, outsfx, n_jobs=4):
    import schema_qp
    
    pool =  multiprocessing.Pool(processes = n_jobs)
    try:
        ax = np.copy(adata1.X.todense().T)
    except:
        ax = adata1.X.T.copy()

    lx = []
    for mc in [ 0.01, 0.10, 0.20, 0.50, 0.90]:
        for mw in [100, 50, 10]:
            for use_first_col in [True, False]:
                lx.append((ax, dz, dir1, outsfx, mc, mw, use_first_col))
                
    pool.map(f_helper_mprun_schemawts_1, lx)
    

    
def mprun_schemawts_2(adata1, dz, dir1, outsfx, n_jobs=4):
    import schema_qp
    
    pool =  multiprocessing.Pool(processes = n_jobs)
    try:
        ax = np.copy(adata1.X.todense().T)
    except:
        ax = adata1.X.T.copy()

    nGenes = adata1.var.shape[0]
    chrx = adata1.var["chr"].apply(lambda s: s.replace("chr",""))
    
    lx = []
    
            
    for gene2fpeak_norm_style in  [0, 1, 2, 3]:
        dz2 = dz
        # if gene2fpeak_norm_style==True:
        #     dz2  = dz.copy(deep=True)
        #     dz2  = dz2 / (1e-12 + dz2.std())
            
        for strand in ["both", "plus", "minus"]:
            gidx_strand = np.full(nGenes, True)
            if strand == "plus":  gidx_strand = np.where(adata1.var["strand"]=="+", True, False)
            if strand == "minus": gidx_strand = np.where(adata1.var["strand"]=="-", True, False)

            for chromosome in ["all", "1--8","9--16","17--23"]: 
                if (strand!="both") and chromosome!="all": continue
                
                gidx_chr = np.full(nGenes, True)
                if chromosome=="1--8":   gidx_chr = chrx.isin("1,2,3,4,5,6,7,8".split(","))
                if chromosome=="9--16":  gidx_chr = chrx.isin("9,10,11,12,13,14,15,16".split(","))
                if chromosome=="17--23": gidx_chr = chrx.isin("17,18,19,20,21,22,X,Y".split(","))

                for mc in [ 0.01, 0.25, 0.50, 0.75, 0.875, 0.90]:
                    if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style==0) and mc not in [0.01, 0.20]: continue
                    #if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and mc not in [0.01, 0.20]: continue      
                    
                    for mw in [100, 50, 30, 20, 10]:
                        if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style==0) and mw!=100: continue
                        #if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and mw!=100: continue
                        
                        for use_first_col in  [True]: #, False]:
                            if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style==0) and use_first_col!=True: continue
                            #if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and use_first_col!=True: continue
                            
                            gidx = gidx_strand & gidx_chr
                            print("Flag 3312.040 ", gidx.shape, np.sum(gidx), ax.shape, dz2.shape, flush=True)
                            lx.append((ax[gidx,:], dz2[dz2.ensembl_id.isin(adata1.var["ensembl_id"][gidx])], dir1, outsfx, mc, mw, use_first_col, strand, chromosome, gene2fpeak_norm_style))
                            print("Flag 3312.050 ", np.sum(gidx), lx[-1][2:], flush=True)
                            
    pool.map(f_helper_mprun_schemawts_2, lx)

    
#################################################################################

if __name__ == "__main__":

    mode = sys.argv[1]
    outsfx = sys.argv[2]
    if mode=="0":
        write_h5_files()

    if mode=="1":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 24)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)

    if mode=="2":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())
        dx = mprun_geneXfpeak_mtx(adata1, 24)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_gexp-standardized_{0}.csv".format(outsfx), index=False)

    if mode=="3":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191126_1900.csv".format(dir1))
        mprun_schemawts_1(adata1, dz, dir1, outsfx, 4)

    if mode=="21":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())
        dx = mprun_geneXfpeak_mtx(adata1, 36)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_gexp-standardized_{0}.csv".format(outsfx), index=False)

    if mode=="22":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 36)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)

    if mode=="221":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 36, style=2)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)

    if mode=="222":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 36, style=3)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)


    if mode=="23":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)

        
    if mode=="231" or mode=="2311" or mode=="2312":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")

        #adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))
        if mode=="2311":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_gexp-standardized_20191202-1245.csv".format(dir1))

        if mode=="2312":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_style2_20191204-1945.csv".format(dir1))

        mprun_schemawts_2(adata1, dz, dir1,  "mode" + mode + "_" + outsfx, 5)

        
    if mode=="232" or mode=="2321" or mode=="2322":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")

        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 
        sc.pp.highly_variable_genes(adata1, n_top_genes=5000, inplace=True)

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        adata1 = adata1[:, adata1.var["highly_variable"]]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))

        if mode=="2321":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_gexp-standardized_20191202-1245.csv".format(dir1))

        if mode=="2322":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_style2_20191204-1945.csv".format(dir1))

        dz = dz[dz.ensembl_id.isin(adata1.var["ensembl_id"])].reset_index(drop=True)
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)
