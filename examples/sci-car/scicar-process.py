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


    # step_size = 1000
    # nGenes = adata1.shape[1]

    # w = None
    # k = gene_cnt
    # i = 0
    # while step_size > 5:
    #     w = (sc.pp.highly_variable_genes(adata1, n_top_genes=nGenes - k, inplace=False)).highly_variable

    #     nw = np.sum(~w)
    #     if i > 20 or abs(nw - gene_cnt) < 10: break

    #     if nw > gene_cnt: 
    #         k -= step_size
    #         i += 1
    #         step_size = int(step_size / 2.0)
        
    #     else:
    #         k += step_size
    #         i += 1
    #         step_size = int(step_size / 2.0)

    #     print ("Flag 7867.100 ",  np.sum(~w), gene_cnt, nGenes, step_size, k, i)
    
    # assert w is not None
    # return ~w



def random_gene_list(nGenes, gene_cnt, num_random_samples=1000):
    L  = []
    for i in range(num_random_samples):
        b = np.zeros(nGenes, dtype=bool)
        b[np.random.choice(nGenes, gene_cnt, replace=False)] = True
        L.append(b)

    return L


def f_mode4_helper(gdist_matrix,  gene_set, mode):
    try:
        gdist2 = gdist_matrix[gene_set, :][:, gene_set]
        print ("Flag 34.20 ", gdist2.shape, np.sum(gene_set), len(gene_set))
        if mode in ["4","5","51","52"]:
            gw = np.ravel(np.amin(gdist2, axis=0))
            wx = gw[ gw < 1e12]
            print ("Flag 34.40 ", gw.shape, len(wx), np.sum(wx), np.mean(wx))
            assert len(wx) > 0
            return np.mean(wx)
    except Exception as e:
        print ("Flag 34.50 saw exception for ", gdist_matrix.shape, len(gene_set), np.sum(gene_set), e)
        traceback.print_exc(file=sys.stdout)
        return np.NaN
    


def f_mode6_helper(gdist_matrix, gsign_matrix,  gene_set, mode):
    try:
        gdist2 = gdist_matrix[gene_set, :][:, gene_set]
        gsign2= gsign_matrix[gene_set, :][:, gene_set]
        print ("Flag 35.20 ", gdist2.shape, gsign2.shape, np.sum(gene_set), len(gene_set))

        if mode in ["6","61","62"]:
            gw = np.ravel(np.amin(gdist2, axis=0))
            gw_idx = np.ravel(np.argmin(gdist2, axis=0))
            gs = np.ravel(gsign2[ np.arange(gdist2.shape[0]), gw_idx])
            wx = gs[ gw < 1e12]
            print ("Flag 35.40 ", gw.shape, gs.shape, len(wx), np.sum(wx), np.mean(wx))
            assert len(wx) > 0
            return np.mean(wx)
    except Exception as e:
        print ("Flag 35.50 saw exception for ", gdist_matrix.shape, gsign_matrix.shape, len(gene_set), np.sum(gene_set), e)
        traceback.print_exc(file=sys.stdout)
        return np.NaN
    

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

def f_mode7_helper(g2tad, gene_sharedtad,  gene_set, mode):
    if mode in "7,71,72".split(","):
        w= f_in_tad_cnt(g2tad, gene_set)/float(np.sum(gene_set))
        print ("Flag 36.0 ", w)
        return w

    if mode in "8,81,82".split(","):
        v  = g2tad[gene_set]
        v = v[~np.isnan(v)]
        n_tads = len(np.unique(v))
        print ("Flag 36.1 ", n_tads)
        return n_tads


def f_mode81x_helper(g2tad, gene_sharedtad,  gene_set, mode):
    if mode == "81x":
        tad2genes = defaultdict(list)
        for i, t in enumerate(g2tad):
            if not np.isnan(t) and gene_set[i]:
                tad2genes[t].append(i)
        return [(k,v,len(v)) for k,v in tad2genes.items()]
                                     
    
#################################################################################

if __name__ == "__main__":

    sys.path.append(os.path.join(sys.path[0],'../../schema'))
    print (sys.path)
    from utils import SciCar

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



    if mode=="33":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        gexp_cnt = np.ravel(np.sum(adata1.X.todense() > 0.1, axis=0))
        valid_genes = gexp_cnt > 5
        adata1 = adata1[:, valid_genes]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191210-2300.csv".format(dir1))
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)

        
    if mode=="331":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        gexp_cnt = np.ravel(np.sum(adata1.X.todense() > 0.1, axis=0))
        valid_genes = gexp_cnt > 5
        adata1 = adata1[:, valid_genes]

        #adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191210-2300.csv".format(dir1))

        mprun_schemawts_2(adata1, dz, dir1,  "mode" + mode + "_" + outsfx, 5)

        
    if mode=="332":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        gexp_cnt = np.ravel(np.sum(adata1.X.todense() > 0.1, axis=0))
        valid_genes = gexp_cnt > 5
        adata1 = adata1[:, valid_genes]

        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 
        sc.pp.highly_variable_genes(adata1, n_top_genes=4000, inplace=True)

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        adata1 = adata1[:, adata1.var["highly_variable"]]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191210-2300.csv".format(dir1))

        dz = dz[dz.ensembl_id.isin(adata1.var["ensembl_id"])].reset_index(drop=True)
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)

        
    if mode=="4":
        import scanpy as sc

        gene_window = [int(w) for w in sys.argv[2].split("-")]
        outsfx = sys.argv[3]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"

        print ("Flag 54123.10 ")
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        mu_gexp = np.ravel(np.mean(adata1.X, axis=0))
        valid_genes = mu_gexp > 1e-5
        adata1 = adata1[:, valid_genes]

        print ("Flag 54123.10 ", adata1.shape, adata1.X.shape, adata1.var.shape)
        
        nGenes = adata1.X.shape[1]

        gdist_matrix = getGeneDistances(adata1)
        print ("Flag 54123.20 ", gdist_matrix.shape) #, describe(gdist_matrix))
        non_variable_genes = findNonVariableGenes(adata1,  gene_window)
        print ("Flag 54123.30 ", gene_window, len(non_variable_genes), np.sum(non_variable_genes))

        n_nonvar_genes = np.sum(non_variable_genes)
    
        lx0 = [non_variable_genes] + random_gene_list(nGenes, n_nonvar_genes, num_random_samples=500)
        lx =  [(gdist_matrix, a, mode) for a in lx0]

        print ("Flag 54123.40 ", len(lx), len(lx[1][1]), np.sum(lx[1][1]))
        
        n_jobs = 36
        print ("Flag 54123.435 ", nGenes, n_nonvar_genes, gdist_matrix.shape, adata1.shape, len(lx))

        pool =  multiprocessing.Pool(processes = n_jobs)

        ly = pool.starmap(f_mode4_helper, lx)
        
        print ("Flag 54123.500 ",  ly[:30])

        d_actual = ly[0]
        d_bl = np.array(ly[1:])
        mu, sd, median = np.nanmean(d_bl), np.std(d_bl), np.nanmedian(d_bl)
        d_actual_rank = float(np.sum( d_actual >  d_bl))/len(d_bl)

        outfile = "{0}/adata1_variablegene_nearestdist_mode{2}_genewindow{3}_{1}".format(dir1, outsfx, mode, sys.argv[2])
        outw = [d_actual_rank, d_actual, mu, sd, median, len(d_bl), gene_window[0], gene_window[1], n_nonvar_genes] + list(d_bl)
        np.savetxt(outfile, np.array(outw))
        #print("RESULT ", d_actual, d_actual_rank, d_bl, mu, sd, median, len(d_bl), gene_cnt, n_nonvar_genes)



    if mode in "5,51,52,6,61,62,7,71,72,8,81,82".split(","):
        import scanpy as sc

        gene_window = [int(w) for w in sys.argv[2].split("-")]
        outsfx = sys.argv[3]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"

        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = adata1.X.todense()
        gexp_cnt = np.ravel(np.sum(adata1.X > 0.1, axis=0))
        #mu_gexp = np.ravel(np.mean(adata1.X, axis=0))
        #valid_genes = mu_gexp > 1e-5
        valid_genes = gexp_cnt > 5
        adata1 = adata1[:, valid_genes]

        nGenes = adata1.shape[1]
        mu_gexp = np.ravel(np.mean(adata1.X, axis=0))
        adata1.var["gene_expression_ranking"] = scipy.stats.rankdata(-mu_gexp) #highest=1

        nGenes = adata1.shape[1]
        sd_gexp = np.ravel(np.std(adata1.X, axis=0))
        adata1.var["gene_expsd_ranking"] = scipy.stats.rankdata(sd_gexp) #lowest=1

        gdist_matrix = getGeneDistances(adata1)
        print ("Flag 54123.20 ", gdist_matrix.shape) #, describe(gdist_matrix))

        if mode in "6,61,62".split(","):
            gsign_matrix = getGeneStrandMatch(adata1)
            print ("Flag 54123.21 ", gsign_matrix.shape) #, describe(gdist_matrix))

        if mode in "7,71,72,8,81,82".split(","):
            tad = pd.read_csv("/afs/csail.mit.edu/u/r/rsingh/work/refdata/hg19_A549_TAD.bed", delimiter="\t", header=None)
            tad.columns = ["chr","start","end", "x1","x2"]
            g2tad, gene_sharedtad = getGene2TADmap(adata1, tad)


        nGenes = adata1.shape[1]
        v = np.full(nGenes, nGenes)
        for i in tqdm(range(500,nGenes-500,500)):
            v0 = sc.pp.highly_variable_genes(adata1, n_top_genes=i,inplace=False).highly_variable
            v = np.where((v==nGenes) & v0, i, v).copy()
        adata1.var["gene_variability_ranking"] = v
            
        window_genes = ((adata1.var["gene_expression_ranking"].values >= gene_window[0]) &  
                                    (adata1.var["gene_expression_ranking"].values <   gene_window[1])) 

        if mode in "51,61,71,81".split(","):
            window_genes = ((adata1.var["gene_variability_ranking"].values >= gene_window[0]) &  
                            (adata1.var["gene_variability_ranking"].values <   gene_window[1])) 


        if mode in "52,62,72,82".split(","):
            window_genes = ((adata1.var["gene_expsd_ranking"].values >= gene_window[0]) &  
                            (adata1.var["gene_expsd_ranking"].values <   gene_window[1])) 

        n_window_genes = np.sum(window_genes)

        print ("Flag 54123.30 ", gene_window, len(window_genes), np.sum(window_genes))

        num_samples = 1000

        lx0 = [window_genes] + random_gene_list(nGenes, n_window_genes, num_random_samples=num_samples)
        lx =  [(gdist_matrix, a, mode) for a in lx0]

        if mode in "8,81,82".split(","):
            in_tad_cnt = f_in_tad_cnt(g2tad, window_genes)
            tad_total_cnt = np.sum(~np.isnan(g2tad))

            rl = random_gene_list(tad_total_cnt, in_tad_cnt, num_random_samples = num_samples)

            window_genes [np.isnan(g2tad)] = False
            n_window_genes = np.sum(window_genes)
            lx0 = [window_genes]
            print ("Flag 54123.32 ",  len(window_genes), np.sum(window_genes), len(lx0))
            for rlx in rl:
                ax = np.full( nGenes, False)
                ax[~np.isnan(g2tad)] = rlx #[rlx] = True
                lx0.append(ax)
            print ("Flag 54123.33 ",  len(window_genes), np.sum(window_genes), len(lx0), len(lx0[1]), np.sum(lx0[1]))
            
        if mode in ["6","61","62"]:
            lx =  [(gdist_matrix, gsign_matrix, a, mode) for a in lx0]

        if mode in "7,71,72,8,81,82".split(","):
            lx =  [(g2tad, gene_sharedtad, a, mode) for a in lx0]


        print ("Flag 54123.40 ", len(lx), len(lx[1][2]), np.sum(lx[1][2]))
        
        n_jobs = 36
        print ("Flag 54123.435 ", nGenes, n_window_genes, gdist_matrix.shape, adata1.shape, len(lx))

        pool =  multiprocessing.Pool(processes = n_jobs)

        if mode in "4,5,51,52".split(","):
            ly = pool.starmap(f_mode4_helper, lx)

        elif mode in "6,61,62".split(","):
            ly = pool.starmap(f_mode6_helper, lx)

        elif mode in "7,71,72,8,81,82".split(","):
            ly = pool.starmap(f_mode7_helper, lx)

        print ("Flag 54123.500 ",  ly[:30])

        d_actual = ly[0]
        d_bl = np.array(ly[1:])
        mu, sd, median = np.nanmean(d_bl), np.std(d_bl), np.nanmedian(d_bl)
        d_actual_rank = float(np.sum( d_actual >  d_bl))/len(d_bl)

        outfile = "{0}/adata1_gexp_nearestdist_mode{2}_genewindow{3}_{1}".format(dir1, outsfx, mode, sys.argv[2])
        outw = [d_actual_rank, d_actual, mu, sd, median, len(d_bl), gene_window[0], gene_window[1], n_window_genes] + list(d_bl)
        np.savetxt(outfile, np.array(outw))


    if mode in "81x".split(","):
        import scanpy as sc

        gene_window = [int(w) for w in sys.argv[2].split("-")]
        outsfx = sys.argv[3]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"

        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = adata1.X.todense()
        gexp_cnt = np.ravel(np.sum(adata1.X > 0.1, axis=0))
        valid_genes = gexp_cnt > 5
        adata1 = adata1[:, valid_genes]

        gdist_matrix = getGeneDistances(adata1)
        print ("Flag 54123.20 ", gdist_matrix.shape) #, describe(gdist_matrix))

        gsign_matrix = getGeneStrandMatch(adata1)
        print ("Flag 54123.21 ", gsign_matrix.shape) #, describe(gdist_matrix))

        tad = pd.read_csv("/afs/csail.mit.edu/u/r/rsingh/work/refdata/hg19_A549_TAD.bed", delimiter="\t", header=None)
        tad.columns = ["chr","start","end", "x1","x2"]
        g2tad, gene_sharedtad = getGene2TADmap(adata1, tad)

        nGenes = adata1.shape[1]
        v = np.full(nGenes, nGenes)
        for i in tqdm(range(500,nGenes-500,500)):
            v0 = sc.pp.highly_variable_genes(adata1, n_top_genes=i,inplace=False).highly_variable
            v = np.where((v==nGenes) & v0, i, v).copy()
        adata1.var["gene_variability_ranking"] = v
            
        window_genes = ((adata1.var["gene_variability_ranking"].values >= gene_window[0]) &  
                        (adata1.var["gene_variability_ranking"].values <   gene_window[1])) 

        n_window_genes = np.sum(window_genes)

        print ("Flag 54123.30 ", gene_window, len(window_genes), np.sum(window_genes))

        num_samples = 100 #1000

        lx0 = [window_genes] + random_gene_list(nGenes, n_window_genes, num_random_samples=num_samples)
        lx =  [(gdist_matrix, a, mode) for a in lx0]

        in_tad_cnt = f_in_tad_cnt(g2tad, window_genes)
        tad_total_cnt = np.sum(~np.isnan(g2tad))

        rl = random_gene_list(tad_total_cnt, in_tad_cnt, num_random_samples = num_samples)

        window_genes [np.isnan(g2tad)] = False
        n_window_genes = np.sum(window_genes)
        lx0 = [window_genes]
        print ("Flag 54123.32 ",  len(window_genes), np.sum(window_genes), len(lx0))
        for rlx in rl:
            ax = np.full( nGenes, False)
            ax[~np.isnan(g2tad)] = rlx #[rlx] = True
            lx0.append(ax)
        print ("Flag 54123.33 ",  len(window_genes), np.sum(window_genes), len(lx0), len(lx0[1]), np.sum(lx0[1]))
            
        lx =  [(g2tad, gene_sharedtad, a, mode) for a in lx0]


        print ("Flag 54123.40 ", len(lx), len(lx[1][2]), np.sum(lx[1][2]))
        
        n_jobs = 36
        print ("Flag 54123.435 ", nGenes, n_window_genes, gdist_matrix.shape, adata1.shape, len(lx))


        pool =  multiprocessing.Pool(processes = n_jobs)
        ly = pool.starmap(f_mode81x_helper, lx)

        outfile = "{0}/adata1_tad_membership_mode{2}_genewindow{3}_{1}".format(dir1, outsfx, mode, sys.argv[2])
        outfh = open(outfile, 'w')
        
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

        outfh.close()


