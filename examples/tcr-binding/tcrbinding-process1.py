#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools
import  copy, multiprocessing, traceback, logging, pickle, traceback
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict
from tqdm import tqdm

def fast_csv_read(filename, *args, **kwargs):
    small_chunk = pd.read_csv(filename, nrows=50)
    if small_chunk.index[0] == 0:
        coltypes = dict(enumerate([a.name for a in small_chunk.dtypes.values]))
        return pd.read_csv(filename, dtype=coltypes, *args, **kwargs)
    else:
        coltypes = dict((i+1,k) for i,k in enumerate([a.name for a in small_chunk.dtypes.values]))
        coltypes[0] = str
        return pd.read_csv(filename, index_col=0, dtype=coltypes, *args, **kwargs)


def processRawData(rawdir, h5file):
    M = pd.concat([fast_csv_read(f) for f in glob.glob("{0}/vdj_v1_hs_aggregated_donor?_binarized_matrix.csv".format(rawdir))])
    truthval_cols = [c for c in M.columns if 'binder' in c] 

    surface_marker_cols = "CD3,CD19,CD45RA,CD4,CD8a,CD14,CD45RO,CD279_PD-1,IgG1,IgG2a,IgG2b,CD127,CD197_CCR7,HLA-DR".split(",")
    dx = copy.deepcopy(M.loc[:, surface_marker_cols])
    
    M.loc[:,surface_marker_cols] = (np.log2(1 + 1e6*dx.divide(dx.sum(axis=1), axis="index"))).values
    
    # get rid of B cells etc. 
    f_trim = lambda v: v < v.quantile(0.975)
    ok_Tcells = f_trim(M["CD19"]) & f_trim(M["CD4"]) & f_trim(M["CD14"])

    M = M.loc[ ok_Tcells, :]

    a = M.loc[:,truthval_cols] 
    a2= M['cell_clono_cdr3_aa'].apply(lambda v: 'TRA:' in v and 'TRB:' in v)
    bc_idx = (a2 & a.any(axis=1))
    M = M[bc_idx].reset_index(drop=True)

    mcols = ["donor","cell_clono_cdr3_aa"] + truthval_cols + surface_marker_cols
    print("Flag 67.10 ", h5file)
    M.loc[:,mcols].to_hdf(h5file, "df", mode="w")



def chunkifyCDRseqs(M, f_tra_filter, f_trb_filter, tra_start=0, trb_start=0, tra_end=100, trb_end=100):
    truthval_cols = [c for c in M.columns if 'binder' in c] 
    surface_marker_cols = "CD3,CD19,CD45RA,CD4,CD8a,CD14,CD45RO,CD279_PD-1,IgG1,IgG2a,IgG2b,CD127,CD197_CCR7,HLA-DR".split(",")

    tra_L = []; trb_L = []; binds_L = []; idxL = []
    for i in tqdm(range(M.shape[0])): #range(M.shape[0]):
        sl = M.at[i,"cell_clono_cdr3_aa"].split(";")
        a_l = [x[4:][tra_start:tra_end] for x in sl if x[:4]=="TRA:" and f_tra_filter(x[4:])]
        b_l = [x[4:][trb_start:trb_end] for x in sl if x[:4]=="TRB:" and f_trb_filter(x[4:])]
        c_np = M.loc[i,truthval_cols].astype(int).values
        A0 = ord('A')
        for a in a_l:
            a_np = np.zeros(26)
            for letter in a:
                a_np[ord(letter)-A0] += 1
                
            for b in b_l:
                b_np = np.zeros(26)
                for letter in b:
                    b_np[ord(letter)-A0] += 1
                
                tra_L.append(a_np)
                trb_L.append(b_np)
                binds_L.append(c_np)
                idxL.append(i)
    tra = np.array(tra_L)
    trb = np.array(trb_L)
    binds = np.array(binds_L)
    return tra, trb, binds, M.loc[:, surface_marker_cols].iloc[idxL,:], M.iloc[idxL,:]["donor"]





def f_dataset_helper(w_vdj, x, md, mw, w_surface_markers):
    trax, trbx, bindsx, d_surface_markers, _ = w_vdj
    try:
        return run_dataset_schema(trax, trbx, bindsx, 0.01, max_w = mw, mode=md, d_surface_markers = d_surface_markers, w_surface_markers=w_surface_markers)
    except:
        print ("Flag 67567.10 Saw exception in f_dataset_helper")
        return (None, None)



    
def run_dataset_schema(tra, trb, binds, min_corr, max_w=1000, mode="both", d_surface_markers=None, w_surface_markers=0):
    alphabet = [chr(ord('A')+i) for i in range(26)]  
    non_aa = np.array([ord(c)-ord('A') for c in "BJOUXZ"]) #list interepretation of string
    if "both" in mode:
        D = np.hstack([tra,trb])
        letters = np.array(['a'+c for c in alphabet] + ['b'+c for c in alphabet])
        to_delete =  list(non_aa)+list(non_aa+26)
    elif "tra" in mode:
        D = tra
        letters = ['{0}'.format(chr(ord('A')+i)) for i in range(26)] 
        to_delete =  list(non_aa)
    elif "trb" in mode:
        D = trb
        letters = ['{0}'.format(chr(ord('A')+i)) for i in range(26)] 
        to_delete =  list(non_aa)

    D = np.delete(D, to_delete, axis=1)
    letters = np.delete(letters, to_delete)
        
    if "bin:" in mode:
        D = 1*(D>0)

    if "std:" in mode:
        D = D / (1e-12 + D.std(axis=0))
        
    g = [binds]; wg = [1]; tg=["feature_vector"]
    
    if w_surface_markers != 0:
        g.append(d_surface_markers.values)
        wg.append(w_surface_markers)
        tg.append("feature_vector")

    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        import schema_qp
    except:
        from schema import schema_qp

    afx = schema_qp.SchemaQP(min_corr, max_w, params = {"require_nonzero_lambda":1}, mode="scale")
    afx.fit(D,g,tg,wg)
    return (pd.Series(np.sqrt(afx._wts), index=letters), afx._soln_info)




def f_tra_filter(v):
    return  v[:2]=="CA" and len(v)>=13

def f_trb_filter(v):
    return  v[:4]=="CASS" and len(v)>=13


def do_dataset_process(M, l, n_jobs, intermediate_file=None, include_full_seq=True, w_surface_markers=0, kmer_type=""):
    try:
        if include_full_seq:
            l.append((2,4,7,11))

        l1 = [(M, f_tra_filter, f_trb_filter, *v) for v in l]

        print ("Flag 681.10 ", len(l1), M.shape, l, n_jobs)

        if intermediate_file is not None and os.path.exists(intermediate_file):
            vdj_data = pickle.load(open(intermediate_file,'rb'))
        else:
            pool =  multiprocessing.Pool(processes = n_jobs)
            try:
                vdj_data = pool.starmap(chunkifyCDRseqs, l1)
            finally:
                pool.close()
                pool.join()

            if intermediate_file is not None:
                pickle.dump(vdj_data, open(intermediate_file,'wb'))

        print ("Flag 681.50 ", len(vdj_data))

        lx = []
        for md in ["tra","trb"]:
            for mw in [1.5, 1.75, 2, 2.25, 3, 5]:
                lx.extend([(w, l[i], kmer_type+":"+md, mw, w_surface_markers) for i,w in enumerate(vdj_data)])


        print ("Flag 681.70 ", len(lx))

        pool2 = multiprocessing.Pool(processes = n_jobs)
        try:
            lx2 = pool2.starmap(f_dataset_helper, lx)
        except:
            lx2
        finally:
            pool2.close()
            pool2.join()

        print ("Flag 681.80 ", len(lx2))

        f_0_1_scaler = lambda v: (v-np.min(v))/(np.max(v)-np.min(v))

        ly = []
        rd = {}
        for i, v in enumerate(lx):
            _, k, md, mw, w_surface = v
            rd[(k, md, mw, w_surface)] = lx2[i][0]

            ly.append(f_0_1_scaler(lx2[i][0]))

        print ("Flag 681.85 ", len(ly), len(rd))

        v_rd = pd.DataFrame(ly).median(axis=0).sort_values()
        return v_rd, rd
    
    except:
        return (None, None)




def f_colprocess_helper(trx, binds, surface_markers, mw, w_surface_markers):
    try:
        g = [binds]; wg = [1]; tg=["feature_vector"]
        
        if w_surface_markers != 0:
            g.append(d_surface_markers.values)
            wg.append(w_surface_markers)
            tg.append("feature_vector")

        try:
            sys.path.append(os.path.join(sys.path[0],'../../schema'))
            import schema_qp
        except:
            from schema import schema_qp

        afx = schema_qp.SchemaQP(0.01, mw, params = {"require_nonzero_lambda":1, 
                                                     "scale_mode_uses_standard_scaler":1,
                                                     "d0_type_is_feature_vector_categorical":1,}, 
                                 mode="scale")
        afx.fit(trx,g,tg,wg)
        return (pd.Series(np.sqrt(afx._wts)), afx._soln_info)

    except:
        print ("Flag 67568.10 Saw exception in f_colprocess_helper")
        return (None, None)



def do_columnwise_process(M, chain, n_jobs, intermediate_file=None, w_surface_markers=0):
    assert chain in ["tra","trb"]

    try:
        truthval_cols = [c for c in M.columns if 'binder' in c] 
        surface_marker_cols = "CD3,CD19,CD45RA,CD4,CD8a,CD14,CD45RO,CD279_PD-1,IgG1,IgG2a,IgG2b,CD127,CD197_CCR7,HLA-DR".split(",")

        def f_pad20(s):
            return [ord(c)-ord('A')+1 for c in s[:20]] + [0]*max(0,20-len(s))

        trx_L = []; binds_L = []; markers_L = []
        for i in tqdm(range(M.shape[0])): #range(M.shape[0]):
            sl = M.at[i,"cell_clono_cdr3_aa"].split(";")
            for x in sl:
                if x[:4].lower() != (chain+":"): continue
                trx_L.append(f_pad20(x[4:]))  
                binds_L.append(M.loc[i,truthval_cols].astype(int).values)
                markers_L.append(M.loc[i,surface_marker_cols].astype(int).values)
        
        trx = np.array(trx_L)
        binds = np.array(binds_L)
        surface_markers = np.array(markers_L)

        vdj_data = (trx, binds, surface_markers)

        print ("Flag 682.10 ",  M.shape, chain, n_jobs)

        if intermediate_file is not None and os.path.exists(intermediate_file):
            vdj_data = pickle.load(open(intermediate_file,'rb'))
        elif intermediate_file is not None:
                pickle.dump(vdj_data, open(intermediate_file,'wb'))
                trx, binds, surface_markers = vdj_data

        print ("Flag 681.50 ", trx.shape)

        lx = []
        for mw in [1.6, 1.8, 2, 2.2]: 
                lx.extend([(trx, binds, surface_markers, mw, w_surface_markers)])


        print ("Flag 681.70 ", len(lx))

        pool2 = multiprocessing.Pool(processes = n_jobs)
        try:
            lx2 = pool2.starmap(f_colprocess_helper, lx)
        except:
            lx2
        finally:
            pool2.close()
            pool2.join()

        print ("Flag 681.80 ", len(lx2))

        f_0_1_scaler = lambda v: (v-np.min(v))/(np.max(v)-np.min(v))

        ly = []
        rd = {}
        for i, v in enumerate(lx):
            _, _, _, mw, w_surface = v
            rd[(chain, mw, w_surface)] = lx2[i][0]

            ly.append(f_0_1_scaler(lx2[i][0]))

        print ("Flag 681.85 ", len(ly), len(rd))

        v_rd = pd.DataFrame(ly).median(axis=0).sort_values()
        v_rd2 = pd.DataFrame(ly).mean(axis=0).sort_values()
        return v_rd, v_rd2, rd
    
    except:
        #raise
        return (None, None, None)

        
#################################################################################

if __name__ == "__main__":
    sys.path.append(os.path.join(sys.path[0],'../../schema'))
    from utils import SlideSeq

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details")
    parser.add_argument("--outdir", help="output directory (can set to '.')", type=str, default=".")
    parser.add_argument("--outsfx", help="suffix to use when producing output files")
    parser.add_argument("--style", help="mode-specific interpretation", type=int, default=0)
    parser.add_argument("--infile", help="path to the .h5 file containing processed 10X binarized-csv dataframe", default="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/10x/processed/vdj_binarized_alldonors.h5")
    parser.add_argument("--njobs", help="number of parallel cores to use", type=int, default=24)

    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    
    args = parser.parse_args()
    assert args.mode is not None
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}

    
    if args.mode== "raw_data_read":
        processRawData( os.path.dirname(args.infile).replace("/processed",""), args.infile)



        
    if args.mode == "compute_2_modality_selection_pressure" or args.mode== "compute_3_modality_selection_pressure":
        K = 2 if args.mode=="compute_2_modality_selection_pressure" else 3
        
        intmdt_file = "{0}/schema_median_wts_{3}-dataset_style{1}_intermediate_{2}.pkl".format(args.outdir, args.style, args.outsfx, K)
        
        M = pd.read_hdf(args.infile)

        l = []
        for i in range(5):
            l += [(2+i, 4+i, min(7,2+1+i), min(11,4+1+i))]
        l += [(2,9,7,10)]
        l += [(2,10,7,11)]
            
        w_surface_markers = 0
        if args.mode=="compute_3_modality_selection_pressure":
            w_surface_markers = float(extra_args.get("w_surface_markers",-0.1))

        # regular run, not for cross-validation
        if int(extra_args.get("separate_by_donor",0))==0 and int(extra_args.get("kfold_split",1))<=1:
            print ("Flag 679.10 ", M.shape, len(l), l)
            v_rd, rd = do_dataset_process(M, l, args.njobs, 
                                          intermediate_file = intmdt_file, 
                                          include_full_seq= int(extra_args.get("include_full_seq",0))==1,
                                          w_surface_markers = w_surface_markers,
                                          kmer_type = extra_args.get("kmer_type",""))


            csvfile = "{0}/schema_median_wts_{3}-dataset_style{1}_{2}.csv".format(args.outdir, args.style, args.outsfx, K)
            pklfile = "{0}/schema_median_wts_{3}-dataset_style{1}_schema-results_{2}.pkl".format(args.outdir, args.style, args.outsfx, K)

            try:            
                v_rd.to_csv(csvfile, index=True, header=False)
                pickle.dump(rd, open(pklfile, "wb"))
            except:
                os.system("echo Error > {0}".format(csvfile))
                traceback.print_exc(open(csvfile,'at'))
                os.system("echo Error > {0}".format(pklfile))
                
        # cross-validation run, split by donor
        elif int(extra_args.get("separate_by_donor",0))>0 :
            dnrL = list(M["donor"].unique())
            for dnr in ["ALL"] + dnrL:
                Mx = M if dnr=="ALL" else M[M["donor"]==dnr].copy().reset_index(drop=True)
                print ("Flag 679.20 ", Mx.shape, len(l), l)
                v_rd, rd = do_dataset_process(Mx, l, args.njobs, 
                                              intermediate_file = None,
                                              include_full_seq= int(extra_args.get("include_full_seq",0))==1,
                                              w_surface_markers = w_surface_markers,
                                              kmer_type = extra_args.get("kmer_type",""))


                csvfile = "{0}/schema_median_wts_{4}-dataset_style{1}_donor-{3}_{2}.csv".format(args.outdir, args.style, args.outsfx, dnr, K)
                pklfile = "{0}/schema_median_wts_{4}-dataset_style{1}_donor-{3}_schema-results_{2}.pkl".format(args.outdir, args.style, args.outsfx, dnr, K)
                try:
                    v_rd.to_csv(csvfile, index=True, header=False)
                    pickle.dump(rd, open(pklfile, "wb"))
                except:
                    os.system("echo Error > {0}".format(csvfile))
                    traceback.print_exc(open(csvfile,'at'))
                    os.system("echo Error > {0}".format(pklfile))

                    
        # cross-validation run, split by epitopes into k subsets
        elif int(extra_args.get("kfold_split",1)) >1:
            n = M.shape[0]

            # split by binding specifities
            truthval_cols = [c for c in M.columns if 'binder' in c]
            Mbinds = M.loc[:,truthval_cols].values
            b1 = np.argmax(Mbinds, axis=1)
            vL = np.array_split(list(set(b1)), int(extra_args["kfold_split"]))
            print ("Flag 681.10 ", vL)
            idxL = [ np.ravel(np.nonzero(np.isin(b1, v))) for v in vL]

            # split randomly
            #X = np.arange(n).reshape(n,1)
            #kfold = sklearn.model_selection.KFold(n_splits = int(extra_args["kfold_split"]), random_state=0)
            #idxL = [idx for idx,_ in kfold.split(X)]
            
            for i, idx in enumerate(idxL + ["ALL"]): #(["ALL"] + idxL):
                if idx=="ALL":
                    Mx = M
                    nm = "ALL"
                else:
                    Mx = M.iloc[idx,:].copy().reset_index(drop=True)
                    #nm = "C{0}".format(i)
                    nm = "B{0}".format(i)

                print ("Flag 681.20 ", Mx.shape, len(l), l)
                v_rd, rd = do_dataset_process(Mx, l, args.njobs, 
                                              intermediate_file = None,
                                              include_full_seq= int(extra_args.get("include_full_seq",0))==1,
                                              w_surface_markers = w_surface_markers,
                                              kmer_type = extra_args.get("kmer_type",""))


                csvfile = "{0}/schema_median_wts_{4}-dataset_style{1}_kfold-{3}_{2}.csv".format(args.outdir, args.style, args.outsfx, nm, K)
                pklfile = "{0}/schema_median_wts_{4}-dataset_style{1}_kfold-{3}_schema-results_{2}.pkl".format(args.outdir, args.style, args.outsfx, nm, K)
                try:
                    v_rd.to_csv(csvfile, index=True, header=False)
                    pickle.dump(rd, open(pklfile, "wb"))
                except:
                    os.system("echo Error > {0}".format(csvfile))
                    traceback.print_exc(open(csvfile,'at'))
                    os.system("echo Error > {0}".format(pklfile))
    


                    
    if args.mode == "compute_2_modality_columnwise_preference" or args.mode == "compute_3_modality_columnwise_preference":
        K = 3 if args.mode=="compute_3_modality_columnwise_preference" else 2
        
        intmdt_file = "{0}/schema_columnwise_median_wts_{3}-dataset_style{1}_intermediate_{2}.pkl".format(args.outdir, args.style, args.outsfx, K)
        
        M = pd.read_hdf(args.infile)
        #M = M.iloc[:5000,:] #for testing

        w_surface_markers = 0
        if args.mode=="compute_3_modality_selection_pressure":
            w_surface_markers = float(extra_args.get("w_surface_markers",-0.1))

        chain = extra_args.get("chain","tra")
                
        print ("Flag 683.20 ", M.shape, chain)
        v_rd, v_rd2, rd = do_columnwise_process(M, chain, args.njobs, 
                                                intermediate_file = None,
                                                w_surface_markers = w_surface_markers)


        csvfile = "{0}/schema_columnwise_median_wts_{3}-dataset_style{1}_chain-{4}_{2}.csv".format(args.outdir, args.style, args.outsfx, K, chain)
        csvfile2 = "{0}/schema_columnwise_mean_wts_{3}-dataset_style{1}_chain-{4}_{2}.csv".format(args.outdir, args.style, args.outsfx, K, chain)
        pklfile = "{0}/schema_columnwise_median_wts_{3}-dataset_style{1}_schema-results_chain-{4}_{2}.pkl".format(args.outdir, args.style, args.outsfx, K, chain)
        try:
            v_rd.to_csv(csvfile, index=True, header=False)
            v_rd2.to_csv(csvfile2, index=True, header=False)
            pickle.dump(rd, open(pklfile, "wb"))
        except:
            os.system("echo Error > {0}".format(csvfile))
            traceback.print_exc(open(csvfile,'at'))
            os.system("echo Error > {0}".format(csvfile2))
            os.system("echo Error > {0}".format(pklfile))


