#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools
import  copy, multiprocessing, traceback, logging, pickle
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
    import utils
    for pid in ['180430_1','180430_5','180430_6']:
        adata1 = utils.SlideSeq.loadRawData("/afs/csail.mit.edu/u/r/rsingh/work/afid/data/slideseq/raw/", pid, 100)
        adata1.write("/afs/csail.mit.edu/u/r/rsingh/work/afid/data/slideseq/processed/puck_{0}.h5ad".format(pid))
                                     

def computeKernelDensityGranuleCells(adata1, kd_fit_granule_only=True, kd_bw=125):
    from sklearn.neighbors import KernelDensity
    fscl = lambda v: 20*(sklearn.preprocessing.MinMaxScaler().fit_transform(np.exp(v[:,None]-v.min()))).ravel()
    d3 = adata1.obs.copy(deep=True) #adata1.uns["Ho"].merge(adata1.obs, how="inner", left_index=True, right_index=True)
    d3c = d3[["xcoord","ycoord"]]
    if kd_fit_granule_only:
        d3["kd"] = fscl(KernelDensity(kernel='gaussian', bandwidth=kd_bw).fit(d3c[d3["atlas_cluster"]==1].values).score_samples(d3c.values))
    else:
        d3["kd"] = fscl(KernelDensity(kernel='gaussian', bandwidth=kd_bw).fit(d3c.values).score_samples(d3c.values))
    adata1.obs["kd"] = d3["kd"]
    return adata1



def checkMaxFeasibleCorr(D, d0, g, tg, wg):
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        import schema_qp
    except:
        from schema import schema_qp
    
    for thresh in [0.30, 0.275, 0.25, 0.225, 0.20, 0.15, 0.10, 0.075, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01]:
        print ("STARTING TRY OF ", thresh)
        try:
            sqp = schema_qp.SchemaQP(thresh, w_max_to_avg=1000, params= {"dist_npairs": 1000000}, mode="scale")
            dz1 = sqp.fit(D, g, tg, wg, d0=d0)
            print ("SUCCEEDED TRY OF ", thresh)
            return 0.9*thresh, thresh
        except:
            print ("EXCEPTION WHEN TRYING ", thresh)
            #raise
    return 0,0


def runSchemaGranuleCellDensity(D, d0, gIn, tgIn, wgIn, min_corr1, min_corr2):
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        import schema_qp
    except:
        from schema import schema_qp
    
    f_linear = lambda v:v
    ret_val = {}

    w_list= [1,10,50,100]
    for w in w_list:
        s="linear"
        f=f_linear
        
        g1, wg1, tg1  = gIn[:], wgIn[:], tgIn[:]  # does maximize negative corr with non-granule 
        wg1[0] = w
        
        g = [g1[0]]; wg = [wg1[0]]; tg=[tg1[0]]  # does NOT maximize negative corr with non-granule
            
        #afx0 = schema_qp.SchemaQP(0.001, 1000, mode="scale")
        #Dx0 = afx0.fit_transform(D,g,tg,wg,d0)
        #ret_val[(s,w,0)] = (np.sqrt(afx0._wts), afx0._soln_info)

        try:
            afx1 = schema_qp.SchemaQP(min_corr1, w_max_to_avg=1000, mode="scale")
            Dx1 = afx1.fit_transform(D,g,tg,wg,d0=d0)
            ret_val[(s,w,1)] = (np.sqrt(afx1._wts), afx1._soln_info )  # does NOT maximize negative corr with non-granule
        except:
            print("TRYING min-corr {0} for afx1 broke here".format(min_corr1))
            continue

        try:
            afx2 = schema_qp.SchemaQP(min_corr1, w_max_to_avg=1000, mode="scale")
            Dx2 = afx2.fit_transform(D,g1,tg1,wg1,d0=d0)
            ret_val[(s,w,2)] = (np.sqrt(afx2._wts), afx2._soln_info)  # does maximize negative corr with non-granule
        except:
            print("TRYING min-corr {0} for afx2 broke here".format(min_corr1))
            continue

        try:
            afx3 = schema_qp.SchemaQP(min_corr2, w_max_to_avg=1000, mode="scale")
            Dx3 = afx3.fit_transform(D,g1,tg1,wg1,d0=d0)  # does maximize negative corr with non-granule
            ret_val[(s,w,3)] = (np.sqrt(afx3._wts), afx3._soln_info)
        except:
            print("TRYING min-corr {0} for afx3 broke here".format(min_corr2))
            continue 

    return ret_val



def getSoftmaxCombinedScores(Wo, schema_ret, use_generanks=True, do_pow2=True, schema_allowed_w1s=None): 
    if use_generanks:
        R = Wo.rank(axis=1, pct=True).values.T
    else:
        R = Wo.values.T
        
    sumr = None; nr=0
    for x in schema_ret:
        style,w,i = x

        if style!="linear" or i not in [2,3]: continue   #2,3 correspond to schema runs that also require disagreement w other altas clusters

        if schema_allowed_w1s is not None and w not in schema_allowed_w1s: continue
        
        wx = schema_ret[x][0]**2
        #wx = wx/np.sum(wx)
        
        schema_wts = wx**(2 if do_pow2 else 1)
        if np.max(schema_wts) > 20:
            schema_wts = 20*schema_wts/np.max(schema_wts)
        schema_probs = np.exp(schema_wts)/np.sum(np.exp(schema_wts))
        g1 = (R*schema_probs).sum(axis=1)
        g2 = g1/np.std(g1.ravel())
        #g2 = scipy.stats.rankdata(g1); g2 = g2/np.max(g2)
        if sumr is None:
            sumr = g2
        else:
            sumr += g2
        nr += 1
    rnks = sumr/nr
    s1= pd.Series(rnks, index=list(Wo.columns)).sort_values(ascending=False)
    return {u:(i+1) for i,u in enumerate(list(s1.index))}



def getCellLoadingSoftmax(d3, Wo, schema_ret):
    R = Wo.values.T
    
    sumr = None; nr=0
    for x in schema_ret:
        style,w,i = x
        if style!="linear" or i not in [2,3]: continue               
        
        wx = schema_ret[x][0]**2
        #wx = wx/np.sum(wx)
        
        schema_wts = wx
        if np.max(schema_wts) > 20:
            schema_wts = 20*schema_wts/np.max(schema_wts)

        schema_probs = np.exp(schema_wts)/np.sum(np.exp(schema_wts))

        if sumr is None:
            sumr = schema_probs
        else:
            sumr += schema_probs
        nr += 1
        
    r = sumr/nr
    v = (d3.iloc[:,:100].multiply(r,axis=1)).sum(axis=1)
    return v.values




def generatePlotGranuleCellDensity(d3, cell_loadings):
    import matplotlib.pyplot as plt
    import seaborn as sns
    score1 = cell_loadings

    np.random.seed(239)
    plt.style.use('seaborn-paper')
    plt.rcParams['lines.markersize'] = np.sqrt(0.25)
    fc = lambda v: np.where(v,'lightslategray','red')

    #fig = plt.figure(constrained_layout=True, figsize=(6.48,2.16), dpi=300) #(2*6.48,2*2.16))
    fig = plt.figure(figsize=(6.48,2.16), dpi=300) #(2*6.48,2*2.16))
    gs = fig.add_gridspec(2,6,wspace=0,hspace=0) #(2, 2)  

    idxY=d3["atlas_cluster"]==1
    idxN=d3["atlas_cluster"]!=1

    coords = d3.loc[:,["xcoord","ycoord"]].values
    clstr = d3["atlas_cluster"].values

    cid_list = [1,2,3,6]

    fc = lambda v: np.where(v,'lightslategray','red')
    axdict = {}

    xyL = [(gs[0,0], coords[clstr==1,:], 'a','Granule Cells'), (gs[0,1], coords[clstr==2,:],'b','Purkinje Cells'), 
           (gs[1,0], coords[clstr==3,:], 'c','Interneuron'), (gs[1,1], coords[clstr==6,:], 'd','Oligodendrocytes')]

    for g, dx, titlestr, desc in xyL:

        ax = fig.add_subplot(g)
        fc = lambda v: np.where(v,'lightslategray','red')
        ax.text(.95, .05, titlestr, horizontalalignment='center', transform=ax.transAxes, size=14 )
        ax.axis('off')
        ax.scatter(dx[:,0], dx[:,1], color='black', alpha=0.20 if titlestr=='a' else 0.6 ) 
        ax.set_aspect('equal')
        axdict[titlestr] = ax


    ax = fig.add_subplot(gs[:,2:4])
    im = ax.scatter(coords[clstr==1,0],coords[clstr==1,1],c=2*d3["kd"].values[clstr==1],cmap="seismic",s=1)
    #im = ax.scatter(coords[:,0],coords[:,1],c=2*d3["kd"].values,cmap="seismic",s=1)
    ax.set_aspect('equal')
    ax.axis('off')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("bottom", size="3%", pad=0.01)
    cbar = fig.colorbar(im, cax=cax, shrink=0.2, orientation='horizontal')
    ax.text(.9, .05, "e", horizontalalignment='center', transform=ax.transAxes, size=14 )

    sx = score1 > np.quantile(score1,0.75)
    for g, titlestr, ii, c1 in [(gs[0,4], "f", idxY & sx, 'r'), (gs[0,5], "g", idxY & (~sx), 'b'), 
                                (gs[1,4], "h", idxN & sx, 'r'), (gs[1,5], "i", idxN & (~sx), 'b')]:

        ax = fig.add_subplot(g)
        ax.text(.95, .05, titlestr, horizontalalignment='center', transform=ax.transAxes, size=14 )
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)
        ax.axis('off')
        ax.scatter(coords[ii,0], coords[ii,1], color=c1, alpha=0.40 ) 
        ax.set_aspect('equal')
        axdict[titlestr] = ax


    ####################################
    fig.tight_layout()
    return fig




def processGranuleCellDensitySchema(adata1, extra_args):
    
    if "kd" not in adata1.obs.columns:
        adata1 = computeKernelDensityGranuleCells(adata1, 
                                                  kd_fit_granule_only = int(extra_args.get("kd_fit_granule_only",1))==1,
                                                  kd_bw = float(extra_args.get("kd_bw",125)))

    d3 = adata1.uns["Ho"].merge(adata1.obs, how="inner", left_index=True, right_index=True)
    Wo = adata1.uns["Wo"]        
    cols_Ho = list(adata1.uns["Ho"].columns)

    D = d3[cols_Ho].values
    d0 = 1*(d3["atlas_cluster"].values==1)
    g = [(d3["kd"].values)]; wg=[10]; tg=["numeric"]
    for clid in [2,3,6,7]:
        g.append(1*(d3["atlas_cluster"].values==clid))
        wg.append(-1)
        tg.append("categorical")


    min_corr1, min_corr2 = checkMaxFeasibleCorr(D, d0, g, tg, wg)
    schema_ret = runSchemaGranuleCellDensity(D, d0, g, tg, wg, min_corr1, min_corr2)
    scores = getSoftmaxCombinedScores(Wo, schema_ret, use_generanks=False, do_pow2=False)
    cell_loadings = getCellLoadingSoftmax(d3, Wo, schema_ret)

    fig = generatePlotGranuleCellDensity(d3, cell_loadings)
    return (fig, d3, schema_ret, min_corr1, min_corr2, scores, cell_loadings)



def doSchemaCCA_CellScorePlot2(d3, cca_x_scores, cell_loadings):
    clstrs = d3["atlas_cluster"]
    kd = d3["kd"]

    cca_sgn = np.sign(scipy.stats.pearsonr(d3["kd"],cca_x_scores)[0]) #flip signs if needed

    R = {}
    for desc,v in [("ccax", cca_sgn*cca_x_scores), ("schema", cell_loadings)]:
        vr = scipy.stats.rankdata(v)
        vr = vr/vr.max()
        l = []
        for t in np.linspace(0,1,100)[:-1]:
            cx = clstrs[vr >=t ]
            granule_frac = (np.sum(cx==1)/(1e-12+ len(cx)))
            cx2 = kd[ vr >= t]
            kd_val = np.median(cx2)
            l.append((granule_frac, kd_val))
        R[desc]= list(zip(*l))

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-paper')
    plt.rcParams['lines.markersize'] = np.sqrt(0.25)

    fig = plt.figure(dpi=300) #(2*6.48,2*2.16))

    a = np.linspace(0,1,100)
    plt.scatter(R["ccax"][0], R["ccax"][1], s=(1+3*a)**2, c="red", figure=fig)
    plt.scatter(R["schema"][0], R["schema"][1], s=(1+3*a)**2, c="blue", figure=fig)
    fig.legend("CCA fit,Schema fit".split(","))
    plt.xlabel("Fraction of Beads labeled as Granule Cells", figure=fig)
    plt.ylabel("Median Kernel Density Score", figure=fig)
    return fig



        
#################################################################################

if __name__ == "__main__":
    try:
        sys.path.append(os.path.join(sys.path[0],'../../schema'))
        from utils import SlideSeq
    except:
        from schema.utils import SlideSeq

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details")
    parser.add_argument("--outdir", help="output directory (can set to '.')", type=str, default=".")
    parser.add_argument("--outpfx", help="prefix to use when producing output files")
    parser.add_argument("--style", help="mode-specific interpretation", type=int, default=-1)
    parser.add_argument("--infile", help="input .h5ad file. Default is SlideSeq 180430_1 h5ad")
    parser.add_argument("--njobs", help="number of parallel cores to use", type=int, default=24)

    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    
    args = parser.parse_args()
    assert args.mode is not None
    if args.mode !="raw_data_read": assert args.outpfx is not None
    if args.infile is None:
        args.infile = "/afs/csail.mit.edu/u/r/rsingh/work/schema/data/slideseq/processed/puck_180430_1.h5ad"
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}

    
    if args.mode== "raw_data_read":
        read_raw_files_and_write_h5_files( args.outdir)


    if args.mode == "schema_kd_granule_cells":
        adata1 = SlideSeq.loadAnnData(args.infile)
        try:
            from schema import schema_qp
        except:
            sys.path.append(os.path.join(sys.path[0],'../../schema'))
            import schema_qp

        schema_qp.schema_loglevel = logging.WARNING
        
        fig, d3, schema_ret, min_corr1, min_corr2, scores, cell_loadings = processGranuleCellDensitySchema(adata1, extra_args)
        fig.tight_layout()
        fig.savefig("{0}_fig-KD.png".format(args.outpfx), dpi=300)
        fig.savefig("{0}_fig-KD.svg".format(args.outpfx))
        pickle.dump((d3[["xcoord","ycoord","kd","atlas_cluster"]], schema_ret, min_corr1, min_corr2, scores, cell_loadings),
                    open("{0}_func_output.pkl".format(args.outpfx), "wb"))


    if args.mode == "cca_kd_granule_cells":
        adata1 = SlideSeq.loadAnnData(args.infile)
        if "kd" not in adata1.obs.columns:
            adata1 = computeKernelDensityGranuleCells(adata1,
                                                      kd_fit_granule_only = int(extra_args.get("kd_fit_granule_only",1))==1,
                                                      kd_bw = float(extra_args.get("kd_bw",125)))
        from sklearn.cross_decomposition import CCA
        cca = CCA(1)
        cca.fit(adata1.X, adata1.obs["kd"])            
        cca_sgn = np.sign(scipy.stats.pearsonr(adata1.obs["kd"], cca.x_scores_[:,0])[0]) #flip signs if needed
        fig = generatePlotGranuleCellDensity(adata1.obs, cca_sgn*cca.x_scores_[:,0])
        fig.tight_layout()
        fig.savefig("{0}_fig-CCA.png".format(args.outpfx), dpi=300)
        fig.savefig("{0}_fig-CCA.svg".format(args.outpfx))
        pickle.dump((adata1.obs[["xcoord","ycoord","kd","atlas_cluster"]], cca.x_scores_[:,0], cca.x_loadings_[:,0], cca.y_scores_[:,0]),
                    open("{0}_CCA_output.pkl".format(args.outpfx), "wb"))



    if args.mode == "cca2step_kd_granule_cells":
        adata1 = SlideSeq.loadAnnData(args.infile)
        if "kd" not in adata1.obs.columns:
            adata1 = computeKernelDensityGranuleCells(adata1,
                                                      kd_fit_granule_only = int(extra_args.get("kd_fit_granule_only",1))==1,
                                                      kd_bw = float(extra_args.get("kd_bw",125)))

        #### adata1 = adata1[:,:40] ## FOR TESTING

        from sklearn.cross_decomposition import CCA
        cca1 = CCA(1)
        cca1.fit(adata1.X, adata1.obs["kd"])           
        cca1_sgn = np.sign(scipy.stats.pearsonr(adata1.obs["kd"],cca1.x_scores_[:,0])[0]) #flip signs if needed

        cca2 = CCA(1)
        cca2.fit(adata1.X, 1*(adata1.obs["atlas_cluster"]==1))            
        cca2_sgn = np.sign(scipy.stats.pearsonr(1*(adata1.obs["atlas_cluster"]==1),cca2.x_scores_[:,0])[0]) #flip signs if needed

        score1 = cca1_sgn*cca1.x_scores_[:,0]
        score2 = cca2_sgn*cca2.x_scores_[:,0]
        scorex = 0.5 * (score1/np.std(score1)  + score2/np.std(score2))

        scorex = scorex/np.sqrt(np.sum(scorex**2))

        loadings = np.matmul(np.transpose(adata1.X), scorex)
        intcpt = 0

        print("Flag 2320.01 ", scorex.shape, adata1.X.shape, loadings.shape, describe(scorex), describe(loadings))


        fig = generatePlotGranuleCellDensity(adata1.obs, scorex)
        fig.tight_layout()
        fig.savefig("{0}_fig-CCA2STEP.png".format(args.outpfx), dpi=300)
        fig.savefig("{0}_fig-CCA2STEP.svg".format(args.outpfx))
        pickle.dump((adata1.obs[["xcoord","ycoord","kd","atlas_cluster"]], scorex, loadings, intcpt),
                    open("{0}_CCA2STEP_output.pkl".format(args.outpfx), "wb"))



    if args.mode == "cca_schema_comparison_plot":
        cca_pkl_file = extra_args["cca_pkl_file"]
        schema_pkl_file = extra_args["schema_pkl_file"]
        cca_d3, cca_x_scores, _ , _  = pickle.load(open(cca_pkl_file,"rb"))
        schema_d3,  _, _, _, _, cell_loadings = pickle.load(open(schema_pkl_file,"rb"))
        #fig = doSchemaCCA_CellScorePlot(cca_d3, cca_x_scores, cell_loadings)
        fig = doSchemaCCA_CellScorePlot2(cca_d3, cca_x_scores, cell_loadings)
        fig.savefig("{0}_fig-Schema-CCA-cmp.png".format(args.outpfx), dpi=300)
        fig.savefig("{0}_fig-Schema-CCA-cmp.svg".format(args.outpfx))
        


    if args.mode == "generate_multipuck_gene_ranks":
        pkl_file_glob = extra_args["pkl_file_glob"]
        assert extra_args["data_type"].lower() in ["schema","cca"]
        data_type = extra_args["data_type"].lower()
        
        pkl_flist = glob.glob(pkl_file_glob)
        print("Flag 67.10 ", pkl_flist)
        assert len(pkl_flist) > 0
        
        L = []
        for f in pkl_flist:
            if data_type == "schema":
                _, _, _, _, scores, _ = pickle.load(open(f,"rb")) 
                L.append([a[0] for a in sorted(scores.items(), key=lambda v:v[1])]) #in schema rankings, low number means top-rank

            elif data_type == "cca":
                d3, cca_x_scores, cca_x_loadings, _ = pickle.load(open(f,"rb"))
                cca_sgn = np.sign(scipy.stats.pearsonr(d3["kd"],cca_x_scores)[0])

                puckid = f[f.index("180430"):][:8]
                adata1 = SlideSeq.loadAnnData("{0}/puck_{1}.h5ad".format(os.path.dirname(f), puckid))
                
                df = pd.DataFrame.from_dict({"gene": list(adata1.uns["Wo"].columns), "cca_scores": cca_sgn*cca_x_loadings})
                df = df.sort_values("cca_scores", ascending=False)
                L.append(list(df.gene.values))

        Nmax  = max(len(a) for a in L)
        print ("Flag 67.40 ", len(L), len(L[0]), Nmax)
        cons_score = {}
        active_set = set()
        for i in range(1,Nmax+1):
            currset = set.intersection(*[set(a[:i]) for a in L])
            if len(currset) > len(active_set):
                for s in currset-active_set:
                    cons_score[s] = len(currset)
                active_set = currset

        g = []; s = []
        for k,v in cons_score.items():
            g.append(k)
            s.append(v)

        pd.DataFrame.from_dict({"gene": g, "rank": s}).to_csv("{0}_generankings_dtype-{1}.csv".format(args.outpfx, data_type), index=False)
        
