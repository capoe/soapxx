import numpy as np
import json
import sys
import os
import pickle
import copy
import soap
import momo
log = momo.osio

def configure_default():
    default_options = soap.soapy.XSpace()
    default_options.uop = ["el|sr2"]
    default_options.bop = ["+-:*"]
    default_options.correlation_measure = "moment"
    default_options.rank_coeff = 0.2
    default_options.sample = 1000
    default_options.decompose = ""
    default_options.bootstrap = 0
    default_options.tail_fraction = 0.01
    default_options.ranking = "cov"
    return default_options

class FGraphRegressor(object):
    def __init__(self, feature_props, options):
        """
        Arg: feature_props = [ [ 
            tag_str,         # e.g. 'x1'
            sign_str,        # one of ('-','+','+-'), 
            zero_str,        # one of ('-0','+0'),
            prefactor_float, # e.g. 1.0
            unit_str         # e.g. 'kg*m^-1*s^2' ], ...]
        Arg: options = soap.XSpace with required fields
            uop, # e.g. ["+-:*"]
            bop, # e.g. ["el|sr2"]
            correlation_measure, # "moment", "rank", "mixed", "auc"
            rank_coeff, # only used for correlation measure "mixed"
            sample, # number of random samples
            ranking, # "cov", "cov*q", "dcov*dq"
            decompose, # top, global, filter, covex
            bootstrap, # default 0 = none, active only if > 0
            tail_fraction # default 0.01
        """
        self.options = options
        self.fgraph = soap.soapy.npfga.generate_graph(
            feature_props, 
            uop_list=options.uop,
            bop_list=options.bop,
            unit_min_exp=0.5, 
            unit_max_exp=3.0,
            correlation_measure=options.correlation_measure,
            rank_coeff=options.rank_coeff)
        self.corder = None
        self.tags = None
        self.xstats = None
        self.cstats = None
        self.cq_values = None
        self.cq_values_std = None
        self.xq_values = None
        self.xq_values_std = None
        self.q_values = None
        self.q_values_nth = None
        self.covs = None
        self.covs_std = None
        self.top_idx = None
        log << "Generated %d nodes" % len(self.fgraph.nodes()) << log.endl
        return
    def fit(self, X, Y, rand_IX_list=None, rand_Y=None):
        if rand_IX_list is None:
            rand_IX_list = soap.soapy.npfga.RandomizeMatrix(method="perm_within_cols").sample(
                X=X, 
                n_samples=self.options.sample, 
                seed=None, 
                log=log)
        if rand_Y is None:
            rand_Y = np.copy(Y)
        (self.tags, self.covs, self.covs_std, self.cq_values, self.cq_values_std, self.xq_values, 
            self.xq_values_std, self.cstats, self.xstats) = soap.soapy.npfga.run_npfga(
                fgraph=self.fgraph,
                IX=X,
                Y=Y,
                rand_IX_list=rand_IX_list,
                rand_Y=Y,
                options=self.options,
                log=log)
        self.q_values = np.max(np.array([self.xq_values, self.cq_values]), axis=0)
        self.q_values_nth = np.max(np.array([self.xstats.q_values_nth, self.cstats.q_values_nth]), axis=0)
        self.corder = np.argsort(np.abs(self.covs))
        if np.abs(self.covs[self.corder[-1]]) > 1.0001:
            log << log.mr << "WARNING Numerical instability:" << log.endl
            log << log.mr << " @ " << self.fgraph.nodes()[self.corder[-1]].expr << log.endl
        self.corder = filter(lambda o: np.abs(self.covs[o]) <= 1+1e-10, self.corder)
        self.selectTop()
        info = {
            "npfga_stats": {
                "ftags": self.tags,
                "covs": self.covs.tolist(),
                "covs_std": self.covs_std.tolist(),
                "qxs": self.xq_values.tolist(),
                "qxs_std": self.xq_values_std.tolist(),
                "qcs": self.cq_values.tolist(),
                "qcs_std": self.cq_values_std.tolist(),
                "order": self.corder }
        }
        self.decompose(X, Y, rand_IX_list, rand_Y, info)
        # Regression
        info["top"] = {
            "expr":  self.fgraph.nodes()[self.top_idx].expr
        }
        return info
    def transform(self, X, selected=None):
        if selected is None:
            selected = [ self.top_idx ]
        phi = self.fgraph.apply(X, str(X.dtype))[:,selected]
        return phi
    def selectTop(self):
        # Top-node selection
        if self.options.ranking == "cov":
            self.top_idx = self.corder[-1]
        elif self.options.ranking == "cov*q":
            merit_x = self.xq_values - self.xq_values_std
            merit_c = self.cq_values - self.cq_values_std
            merit = np.max(np.array([merit_x, merit_c]), axis=0)
            merit = merit*(np.abs(self.covs)-self.covs_std)
            self.top_idx = np.argmax(merit)
        elif self.options.ranking == "dcov*dq":
            merit_x = self.xq_values
            merit_c = self.cq_values
            merit = np.max(np.array([merit_x, merit_c]), axis=0)
            merit = merit*(np.abs(self.covs)-self.covs_std)
            self.top_idx = np.argmax(merit)
        else: raise ValueError(self.options.ranking)
        log << "Select:" << self.tags[self.top_idx] << log.endl
        return
    def decompose(self, X, Y, rand_IX_list, rand_Y, info={}):
        fnodes = self.fgraph.nodes()
        # Expected covariance exceedence
        if "covex" in self.options.decompose:
            rank_to_idx, pctiles, covx_1st, covx_nth = self.xstats.calculateCovExceedencePercentiles(log=log)
            log << log.mg << "Covariance exceedence percentiles for '%s'" % self.tags[rank_to_idx[0]] << log.endl
            for pidx, p in enumerate(pctiles):
                log << " - p=%1.2f:   (cov - <cov>_[1],p) = %+1.4f   (cov - <cov>_[r],p) = %+1.4f" % (
                    0.01*p, self.q_values[rank_to_idx[0]]*covx_1st[pidx, 0], self.q_values_nth[rank_to_idx[0]]*covx_nth[pidx, 0]) << log.endl
            info["npfga_cov_exceedence"] = {
                "rank_to_idx": rank_to_idx.tolist(),
                "pctiles": pctiles.tolist(),
                "covx_1st": covx_1st.tolist(),
                "covx_nth": covx_nth.tolist()
            }
        if "global" in self.options.decompose:
            # Decomposition and root weights
            row_tuples, cov_decomposition, cov_decomposition_std = soap.soapy.npfga.run_cov_decomposition(
                fgraph=self.fgraph,
                IX=X,
                Y=Y,
                rand_IX_list=rand_IX_list,
                rand_Y=rand_Y,
                bootstrap=0, 
                log=log)
            for rank in [-1,-2,-3,-4,-5]:
                log << "Decomposition for phi=%s" % (self.tags[self.corder[rank]]) << log.endl
                row_order = np.argsort(cov_decomposition[:,self.corder[rank]])
                for r in row_order:
                    log << "  i...j = %-50s  cov(i..j) = %+1.4f (+-%1.4f)" % (
                        ':'.join(row_tuples[r]), cov_decomposition[r,self.corder[rank]], cov_decomposition_std[r,self.corder[rank]]) << log.endl
            root_tags_sorted, root_weights, root_counts = soap.soapy.npfga.calculate_root_weights(
                fgraph=self.fgraph,
                q_values=self.q_values,
                row_tuples=row_tuples,
                cov_decomposition=cov_decomposition,
                log=log)
            data_section["npfga_decomposition"] = {
                "root_weights": {
                    "root_tags_sorted": root_tags_sorted, 
                    "root_weights": root_weights, 
                    "root_counts": root_counts 
                },
                "row_tuples": row_tuples,
                "cov_decomposition": cov_decomposition.tolist()
            }
        if "top" in self.options.decompose:
            # Decomposition with bootstrap for top node
            row_names, cov_decomposition, cov_decomposition_std = soap.soapy.npfga.run_cov_decomposition_single(
                fgraph=self.fgraph,
                fnode=fnodes[self.corder[-1]],
                IX=X,
                Y=Y,
                rand_IX_list=rand_IX_list,
                rand_Y=Y,
                bootstrap=0, #self.options.bootstrap,
                log=log)
            info["npfga_decomposition_top"] = {
                "row_tuples": row_names,
                "cov_decomposition": cov_decomposition.tolist(),
                "cov_decomposition_std": cov_decomposition_std.tolist()
            }
            row_order = np.argsort(cov_decomposition[:,0])
            for r in row_order:
                log << "i...j = %-50s  cov(i..j) = %+1.4f (+-%1.4f)" % (
                    row_names[r], cov_decomposition[r,0], cov_decomposition_std[r,0]) << log.endl
        if "filter" in self.options.decompose:
            log << "Decomposition filter" << log.endl
            corder_short = []
            #cov_cutoff = np.abs(self.covs[self.top_idx])-0.5*self.covs_std[self.top_idx]
            #for rank in xrange(-1, -len(corder)-1, -1):
            #    if np.abs(covs[corder[rank]]) >= cov_cutoff:
            #        log << "- Candidate '%s'" % tags[corder[rank]] << log.endl
            #        corder_short.append(corder[rank])
            top_n = 20
            corder_short = [ self.corder[i] for i in xrange(-1, -top_n-1, -1) ]
            ftags_short = [ fnodes[c].expr for c in corder_short ]
            corder_short.reverse()
            self.top_idx, root_contributions = soap.soapy.npfga.run_cov_decomposition_filter(
                fgraph=self.fgraph,
                order=corder_short,
                IX=X,
                Y=Y,
                rand_IX_list=rand_IX_list,
                rand_Y=rand_Y,
                bootstrap=self.options.bootstrap,
                log=log)
            info["npfga_decomposition_filter"] = {
                "root_contributions": root_contributions
            }
            log << "Select:" << self.tags[self.top_idx] << log.endl
        return
    def summarize(self):
        # Summarize
        log << log.mg << "Highest-covariance features" << log.endl
        for idx in self.corder[-100:]:
            log << "%-70s c=%+1.4f+-%1.2f e=%+1.4f cq=%+1.4f+-%1.2f xq=%+1.4f+-%1.2f" % (
                self.tags[idx],
                self.covs[idx],
                self.covs_std[idx],
                self.xstats.exs[idx],
                self.cq_values[idx],
                self.cq_values_std[idx],
                self.xq_values[idx],
                self.xq_values_std[idx]) << log.endl
        self.xstats.summarize(log)
    def predict(self, X):
        raise NotImplementedError()
        return

