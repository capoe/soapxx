#! /usr/bin/env python
import numpy as np
import json
import sys
import os
import pickle
import copy
import librmt as rmt
import soap
log = rmt.log

log.Connect()
log.AddArg("statefile", typ=str, help="Path to binary file storing state object with fields: 'IX', 'T', 'tag', 'features_with_props'")
log.AddArg("output_folder", typ=str, default='out_npfga', help='Output folder')
log.AddArg("write_jar", typ=bool, default=False, help='Toggle graph archiving')
log.AddArg("write_xup", typ=bool, default=False, help='Write feature array')
log.AddArg("model", typ=bool, default=True, help='Toggle regression')
# CV OPTIONS
log.AddArg("cv_mode", typ=str, default='none', help="CV method: 'loo', 'mc', 'none', 'user'")
log.AddArg("f_mccv", typ=float, default=0.9, help='CV-MC subsampling fraction')
log.AddArg("n_mccv", typ=int, default=10, help='CV-MC repetitions')
log.AddArg("test_on", typ=(list,int), default=[], help="List of sample idcs for test set in mode 'user'")
# GRAPH OPTIONS
log.AddArg("n_iters", typ=int, default=1, help='Number of NPFGA iterations')
log.AddArg("uop", typ=(list,str), default=["el|sr2"], help='Unary operator sequence') #, "r2" ])
log.AddArg("bop", typ=(list,str), default=["+-:*"], help='Binary operator sequence') #, "+-" ])
# FILTERING
log.AddArg("correlation_measure", typ=str, default="moment", help="'moment', 'rank' or 'mixed'")
log.AddArg("rank_coeff", typ=float, default=0.2, help="'moment', 'rank' or 'mixed'")
log.AddArg("bootstrap", typ=int, default=0, help='Number of bootstrap samples used when calculating feature statistics')
log.AddArg("sample", typ=int, default=1000, help='Number of random samples for constructing null distributions')
log.AddArg("tail_fraction", typ=float, default=0.01, help='Tail percentile used for calculating exceedences')
log.AddArg("seed", typ=int, default=830292, help='RNG seed')
log.AddArg("verbose", typ=bool, default=False, help='Verbosity toggle')
# ANALYSIS
log.AddArg("decompose", typ=str, default="", help="Covariance decomposition: 'global', 'top', 'global+top'")
options = log.Parse()

# PREPARATORY STEPS
state_base = rmt.State().unpickle(options.statefile)
cv_iterator = soap.soapy.npfga.cv_iterator[options.cv_mode](state_base, options)
np.random.seed(options.seed)
if not options.verbose: soap.silence()
if not os.path.exists(options.output_folder):
    log >> 'mkdir -p %s' % options.output_folder

# Generate random feature matrices 
log << log.mg << "Sample random feature matrices" << log.endl
rand_IX_list_base = soap.soapy.npfga.RandomizeMatrix(method="perm_within_cols").sample(
    X=state_base["IX"], 
    n_samples=options.sample, 
    seed=None, 
    log=log)

# Generate graph
fgraph = soap.soapy.npfga.generate_graph(
    state_base["features_with_props"], 
    uop_list=options.uop,
    bop_list=options.bop,
    unit_min_exp=0.5, 
    unit_max_exp=3.0,
    correlation_measure=options.correlation_measure,
    rank_coeff=options.rank_coeff)
fnodes = [ f for f in fgraph ]
log << "Generated %d nodes" % len(fnodes) << log.endl

if options.model:
    booster = soap.soapy.npfga.Booster(options)
else:
    booster = None

# ===============
# Apply & analyse
# ===============

feature_gen_log = {}
if options.write_xup:
    xup_file = '%s/%s_xup.npy' % (options.output_folder, os.path.basename(options.statefile))
    IX_up = fgraph.apply(state_base["IX"], str(state_base["IX"].dtype))
    np.save(xup_file, IX_up)
data_log = {
    "state": {
        "statefile": options.statefile,
        "config_tags": [ c["tag"] for c in state_base["configs"] ],
        "IX": state_base["IX"].tolist(),
        "Y": state_base["T"].tolist(),
    },
    "cv_instances": [
    ]
}
while not cv_iterator.isDone():
    # LOAD STATE
    log << log.mg << "Clone state" << log.endl
    state = state_base.clone()

    # TRAIN-TEST SPLITTING
    cv_info_str, idcs_train, idcs_test = cv_iterator.next()
    log << "%s: Train on %d samples, test on %d" % (cv_info_str, len(idcs_train), len(idcs_test)) << log.endl
    log << "- Training idcs" << idcs_train[0:5] << "..." << log.endl
    log << "- Test idcs" << idcs_test[0:5] << "..." << log.endl

    # Descriptor matrix and target
    IX_train = state["IX"][idcs_train]
    IX_test = state["IX"][idcs_test]
    rand_IX_list_train = [ ix[idcs_train] for ix in rand_IX_list_base ]
    Y_train = state["T"][idcs_train]
    Y_test = state["T"][idcs_test]
    rand_Y_train = np.copy(Y_train)
    data_cv_instance = {
        "cv_info": {
            "cv_tag": cv_info_str,
            "idcs_train": idcs_train.tolist() if type(idcs_train) != list else idcs_train,
            "idcs_test": idcs_test.tolist() if type(idcs_test) != list else idcs_test
        },
        "iterations": []
    }

    for iteration in range(options.n_iters):
        log << log.mb << "Iteration %d" % iteration << log.endl
        if booster is not None:
            booster.dispatchY(iteration, Y_train, Y_test)
            Y_train, Y_test = booster.getResidues()
        # Analysis
        log << log.mg << "NPFG analysis" << log.endl
        tags, covs, covs_std, cq_values, cq_values_std, xq_values, xq_values_std, cstats, xstats = soap.soapy.npfga.run_npfga(
            fgraph=fgraph,
            IX=IX_train,
            Y=Y_train,
            rand_IX_list=rand_IX_list_train,
            rand_Y=Y_train,
            options=options,
            log=log)
        q_values = np.max(np.array([xq_values, cq_values]), axis=0)
        q_values_nth = np.max(np.array([xstats.q_values_nth, cstats.q_values_nth]), axis=0)
        corder = np.argsort(np.abs(covs))
        if np.abs(covs[corder[-1]]) > 1.0001:
            log << log.mr << "WARNING Numerical instability:" << log.endl
            log << log.mr << " @ " << fnodes[corder[-1]].expr << log.endl
        corder = filter(lambda o: np.abs(covs[o]) <= 1+1e-10, corder)
        top_idx = corder[-1]

        # HACK >>>>
        #merit_x = xq_values - xq_values_std
        #merit_c = cq_values - cq_values_std
        #merit = np.max(np.array([merit_x, merit_c]), axis=0)
        #merit = merit*(np.abs(covs)-covs_std)
        #top_idx = np.argmax(merit)
        #print "Select:", tags[top_idx]
        #raw_input('...')
        # <<<<<<<<<

        data_section = {
            "npfga_stats": {
                "ftags": tags,
                "covs": covs.tolist(),
                "covs_std": covs_std.tolist(),
                "qxs": xq_values.tolist(),
                "qxs_std": xq_values_std.tolist(),
                "qcs": cq_values.tolist(),
                "qcs_std": cq_values_std.tolist(),
                "order": corder }
        }
        # Expected covariance exceedence
        if "covex" in options.decompose:
            rank_to_idx, pctiles, covx_1st, covx_nth = xstats.calculateCovExceedencePercentiles(log=log)
            log << log.mg << "Covariance exceedence percentiles for '%s'" % tags[rank_to_idx[0]] << log.endl
            for pidx, p in enumerate(pctiles):
                log << " - p=%1.2f:   (cov - <cov>_[1],p) = %+1.4f   (cov - <cov>_[r],p) = %+1.4f" % (
                    0.01*p, q_values[rank_to_idx[0]]*covx_1st[pidx, 0], q_values_nth[rank_to_idx[0]]*covx_nth[pidx, 0]) << log.endl
            data_section["npfga_cov_exceedence"] = {
                "rank_to_idx": rank_to_idx.tolist(),
                "pctiles": pctiles.tolist(),
                "covx_1st": covx_1st.tolist(),
                "covx_nth": covx_nth.tolist()
            }
        if "global" in options.decompose:
            # Decomposition and root weights
            row_tuples, cov_decomposition, cov_decomposition_std = soap.soapy.npfga.run_cov_decomposition(
                fgraph=fgraph,
                IX=IX_train,
                Y=Y_train,
                rand_IX_list=rand_IX_list_train,
                rand_Y=Y_train,
                bootstrap=0, 
                log=log)
            for rank in [-1,-2,-3,-4,-5]:
                log << "Decomposition for phi=%s" % (tags[corder[rank]]) << log.endl
                row_order = np.argsort(cov_decomposition[:,corder[rank]])
                for r in row_order:
                    log << "  i...j = %-50s  cov(i..j) = %+1.4f (+-%1.4f)" % (
                        ':'.join(row_tuples[r]), cov_decomposition[r,corder[rank]], cov_decomposition_std[r,corder[rank]]) << log.endl
            root_tags_sorted, root_weights, root_counts = soap.soapy.npfga.calculate_root_weights(
                fgraph=fgraph,
                q_values=q_values,
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
        if "top" in options.decompose:
            # Decomposition with bootstrap for top node
            row_names, cov_decomposition, cov_decomposition_std = soap.soapy.npfga.run_cov_decomposition_single(
                fgraph=fgraph,
                fnode=fnodes[corder[-1]],
                IX=IX_train,
                Y=Y_train,
                rand_IX_list=rand_IX_list_train,
                rand_Y=Y_train,
                bootstrap=options.bootstrap,
                log=log)
            data_section["npfga_decomposition_top"] = {
                "row_tuples": row_names,
                "cov_decomposition": cov_decomposition.tolist(),
                "cov_decomposition_std": cov_decomposition_std.tolist()
            }
            row_order = np.argsort(cov_decomposition[:,0])
            for r in row_order:
                log << "i...j = %-50s  cov(i..j) = %+1.4f (+-%1.4f)" % (
                    row_names[r], cov_decomposition[r,0], cov_decomposition_std[r,0]) << log.endl
        if "filter" in options.decompose:
            log << "Decomposition filter" << log.endl
            cov_cutoff = np.abs(covs[top_idx])-0.5*covs_std[top_idx]
            corder_short = []
            for rank in xrange(-1, -len(corder)-1, -1):
                if np.abs(covs[corder[rank]]) >= cov_cutoff:
                    log << "- Candidate '%s'" % tags[corder[rank]] << log.endl
                    corder_short.append(corder[rank])
            corder_short.reverse()
            top_idx, root_contributions = soap.soapy.npfga.run_cov_decomposition_filter(
                fgraph=fgraph,
                order=corder_short,
                IX=IX_train,
                Y=Y_train,
                rand_IX_list=rand_IX_list_train,
                rand_Y=Y_train,
                bootstrap=options.bootstrap,
                log=log)
            data_section["npfga_decomposition_filter"] = {
                "root_contributions": root_contributions
            }
        # Regression
        data_section["top"] = {
            "expr":  fnodes[top_idx].expr,
            "idx": corder[top_idx]
        }
        if booster is not None:
            log << "Regression using phi=%s" % fnodes[top_idx].expr << log.endl
            # X train test, Y train test
            selected = [ top_idx ]
            IX_up_train = fgraph.apply(IX_train, str(IX_train.dtype))[:,selected]
            IX_up_test = fgraph.apply(IX_test, str(IX_test.dtype))[:,selected]
            booster.dispatchX(iteration, IX_up_train, IX_up_test)
            booster.train(bootstraps=5000)
            rmse_train, rho_train, rmse_test, rho_test = booster.evaluate()
            log << "Train: RMSE=%1.4e RHO=%1.4e" % (rmse_train, rho_train) << log.endl
            log << "Test:  RMSE=%1.4e RHO=%1.4e" % (rmse_test, rho_test) << log.endl
            np.savetxt("%s/cov_%s_%s.tab" % (
                options.output_folder, 
                cv_info_str+"_iter%d" % iteration, 
                state["tag"]), np.concatenate([ IX_up_train, Y_train.reshape((-1,1))], axis=1))
            data_section["regression"] = {
                "IX_up_train": IX_up_train.tolist(),
                "IX_up_test": IX_up_test.tolist(),
                "Y_train": Y_train.tolist(),
                "Y_test": Y_test.tolist()
            }
        # Summarize
        log << log.mg << "Highest-covariance features" << log.endl
        for idx in corder[-100:]:
            log << "%-70s c=%+1.4f+-%1.2f e=%+1.4f cq=%+1.4f+-%1.2f xq=%+1.4f+-%1.2f" % (
                tags[idx],
                covs[idx],
                covs_std[idx],
                xstats.exs[idx],
                cq_values[idx],
                cq_values_std[idx],
                xq_values[idx],
                xq_values_std[idx]) << log.endl
        xstats.summarize(log)
        # Serialize feature statistics info
        if options.write_jar:
            fgraph_file = "fgraph_%s_%s.arch" % (
                cv_info_str+"_iter%d" % iteration, 
                state["tag"])
            fgraph.save("%s/%s" % (
                options.output_folder, fgraph_file))
            data_section["fgraph"] = fgraph_file
        # Log top feature
        logtag = fnodes[top_idx].expr
        if logtag in feature_gen_log:
            feature_gen_log[logtag]["count"] += 1
            feature_gen_log[logtag]["cv"].append(cv_info_str+"_iter%d" % iteration)
        else:
            feature_gen_log[logtag] = { "count": 1, "cv": [ cv_info_str+"_iter%d" % iteration ] }
        data_cv_instance["iterations"].append(data_section)
    data_log["cv_instances"].append(data_cv_instance)

json.dump(data_log, open("%s/npfga_%s_%s.json" % (
    options.output_folder, os.path.basename(options.statefile), options.cv_mode), 'w'),
    indent=1,
    sort_keys=True)
if booster is not None:
    booster.write("%s/%s_%s_pred_i%%d" % (
        options.output_folder, os.path.basename(options.statefile), options.cv_mode))

log << log.mg << "Feature gen summary" << log.endl
for t in sorted(feature_gen_log, key=lambda f: feature_gen_log[f]["count"]):
    log << "%2d %s" % (feature_gen_log[t]["count"], t) << log.endl
    log << "  "
    for c in feature_gen_log[t]["cv"]:
        log << c
    log << log.endl

