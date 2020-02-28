#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>
#include "gylm.hpp"
#include "ylm.hpp"
#include "celllist.hpp"

// Unfortunately icc forbids use of math functions in constexpr
constexpr double radial_epsilon  = +1.00000000000e-10; // constexpr double radial_epsilon = 1e-10;
constexpr double lnorm_0         = +1.00000000000e+00; // constexpr double lnorm_0  = 1.;
constexpr double lnorm_1         = +5.77350269190e-01; // constexpr double lnorm_1  = 1./sqrt(3.);
constexpr double lnorm_2         = +4.47213595500e-01; // constexpr double lnorm_2  = 1./sqrt(5.);
constexpr double lnorm_3         = +3.77964473009e-01; // constexpr double lnorm_3  = 1./sqrt(7.);
constexpr double lnorm_4         = +3.33333333333e-01; // constexpr double lnorm_4  = 1./sqrt(9.);
constexpr double lnorm_5         = +3.01511344578e-01; // constexpr double lnorm_5  = 1./sqrt(11.);
constexpr double lnorm_6         = +2.77350098113e-01; // constexpr double lnorm_6  = 1./sqrt(13.);
constexpr double lnorm_7         = +2.58198889747e-01; // constexpr double lnorm_7  = 1./sqrt(15.);
constexpr double lnorm_8         = +2.42535625036e-01; // constexpr double lnorm_8  = 1./sqrt(17);
constexpr double lnorm_9         = +2.29415733871e-01; // constexpr double lnorm_9  = 1./sqrt(19.);
constexpr double lnorm_10        = +2.18217890236e-01; // constexpr double lnorm_10 = 1./sqrt(21.);
constexpr double lnorm_11        = +2.08514414057e-01; // constexpr double lnorm_11 = 1./sqrt(23.);

void evaluate_deltas(
        double xi, double yi, double zi, 
        double *tpos,
        double *dx, double *dy, double *dz,
        double *dr, double *dr2,
        const vector<int> &nbs) {
    int jj = 0;
    for (const int &j : nbs) {
        double dx_jj = tpos[3*j] - xi;
        double dy_jj = tpos[3*j+1] - yi;
        double dz_jj = tpos[3*j+2] - zi;
        dr2[jj] = dx_jj*dx_jj + dy_jj*dy_jj + dz_jj*dz_jj;
        dr[jj] = sqrt(dr2[jj]);
        dx[jj] = dx_jj;
        dy[jj] = dy_jj;
        dz[jj] = dz_jj;
        ++jj;
    }
}

void evaluate_weights(
        double *dr, int n_parts, 
        double r_cut, double r_cut_width, 
        double *w) {
    for (int j=0; j<n_parts; ++j) {
        if (dr[j] <= r_cut-r_cut_width) w[j] = 1.;
        else if (dr[j] >= r_cut) w[j] = 0.;
        else {
            w[j] = 0.5*(1+cos(M_PI*(dr[j]-r_cut+r_cut_width)/r_cut_width));
        }
    }
}

void evaluate_gnl(
        double *dr, double *dr2, int n_parts, 
        double *centres, double *alphas, 
        int nmax, int lmax, 
        double *gn, double *hl, double *gnl,
        double part_sigma,
        double wconstant,
        double wscale,
        double wcentre,
        double ldamp) {
    double invs_wscale2 = 1./(wscale*wscale);
    int c_gnl = -1;
    for (int j=0; j<n_parts; ++j) {
        double w = 1.;
        if (wconstant || dr[j] < radial_epsilon) w = wcentre;
        else {
            double r2eff = dr2[j]*invs_wscale2;
            double t = exp(-r2eff);
            w = (1.-t)/r2eff + t*(wcentre-1.);
        }
        for (int n=0; n<nmax; ++n) {
            gn[n] = w*exp(-alphas[n]*(centres[n]-dr[j])*(centres[n]-dr[j]));
        }
        for (int l=0; l<lmax+1; ++l) {
            hl[l] = exp(-
                sqrt(2*l)*ldamp*part_sigma /
                (sqrt(4.*M_PI)*dr[j] + radial_epsilon)
            );
        }
        for (int n=0; n<nmax; ++n) {
            for (int l=0; l<lmax+1; ++l) {
                gnl[++c_gnl] = gn[n]*hl[l];
            }
        }
    }
}

void evaluate_qitnlm(double *jw, double *jgnl, double *jylm, int n_nbs, 
        int nmax, int lmax, int offset_idx, int dim_nlm, int dim_nl, int dim_lm,
        double *qitnlm) {
    // >>> # NOTE Array layout:
    // >>> i = 0
    // >>> for nb in nbs:
    // >>>     for l in range(lmax+1):
    // >>>         for m in range(-l,l+1):
    // >>>             ylmi = ylm[i++]
    // >>> i = 0
    // >>> for nb in nbs:
    // >>>     for n in range(nmax):
    // >>>         for l in range(lmax+1):
    // >>>             gnli = gnl[i++]
    // >>> # The offset index takes into account source 
    // >>> # (= centre) and nb type index
    // >>> offset = src_idx*dim_tnlm + type_idx*dim_nlm
    int c_itnlm = offset_idx;
    for (int nlm=0; nlm<dim_nlm; ++nlm) qitnlm[c_itnlm++] = 0.;
    int c_gnl = 0;
    for (int j=0; j<n_nbs; ++j) {
        c_itnlm = offset_idx;
        // Note that here c_gnl == j*dim_nl
        for (int n=0; n<nmax; ++n) {
            int c_ylm = j*dim_lm;
            for (int l=0; l<lmax+1; ++l) {
                for (int m=-l; m<l+1; ++m) {
                    qitnlm[c_itnlm++] += jw[j]*jgnl[c_gnl]*jylm[c_ylm];
                    ++c_ylm;
                }
                ++c_gnl;
            }
        }
    }
}

void evaluate_xtunkl(double *xitunkl, double *qitnlm, vector<double> &lnorm,
        int n_src, int n_types, int nmax, int lmax, 
        int dim_tnlm, int dim_nlm, int dim_lm) {
    int itunkl = 0;
    for (int i=0; i<n_src; ++i) {
        for (int t=0; t<n_types; ++t) {
            int tnlm_off = i*dim_tnlm + t*dim_nlm;
            for (int u=t; u<n_types; ++u) {
                int uklm_off = i*dim_tnlm + u*dim_nlm;
                for (int n=0; n<nmax; ++n) {
                    for (int k=0; k<nmax; ++k) {
                        int tnlm = tnlm_off + n*dim_lm;
                        int uklm = uklm_off + k*dim_lm;
                        for (int l=0; l<lmax+1; ++l) {
                            double x = 0.;
                            for (int m=-l; m<l+1; ++m) {
                                x += qitnlm[tnlm++]*qitnlm[uklm++];
                            }
                            xitunkl[itunkl++] = lnorm[l]*x;
                        }
                    }
                }
            }
        }
    }
}

void evaluate_xtunkl(double *xitunkl, double *qitnlm, vector<double> &lnorm, 
    int src_idx, set<int> &nb_type_indices, int n_types, int nmax, int lmax,
    int dim_tunkl, int dim_nkl, int dim_tnlm, int dim_nlm, int dim_lm) {
    int itunkl_off = src_idx*dim_tunkl;
    int itunkl = 0;
    for (auto t_it=nb_type_indices.begin(); t_it!=nb_type_indices.end(); ++t_it) {
        int t = *t_it;
        int tnlm_off = src_idx*dim_tnlm + t*dim_nlm;
        for (auto u_it=t_it; u_it!=nb_type_indices.end(); ++u_it) {
            int u = *u_it;
            int uklm_off = src_idx*dim_tnlm + u*dim_nlm;
            itunkl = itunkl_off + (
                t*n_types - t*(t+1)/2 + u)*dim_nkl;
            for (int n=0; n<nmax; ++n) {
                for (int k=0; k<nmax; ++k) {
                    int tnlm = tnlm_off + n*dim_lm;
                    int uklm = uklm_off + k*dim_lm;
                    for (int l=0; l<lmax+1; ++l) {
                        double x = 0.;
                        for (int m=-l; m<l+1; ++m) {
                            x += qitnlm[tnlm++]*qitnlm[uklm++];
                        }
                        xitunkl[itunkl++] = lnorm[l]*x;
                    }
                }
            }
        }
    }
}

void _py_evaluate_xtunkl(
    py::array_t<double> _xitunkl,
    py::array_t<double> _qitnlm,
    int n_src,
    int n_types,
    int nmax,
    int lmax) {
    double *xitunkl = (double*) _xitunkl.request().ptr;
    double *qitnlm = (double*) _qitnlm.request().ptr;
    int dim_lm = (lmax+1)*(lmax+1);
    int dim_nlm = nmax*dim_lm;
    int dim_tnlm = n_types*dim_nlm;
    vector<double> lnorm = { 
        lnorm_0, lnorm_1, lnorm_2,  lnorm_3,
        lnorm_4, lnorm_5, lnorm_6,  lnorm_7,
        lnorm_8, lnorm_9, lnorm_10, lnorm_11 };
    evaluate_xtunkl(xitunkl, qitnlm, lnorm,
        n_src, n_types, nmax, lmax,
        dim_tnlm, dim_nlm, dim_lm);
}

void evaluate_gylm(
        py::array_t<double> coeffs, 
        py::array_t<double> src_pos, 
        py::array_t<double> tgt_pos, 
        py::array_t<double> gnl_centres_py, 
        py::array_t<double> gnl_alphas_py, 
        py::array_t<int> tgt_types, 
        py::array_t<int> all_types, 
        double r_cut, 
        double r_cut_width, 
        int n_src, 
        int n_tgt, 
        int n_types, 
        int nmax, 
        int lmax,
        double part_sigma,
        bool wconstant,
        double wscale,
        double wcentre,
        double ldamp,
        bool power,
        bool verbose) {

    // Get array pointers
    double *xitunkl = (double*) coeffs.request().ptr;
    double *gnl_centres = (double*) gnl_centres_py.request().ptr;
    double *gnl_alphas = (double*) gnl_alphas_py.request().ptr;
    double *spos = (double*) src_pos.request().ptr;
    double *tpos = (double*) tgt_pos.request().ptr;
    int *ttypes = (int*) tgt_types.request().ptr;

    // System info
    if (verbose) {
        std::cout << "# targets, sources =" << n_tgt 
            << "," << n_src << std::endl;
        for (int i=0; i<n_src; ++i) {
            std::cout << "src @ " << spos[3*i] << " " 
                << spos[3*i+1] << " " << spos[3*i+2] << std::endl;
        }
        for (int i=0; i<n_tgt; ++i) {
            std::cout << ttypes[i] << " @ " << tpos[3*i] 
                << " " << tpos[3*i+1] << " " << tpos[3*i+2] << std::endl;
        }
        for (int i=0; i<nmax; ++i) {
            std::cout << "G of width " << gnl_alphas[i] 
                << " @ " << gnl_centres[i] << std::endl;
        }
    }

    // Cell list
    CellList cell_list(tgt_pos, r_cut);

    // Type mapping
    auto tgt_types_list = tgt_types.unchecked<1>();
    auto all_types_list = all_types.unchecked<1>();
    map<int, int> type_index_map;
    set<int> type_set;
    for (int i=0; i<n_types; ++i) {
        if (verbose) std::cout << "Register type " 
            << all_types_list(i) << std::endl;
        type_set.insert(all_types_list(i));
    }
    int type_index = 0;
    for (auto it=type_set.begin(); it!=type_set.end(); ++it) {
        if (verbose) std::cout << "Place type " 
            << *it << " at index " << type_index << std::endl;
        type_index_map[*it] = type_index++;
    }

    // Ancillary arrays
    int dim_nl = nmax*(lmax+1);
    int dim_lm = (lmax+1)*(lmax+1);
    int dim_nlm = nmax*dim_lm;
    int dim_tnlm = n_types*dim_nlm;
    int dim_itnlm = n_src*dim_tnlm;
    int dim_nkl = nmax*nmax*(lmax+1);
    int dim_tunkl = n_types*(n_types+1)/2*nmax*nmax*(lmax+1);
    int dim_itunkl = n_src*n_types*(n_types+1)/2*nmax*nmax*(lmax+1);
    vector<double> lnorm = { 
        lnorm_0, lnorm_1, lnorm_2,  lnorm_3,
        lnorm_4, lnorm_5, lnorm_6,  lnorm_7,
        lnorm_8, lnorm_9, lnorm_10, lnorm_11 };
    // Deltas
    double *dx   = (double*) malloc(sizeof(double)*n_tgt);
    double *dy   = (double*) malloc(sizeof(double)*n_tgt);
    double *dz   = (double*) malloc(sizeof(double)*n_tgt);
    double *dr   = (double*) malloc(sizeof(double)*n_tgt);
    double *dr2  = (double*) malloc(sizeof(double)*n_tgt);
    // Weights, Gnl's, Ylm's
    double *jw   = (double*) malloc(sizeof(double)*n_tgt);
    double *jgnl = (double*) malloc(sizeof(double)*n_tgt*dim_nl);
    double *jgn  = (double*) malloc(sizeof(double)*nmax);
    double *jhl  = (double*) malloc(sizeof(double)*(lmax+1));
    double *jylm = (double*) malloc(sizeof(double)*n_tgt*dim_lm);
    // Qtnlm's
    double *qitnlm = NULL;
    if (power) {
        qitnlm = (double*) malloc(sizeof(double)*dim_itnlm);
        for (int i=0; i<dim_itnlm; ++i) qitnlm[i] = 0.;
    }
    else {
        qitnlm = xitunkl;
    }

    // Expansion loop
    int src_pos_idx = -1;
    for (int src_idx=0; src_idx<n_src; ++src_idx) {
        double xi = spos[++src_pos_idx];
        double yi = spos[++src_pos_idx];
        double zi = spos[++src_pos_idx];
        if (verbose) {
            std::cout << std::endl;
            std::cout << "Src @ " << xi << " " << yi << " " << zi << std::endl;
        }
        CellListResult nbs = cell_list.getNeighboursForPosition(xi, yi, zi);
        map<int, vector<int>> nb_type_map;
        for (const int &j : nbs.indices) {
            int t = tgt_types_list(j);
            nb_type_map[t].push_back(j);
        }
        set<int> nb_type_indices;
        for (const auto &nbs_of_type : nb_type_map) {
            int type_index = type_index_map[nbs_of_type.first];
            int n_nbs_of_type = nbs_of_type.second.size();
            nb_type_indices.insert(type_index);
            if (verbose) {
                std::cout << "Src " << src_idx << " : " << n_nbs_of_type 
                    << " nbs of type " << type_index << std::endl;
            }
            evaluate_deltas(xi, yi, zi, tpos, 
                dx, dy, dz, dr, dr2, 
                nbs_of_type.second);
            if (verbose) {
                for (int j=0; j<n_nbs_of_type; ++j) {
                    std::cout << "  " << dx[j] << " " << dy[j] << " " << dz[j] 
                        << " r=" << dr[j] << std::endl;
                }
            }
            // Weight coefficients, Gnl's, Ylm's
            evaluate_weights(dr, n_nbs_of_type, 
                r_cut, r_cut_width, jw);
            evaluate_gnl(dr, dr2, n_nbs_of_type, 
                gnl_centres, gnl_alphas, nmax, lmax, jgn, jhl, jgnl,
                part_sigma, wconstant, wscale, wcentre, ldamp);
            evaluate_ylm(dx, dy, dz, dr,
                n_nbs_of_type, lmax, jylm);
            // Expansion coefficients Qnlm's
            int offset_idx = src_idx*dim_tnlm + type_index*dim_nlm;
            if (verbose) {
                std::cout << "  Store @ " << offset_idx 
                    << ", length = " << dim_nlm << std::endl;
            }
            evaluate_qitnlm(jw, jgnl, jylm, n_nbs_of_type, 
                nmax, lmax, offset_idx, dim_nlm, dim_nl, dim_lm,
                qitnlm);
        }
        // Contractions for this source
        if (power) evaluate_xtunkl(xitunkl, qitnlm, lnorm, src_idx, nb_type_indices,
            n_types, nmax, lmax, dim_tunkl, dim_nkl, dim_tnlm, dim_nlm, dim_lm);
    }

    // Contractions
    if (verbose) {
        std::cout << "Contractions per centre: " << dim_tnlm  << " o " << dim_tnlm  
            << " -> " << dim_tunkl << std::endl;
        std::cout << "Contractions for system: " << dim_itnlm << " o " << dim_itnlm 
            << " -> " << dim_itunkl << std::endl;
    }
    // >>> if (power) {
    // >>>     evaluate_xtunkl(xitunkl, qitnlm, lnorm,
    // >>>         n_src, n_types, 
    // >>>         nmax, lmax, dim_tnlm, dim_nlm, dim_lm);
    // >>> }

    free(dx);
    free(dy);
    free(dz);
    free(dr);
    free(dr2);
    free(jw);
    free(jgnl);
    free(jgn);
    free(jhl);
    free(jylm);
    if (power) {
        free(qitnlm);
    }
    return;
}


