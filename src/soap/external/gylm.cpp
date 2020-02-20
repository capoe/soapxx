#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include "gylm.hpp"
#include "ylm.hpp"
#include "celllist.hpp"

constexpr double radial_epsilon = 1e-10;

void evaluate_deltas(
        double xi, double yi, double zi, 
        double *tpos,
        double *dx, double *dy, double *dz,
        double *dr, double *dr2,
        const vector<int> &nbs) {
    int jj = 0;
    for (const int &j : nbs) {
        double dx_jj = tpos[3*jj] - xi;
        double dy_jj = tpos[3*jj+1] - yi;
        double dz_jj = tpos[3*jj+2] - zi;
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
        double *gn, double *hl, double *gnl) {
    double part_sigma = 0.5;
    double wcentre = 1.;
    double wscale = 0.5;
    double invs_wscale2 = 1./(wscale*wscale);
    double ldamp = 4.;
    int c_gnl = -1;
    for (int j=0; j<n_parts; ++j) {
        double w = 1.;
        if (dr[j] < radial_epsilon) w = wcentre;
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

void evaluate_qitnlm(double *jw, double *jgnl, double *jylm, 
        int part_idx, int type_index, int n_nbs, 
        int n_types, int nmax, int lmax, 
        double *qitnlm) {
    int dim_nlm = nmax*(lmax+1)*(lmax+1);
    int off = part_idx*type_index*dim_nlm;
    int c_itnlm = off-1;
    for (int nlm=0; nlm<dim_nlm; ++nlm) qitnlm[++c_itnlm] = 0.;
    int c_gnl = 0;
    int c_ylm = 0;
    for (int j=0; j<n_nbs; ++j) {
        c_itnlm = off-1;
        for (int n=0; n<nmax; ++n) {
            for (int l=0; l<lmax+1; ++l) {
                for (int m=-l; m <l+1; ++m) {
                    qitnlm[++c_itnlm] += jw[j]*jgnl[c_gnl]*jylm[c_ylm];
                    ++c_ylm;
                }
                ++c_gnl;
            }
        }
    }
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
        bool verbose) {

    // Get array pointers
    double *gnl_centres = (double*) gnl_centres_py.request().ptr;
    double *gnl_alphas = (double*) gnl_alphas_py.request().ptr;
    double *spos = (double*) src_pos.request().ptr;
    double *tpos = (double*) tgt_pos.request().ptr;
    int *ttypes = (int*) tgt_types.request().ptr;
    std::cout << "# targets, sources =" << n_tgt << "," << n_src << std::endl;
    for (int i=0; i<n_src; ++i) {
        std::cout << "src @ " << spos[3*i] << " " << spos[3*i+1] << " " << spos[3*i+2] << std::endl;
    }
    for (int i=0; i<n_tgt; ++i) {
        std::cout << ttypes[i] << " @ " << tpos[3*i] << " " << tpos[3*i+1] << " " << tpos[3*i+2] << std::endl;
    }
    for (int i=0; i<nmax; ++i) {
        std::cout << "G of width " << gnl_alphas[i] << " @ " << gnl_centres[i] << std::endl;
    }

    // Cell list
    CellList cell_list(tgt_pos, r_cut);

    // Type mapping
    auto tgt_types_list = tgt_types.unchecked<1>();
    auto all_types_list = all_types.unchecked<1>();
    map<int, int> type_index_map;
    set<int> type_set;
    for (int i=0; i<n_types; ++i) {
        if (verbose) std::cout << "Register type " << all_types_list(i) << std::endl;
        type_set.insert(all_types_list(i));
    }
    int type_index = 0;
    for (auto it=type_set.begin(); it!=type_set.end(); ++it) {
        if (verbose) std::cout << "Place type " << *it << " at index " << type_index << std::endl;
        type_index_map[*it] = type_index++;
    }

    // Ancillary arrays
    double *dx   = (double*) malloc(sizeof(double)*n_tgt);
    double *dy   = (double*) malloc(sizeof(double)*n_tgt);
    double *dz   = (double*) malloc(sizeof(double)*n_tgt);
    double *dr   = (double*) malloc(sizeof(double)*n_tgt);
    double *dr2  = (double*) malloc(sizeof(double)*n_tgt);
    double *jw   = (double*) malloc(sizeof(double)*n_tgt);
    double *jgnl = (double*) malloc(sizeof(double)*n_tgt*nmax*(lmax+1));
    double *jgn  = (double*) malloc(sizeof(double)*nmax);
    double *jhl  = (double*) malloc(sizeof(double)*(lmax+1));
    double *jylm = (double*) malloc(sizeof(double)*n_tgt*(lmax+1)*(lmax+1));
    // Expansions
    // n_src x t x n x lm
    int dim_itnlm = n_src*n_types*nmax*(lmax+1)*(lmax+1);
    //int dim_itunkl = n_src*n_types*(n_types+1)/2*nmax*nmax*(lmax+1);
    double *qitnlm = (double*) malloc(sizeof(double)*dim_itnlm);
    //double *xitunkl = (double*) malloc(sizeof(double)*dim_itunkl);

    int src_pos_idx = -1;
    for (int src_idx=0; src_idx<n_src; ++src_idx) {
        double xi = spos[++src_pos_idx];
        double yi = spos[++src_pos_idx];
        double zi = spos[++src_pos_idx];
        CellListResult nbs = cell_list.getNeighboursForPosition(xi, yi, zi);
        map<int, vector<int>> nb_type_map;
        for (const int &j : nbs.indices) {
            int t = tgt_types_list(j);
            nb_type_map[t].push_back(j);
        }
        for (const auto &nbs_of_type : nb_type_map) {
            int type_index = type_index_map[nbs_of_type.first];
            int n_nbs_of_type = nbs_of_type.second.size();
            if (verbose) {
                std::cout << "Src " << src_idx << " : " << n_nbs_of_type 
                    << " nbs of type " << type_index << std::endl;
            }
            evaluate_deltas(xi, yi, zi, tpos, 
                dx, dy, dz, dr, dr2, 
                nbs_of_type.second);
            evaluate_weights(dr, n_nbs_of_type, 
                r_cut, r_cut_width, jw);
            evaluate_gnl(dr, dr2, n_nbs_of_type, 
                gnl_centres, gnl_alphas, nmax, lmax, jgn, jhl, jgnl);
            evaluate_ylm(dx, dy, dz, dr,
                n_nbs_of_type, lmax, jylm);
            evaluate_qitnlm(jw, jgnl, jylm, 
                src_idx, type_index, n_nbs_of_type, n_types, nmax, lmax,
                qitnlm);
        }
    }

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
    free(qitnlm);
    return;
}


