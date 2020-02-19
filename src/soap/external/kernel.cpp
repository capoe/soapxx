#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include "kernel.hpp"

void smooth_match(
        py::array_t<double> arg_P,
        py::array_t<double> arg_K,
        int nx,
        int ny,
        double gamma,
        double epsilon,
        bool verbose) {

    double ax = 1./nx;
    double ay = 1./ny;
    double lambda = 1./gamma;

    double *K = (double*) arg_K.request().ptr;
    double *P = (double*) arg_P.request().ptr;

    double* u = (double*) malloc(sizeof(double)*nx);
    double* u_in = (double*) malloc(sizeof(double)*nx);
    double* v = (double*) malloc(sizeof(double)*ny);
    double* v_in = (double*) malloc(sizeof(double)*ny);
    double* Kg = (double*) malloc(sizeof(double)*nx*ny);
    for (int i=0; i<nx; ++i) { u[i] = 1.; u_in[i] = 1.; }
    for (int j=0; j<ny; ++j) { v[j] = 1.; v_in[j] = 1.; }
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            int ij = i*ny+j;
            Kg[ij] = exp(-(1.-K[ij])*lambda);
        }
    }

    int i_iter = 0;
    double err = 0.;
    while (true) {
        // Update u
        for (int i=0; i<nx; ++i) {
            u[i] = 0.;
            for (int j=0; j<ny; ++j) {
                u[i] += Kg[i*ny+j]*v[j];
            }
        }
        err = 0.;
        for (int i=0; i<nx; ++i) err += pow(ax-u[i]*u_in[i], 2);
        for (int i=0; i<nx; ++i) u[i] = ax/u[i];
        // Update v
        for (int j=0; j<ny; ++j) {
            v[j] = 0.;
            for (int i=0; i<nx; ++i) {
                v[j] += Kg[i*ny+j]*u[i];
            }
        }
        for (int j=0; j<ny; ++j) v[j] = ay/v[j];
        // Converged?
        if (err < epsilon) break;
        for (int i=0; i<nx; ++i) u_in[i] = u[i];
        for (int j=0; j<ny; ++j) v_in[j] = v[j];
        i_iter += 1;
    }
    if (verbose)
        std::cout << "Converged " << nx << "x" << ny << " in " 
            << i_iter << " iterations" << std::endl;

    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            int ij = i*ny+j;
            P[ij] = u[i]*Kg[ij]*v[j];
        }
    }

    free(u);
    free(u_in);
    free(v);
    free(v_in);
    free(Kg);
}

