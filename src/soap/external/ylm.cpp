#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include "ylm.hpp"

void _py_ylm(
        py::array_t<double> x_py, 
        py::array_t<double> y_py, 
        py::array_t<double> z_py,
        int n_pts,
        int lmax, 
        py::array_t<double> py_ylm_out) {
    double *x = (double*) x_py.request().ptr;
    double *y = (double*) y_py.request().ptr;
    double *z = (double*) z_py.request().ptr;
    double *ylm_out = (double*) py_ylm_out.request().ptr;
    double *r = (double*) malloc(sizeof(double)*n_pts);
    for (int i=0; i<n_pts; ++i) {
        r[i] = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
    }
    evaluate_ylm(x, y, z, r, n_pts, lmax, ylm_out);
}

// Unfortunately icc forbids math functions in constexpr
constexpr double radius_eps  = +1.00000000000e-10;  // constexpr double radius_eps = 1e-10;
constexpr double pi          = +3.14159265359e+00;  // constexpr double pi = 3.141592653589793;
constexpr double invs_pi     = +5.64189583548e-01;  // constexpr double invs_pi = sqrt(1./pi);
constexpr double invs_2      = +7.07106781187e-01;  // constexpr double invs_2 = sqrt(0.5);
constexpr double s_2         = +1.41421356237e+00;  // constexpr double s_2 = sqrt(2.);
constexpr double s_3         = +1.73205080757e+00;  // constexpr double s_3 = sqrt(3.);
constexpr double s_5         = +2.23606797750e+00;  // constexpr double s_5 = sqrt(5.);
constexpr double s_7         = +2.64575131106e+00;  // constexpr double s_7 = sqrt(7.);
constexpr double s_11        = +3.31662479036e+00;  // constexpr double s_11 = sqrt(11.);
constexpr double s_13        = +3.60555127546e+00;  // constexpr double s_13 = sqrt(13.);
constexpr double pre1        = +3.45494149471e-01;  // constexpr double pre1 = invs_2*0.5*s_3*invs_pi;
constexpr double pre1_a      = +2.44301255951e-01;  // constexpr double pre1_a = invs_2*pre1;
constexpr double pre2        = +2.23015514519e-01;  // constexpr double pre2 = invs_2* 0.25*s_5*invs_pi;
constexpr double pre2_a      = +5.46274215296e-01;  // constexpr double pre2_a = 2*pre2*s_3*invs_2;
constexpr double pre2_b      = +2.73137107648e-01;  // constexpr double pre2_b = pre2*s_3*invs_2;
constexpr double pre3        = +2.63875515353e-01;  // constexpr double pre3 = invs_2* 0.25*invs_pi*s_7;
constexpr double pre3_a      = +2.28522899732e-01;  // constexpr double pre3_a = 0.5*pre3*s_3;
constexpr double pre3_b      = +7.22652860660e-01;  // constexpr double pre3_b = pre3*s_3*s_5*invs_2;
constexpr double pre3_c      = +2.95021794963e-01;  // constexpr double pre3_c = 0.5*pre3*s_5;
constexpr double pre4        = +7.48016775753e-02;  // constexpr double pre4 = invs_2* 0.1875*invs_pi;
constexpr double pre4_a      = +3.34523271779e-01;  // constexpr double pre4_a = 2*pre4*s_5;
constexpr double pre4_b      = +2.36543673939e-01;  // constexpr double pre4_b = pre4_a*invs_2;
constexpr double pre4_c      = +8.85065384890e-01;  // constexpr double pre4_c = pre4_a*s_7;
constexpr double pre4_d      = +3.12917867725e-01;  // constexpr double pre4_d = pre4_b*0.5*s_7;
constexpr double pre5        = +8.26963660688e-02;  // constexpr double pre5 = invs_2* 0.0625*invs_pi*s_11;
constexpr double pre5_a      = +2.26473325598e-01;  // constexpr double pre5_a = pre5*s_3*s_5*invs_2;
constexpr double pre5_b      = +1.19838419624e+00;  // constexpr double pre5_b = pre5_a*s_7*2;
constexpr double pre5_c      = +2.44619149718e-01;  // constexpr double pre5_c = pre5*0.5*s_7*s_5;
constexpr double pre5_d      = +1.03783115744e+00;  // constexpr double pre5_d = pre5_c*6*invs_2;
constexpr double pre5_e      = +3.28191028420e-01;  // constexpr double pre5_e = pre5*1.5*s_7;
constexpr double pre6        = +4.49502139981e-02;  // constexpr double pre6 = invs_2* 0.03125*invs_pi*s_13;
constexpr double pre6_a      = +2.91310681259e-01;  // constexpr double pre6_a = pre6*s_7*s_3*invs_2*2;
constexpr double pre6_b      = +2.30301314879e-01;  // constexpr double pre6_b = pre6_a*s_5*0.25*s_2;
constexpr double pre6_c      = +4.60602629757e-01;  // constexpr double pre6_c = pre6_b*2;
constexpr double pre6_d      = +2.52282450364e-01;  // constexpr double pre6_d = pre6*s_7*invs_2*3;
constexpr double pre6_e      = +1.18330958112e+00;  // constexpr double pre6_e = pre6_d*s_11*s_2;
constexpr double pre6_f      = +3.41592052596e-01;  // constexpr double pre6_f = pre6_e*s_3/6.;

void evaluate_ylm(
        double *x, double *y, double *z, double *r, 
        int n_pts, int lmax, double *ylm_out) {
    int dim = (lmax+1)*(lmax+1);
    for (int pt=0; pt<n_pts; ++pt) {
        int c = pt*dim;
        ylm_out[c] = 0.5*invs_pi;
        if (r[pt] < radius_eps) {
            for (int lm=1; lm<dim; ++lm) ylm_out[++c] = 0.;
            continue;
        }
        double zr = z[pt]/r[pt];
        double xr = x[pt]/r[pt];
        double yr = y[pt]/r[pt];
        // Angular factors
        //   an = exp(-i*phi*n)*sin^n(theta) = anr + i*ani
        //   bn = exp(+i*phi*n)*sin^n(theta) = bnr + i*bni
        //   cn = cos^n(theta) = cnr + i*0
        double a1r = xr;
        double a1i = -yr;
        double c1r = zr;
        double b1r = xr;
        double b1i = yr;
        double a2r = a1r*a1r - a1i*a1i;
        double a2i = 2*a1r*a1i;
        double c2r = c1r*c1r;
        double c2r_a = 3*c2r-1.;
        double b2r = b1r*b1r - b1i*b1i;
        double b2i = 2*b1r*b1i;
        double a3r = a2r*a1r - a2i*a1i;
        double a3i = a2r*a1i + a2i*a1r;
        double c3r = c2r*c1r;
        double c3r_a = 5*c2r-1.;
        double c3r_b = 5*c3r-3*c1r;
        double b3r = b2r*b1r - b2i*b1i;
        double b3i = b2r*b1i + b2i*b1r;
        double a4r = a2r*a2r - a2i*a2i;
        double a4i = 2*a2r*a2i;
        double c4r = c2r*c2r;
        double c4r_0 = 35.*c4r - 30*c2r + 3.;
        double c4r_1 = 7*c3r - 3*c1r;
        double c4r_2 = 7*c2r - 1.;
        double b4r = b2r*b2r - b2i*b2i;
        double b4i = 2*b2r*b2i;
        double a5r = a4r*a1r - a4i*a1i;
        double a5i = a4r*a1i + a4i*a1r;
        double c5r = c4r*c1r;
        double c5r_0 = 63*c5r - 70*c3r + 15*c1r;
        double c5r_1 = 21*c4r - 14*c2r + 1;
        double c5r_2 = 3*c3r - c1r;
        double c5r_3 = 9*c2r - 1;
        double b5r = b4r*b1r - b4i*b1i;
        double b5i = b4r*b1i + b4i*b1r;
        double a6r = a3r*a3r - a3i*a3i;
        double a6i = 2*a3r*a3i;
        double c6r = c3r*c3r;
        double c6r_0 = 231*c6r - 315*c4r + 105*c2r - 5;
        double c6r_1 = 33*c5r - 30*c3r + 5*c1r;
        double c6r_2 = 33*c4r - 18*c2r + 1;
        double c6r_3 = 11*c3r - 3*c1r;
        double c6r_4 = 11*c2r - 1;
        double b6r = b3r*b3r - b3i*b3i;
        double b6i = 2*b3r*b3i;
        if (lmax > 0) {
            ylm_out[++c] = -pre1_a* (a1i-b1i);        // +-
            ylm_out[++c] =  s_2*pre1*c1r;
            ylm_out[++c] =  pre1_a* (a1r+b1r);        // --
        if (lmax > 1) {
            ylm_out[++c] = -pre2_b*  (a2i-b2i);       // -+
            ylm_out[++c] = -pre2_a*  (a1i - b1i)*c1r; // +-
            ylm_out[++c] =  s_2*pre2*c2r_a;
            ylm_out[++c] =  pre2_a*  (a1r + b1r)*c1r; // --
            ylm_out[++c] =  pre2_b*  (a2r+b2r);       // ++
        if (lmax > 2) {
            ylm_out[++c] = -pre3_c* (a3i-b3i);        // +-
            ylm_out[++c] = -pre3_b* (a2i-b2i)*c1r;    // -+
            ylm_out[++c] = -pre3_a* (a1i-b1i)*c3r_a;  // +-
            ylm_out[++c] =  s_2*pre3*c3r_b;
            ylm_out[++c] =  pre3_a* (a1r+b1r)*c3r_a;  // --
            ylm_out[++c] =  pre3_b* (a2r+b2r)*c1r;    // ++
            ylm_out[++c] =  pre3_c* (a3r+b3r);        // --
        if (lmax > 3) {
            ylm_out[++c] = -pre4_d* (a4i-b4i);        // ..
            ylm_out[++c] = -pre4_c* (a3i-b3i)*c1r;    // ..
            ylm_out[++c] = -pre4_b* (a2i-b2i)*c4r_2;
            ylm_out[++c] = -pre4_a* (a1i-b1i)*c4r_1;
            ylm_out[++c] =  s_2*pre4*c4r_0;
            ylm_out[++c] =  pre4_a* (a1r+b1r)*c4r_1;
            ylm_out[++c] =  pre4_b* (a2r+b2r)*c4r_2;
            ylm_out[++c] =  pre4_c* (a3r+b3r)*c1r;
            ylm_out[++c] =  pre4_d* (a4r+b4r);
        if (lmax > 4) {
            ylm_out[++c] = -pre5_e* (a5i-b5i);
            ylm_out[++c] = -pre5_d* (a4i-b4i)*c1r;
            ylm_out[++c] = -pre5_c* (a3i-b3i)*c5r_3;
            ylm_out[++c] = -pre5_b* (a2i-b2i)*c5r_2;
            ylm_out[++c] = -pre5_a* (a1i-b1i)*c5r_1;
            ylm_out[++c] =  s_2*pre5*c5r_0;
            ylm_out[++c] =  pre5_a* (a1r+b1r)*c5r_1;
            ylm_out[++c] =  pre5_b* (a2r+b2r)*c5r_2;
            ylm_out[++c] =  pre5_c* (a3r+b3r)*c5r_3;
            ylm_out[++c] =  pre5_d* (a4r+b4r)*c1r;
            ylm_out[++c] =  pre5_e* (a5r+b5r);
        if (lmax > 5) {
            ylm_out[++c] = -pre6_f* (a6i-b6i);
            ylm_out[++c] = -pre6_e* (a5i-b5i)*c1r;
            ylm_out[++c] = -pre6_d* (a4i-b4i)*c6r_4;
            ylm_out[++c] = -pre6_c* (a3i-b3i)*c6r_3;
            ylm_out[++c] = -pre6_b* (a2i-b2i)*c6r_2;
            ylm_out[++c] = -pre6_a* (a1i-b1i)*c6r_1;
            ylm_out[++c] =  s_2*pre6*c6r_0;
            ylm_out[++c] =  pre6_a* (a1r+b1r)*c6r_1;
            ylm_out[++c] =  pre6_b* (a2r+b2r)*c6r_2;
            ylm_out[++c] =  pre6_c* (a3r+b3r)*c6r_3;
            ylm_out[++c] =  pre6_d* (a4r+b4r)*c6r_4;
            ylm_out[++c] =  pre6_e* (a5r+b5r)*c1r;
            ylm_out[++c] =  pre6_f* (a6r+b6r);
        }}}}}}
    }
}

