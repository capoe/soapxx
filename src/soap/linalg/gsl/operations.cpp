#include "soap/linalg/operations.hpp"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_eigen.h>

namespace soap { namespace linalg {

using namespace std;

void linalg_cholesky_decompose( ub::matrix<double> &A){
        // Cholesky decomposition using GSL
        const size_t N = A.size1();
        
        gsl_matrix_view A_view = gsl_matrix_view_array(&A(0,0), N, N);
        
        // get the Cholesky matrices
        (void)gsl_linalg_cholesky_decomp ( &A_view.matrix );
}

void linalg_dot(ub::vector<double> &x, ub::vector<double> &y, double &r) {
    gsl_vector_view gsl_x = gsl_vector_view_array(&x(0), x.size());
    gsl_vector_view gsl_y = gsl_vector_view_array(&y(0), y.size());
    gsl_blas_ddot(&gsl_x.vector, &gsl_y.vector, &r);
}

void linalg_cholesky_solve(ub::vector<double> &x, ub::matrix<double> &A, ub::vector<double> &b){
    /* calling program should catch the error error code GSL_EDOM
     * thrown by gsl_linalg_cholesky_decomp and take
     * necessary steps
     */
    
    gsl_matrix_view m
        = gsl_matrix_view_array (&A(0,0), A.size1(), A.size2());

    gsl_vector_view gb
        = gsl_vector_view_array (&b(0), b.size());

    gsl_vector *gsl_x = gsl_vector_alloc (x.size());

    gsl_set_error_handler_off();
    int status = gsl_linalg_cholesky_decomp(&m.matrix);

    if( status == GSL_EDOM)
        throw std::runtime_error("Matrix not symmetric positive definite");

    
    gsl_linalg_cholesky_solve(&m.matrix, &gb.vector, gsl_x);

    for (size_t i =0 ; i < x.size(); i++)
        x(i) = gsl_vector_get(gsl_x, i);

    gsl_vector_free (gsl_x);
}

void linalg_invert(ub::matrix<double> &A, ub::matrix<double> &V){
    // matrix inversion using gsl

    gsl_error_handler_t *handler = gsl_set_error_handler_off();
    const size_t N = A.size1();
    // signum s (for LU decomposition)
    int s;
        //make copy of A as A is destroyed by GSL
        ub::matrix<double> work=A;
        V.resize(N, N, false);

    // Define all the used matrices
        gsl_matrix_view A_view = gsl_matrix_view_array(&work(0,0), N, N);
        gsl_matrix_view V_view = gsl_matrix_view_array(&V(0,0), N, N);
    gsl_permutation * perm = gsl_permutation_alloc (N);

    // Make LU decomposition of matrix A_view
    gsl_linalg_LU_decomp (&A_view.matrix, perm, &s);

    // Invert the matrix A_view
    (void)gsl_linalg_LU_invert (&A_view.matrix, perm, &V_view.matrix);

        gsl_set_error_handler(handler);

    // return (status != 0);
}


}}
