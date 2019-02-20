#include "soap/linalg/operations.hpp"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <mkl.h>
#include <mkl_lapacke.h>

namespace soap { namespace linalg {

using namespace std;

void linalg_dot(ub::vector<double> &x, ub::vector<double> &y, double &r) {
    MKL_INT n = x.size();
    MKL_INT incr = 1;
    double *mkl_x = const_cast<double*>(&x.data()[0]);
    double *mkl_y = const_cast<double*>(&y.data()[0]);
    r = cblas_ddot(n, mkl_x, incr, mkl_y, incr);
}

void linalg_cholesky_decompose( ub::matrix<double> &A){
    // Cholesky decomposition using MKL
    // input matrix A will be changed

    // LAPACK variables
    MKL_INT info;
    MKL_INT n = A.size1();
    char uplo = 'L';
    
    // pointer for LAPACK
    double * pA = const_cast<double*>(&A.data().begin()[0]);
    info = LAPACKE_dpotrf( LAPACK_ROW_MAJOR , uplo , n, pA, n );
    if ( info != 0 )
        throw std::runtime_error("Matrix not symmetric positive definite");
}

void linalg_cholesky_solve( ub::vector<double> &x, ub::matrix<double> &A, ub::vector<double> &b ){
    /* calling program should catch the error error code
     * thrown by LAPACKE_dpotrf and take
     * necessary steps
     */
    
    
    // LAPACK variables
    MKL_INT info;
    MKL_INT n = A.size1();
    char uplo = 'L';
    
    // pointer for LAPACK LU factorization of input matrix
    double * pA = const_cast<double*>(&A.data().begin()[0]); // input array
     
    // get LU factorization
    info = LAPACKE_dpotrf( LAPACK_ROW_MAJOR , uplo , n, pA, n );
    
    if ( info != 0 )
        throw std::runtime_error("Matrix not symmetric positive definite");
    
    MKL_INT nrhs = 1;
    
    // pointer of LAPACK LU solver
    double * pb = const_cast<double*>(&b.data()[0]);
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, uplo, n, nrhs, pA, n, pb, n );

    // on output, b contains solution
    x = b;
}

void linalg_invert( ub::matrix<double> &A, ub::matrix<double> &V){
    // matrix inversion using MKL
    // input matrix is destroyed, make local copy
    ub::matrix<double> work = A;
    
    // define LAPACK variables
    MKL_INT n = A.size1();
    MKL_INT info;
    MKL_INT ipiv[n];
    
    // initialize V
    V = ub::identity_matrix<double>(n,n);

    // pointers for LAPACK
    double * pV = const_cast<double*>(&V.data().begin()[0]);
    double * pwork = const_cast<double*>(&work.data().begin()[0]);
    
    // solve
    info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, n, pwork , n, ipiv, pV, n );
}


}}
