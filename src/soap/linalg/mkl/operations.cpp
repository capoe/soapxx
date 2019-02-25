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

void linalg_dot(ub::vector<float> &x, ub::vector<float> &y, float &r) {
    MKL_INT n = x.size();
    MKL_INT incr = 1;
    float *mkl_x = const_cast<float*>(&x.data()[0]);
    float *mkl_y = const_cast<float*>(&y.data()[0]);
    r = cblas_sdot(n, mkl_x, incr, mkl_y, incr);
}

void linalg_matrix_vector_dot(
    ub::matrix<double> &A, 
    ub::vector<double> &b, 
    ub::vector<double> &c,
    bool transpose,
    double alpha,
    double beta) {
    MKL_INT m = A.size1();
    MKL_INT n = A.size2();
    MKL_INT incr = 1;
    //if (transpose) {
    //    k = A.size1();
    //    m = A.size2();
    //}
    double *pA = const_cast<double*>(&A.data().begin()[0]);
    double *pb = const_cast<double*>(&b.data()[0]);
    double *pc = const_cast<double*>(&c.data()[0]);
    cblas_dgemv(CblasRowMajor, (transpose) ? CblasTrans : CblasNoTrans,
        m, n, alpha, pA, n, pb, incr, beta, pc, incr);
}

void linalg_matrix_dot(ub::matrix<double> &A, ub::matrix<double> &B, ub::matrix<double> &C) {
    // Inputs A: (m x k), B: (k x n) -> C: (m x n)
    MKL_INT m = A.size1();
    MKL_INT n = B.size2();
    MKL_INT k = A.size2();
    double alpha = 1.0;
    double beta = 1.0;
    double *pA = const_cast<double*>(&A.data().begin()[0]);
    double *pB = const_cast<double*>(&B.data().begin()[0]);
    double *pC = const_cast<double*>(&C.data().begin()[0]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           m, n, k, alpha, pA, k, pB, n, beta, pC, n); 
}

void linalg_matrix_block_dot(
    ub::matrix<double> &A, 
    ub::matrix<double> &B, 
    ub::matrix<double> &C,
    int i_off, int j_off) {
    MKL_INT m = A.size1();
    MKL_INT n = B.size2();
    MKL_INT k = A.size2(); // NOTE Here != B.size1()
    double alpha = 1.0;
    double beta = 1.0;
    double *pA = const_cast<double*>(&A.data().begin()[0]);
    double *pB = const_cast<double*>(&B.data().begin()[0]);
    double *pC = const_cast<double*>(&C.data().begin()[0]);
    pB += j_off*n;
    pC += i_off*n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           m, n, k, alpha, pA, k, pB, n, beta, pC, n); 
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
