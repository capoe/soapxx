#ifndef _SOAP_LINALG_OPERATIONS_HPP
#define	_SOAP_LINALG_OPERATIONS_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

namespace soap { namespace linalg {
    namespace ub = boost::numeric::ublas;

    // Standard vector inner (dot) product c = x.y
    void linalg_dot(
        ub::vector<double> &x, 
        ub::vector<double> &y, 
        double &c);

    void linalg_dot(
        ub::vector<float> &x, 
        ub::vector<float> &y, 
        float &c);

    // Standard (inner) matrix-vector product c = alpha*A^(t).b + beta*c
    void linalg_matrix_vector_dot(
        ub::matrix<double> &A, 
        ub::vector<double> &b, 
        ub::vector<double> &c,
        bool transpose,
        double alpha,
        double beta);

    // Standard (inner) matrix product C = A.B
    void linalg_matrix_dot(
        ub::matrix<double> &A, 
        ub::matrix<double> &B, 
        ub::matrix<double> &C);

    void linalg_matrix_dot(
        ub::matrix<double> &A, 
        ub::matrix<double> &B, 
        ub::matrix<double> &C,
        double alpha,
        double beta,
        bool transpose_A,
        bool transpose_B);

    // Component-wise multiplication
    void linalg_mul(
        ub::matrix<double> &A, 
        ub::matrix<double> &B,
        ub::matrix<double> &C,
        int n,
        int off_A,
        int off_B,
        int off_C);

    void linalg_mul(
        ub::matrix<double> &A, 
        ub::vector<double> &b,
        ub::matrix<double> &C,
        int n,
        int off_A,
        int off_b,
        int off_C);

    // Component-wise subtraction
    void linalg_sub(
        ub::matrix<double> &A, 
        ub::vector<double> &b,
        ub::matrix<double> &C,
        int n,
        int off_A,
        int off_b,
        int off_C);

    // Matrix product between a small mxn block A and large NxK matrix B,
    // where A is excised from a larger block-diagonal MxN matrix AA. The
    // upper left corner of A starts at (i_off, j_off) within AA.
    // Output C should have dimension MxN, where M is the number of rows
    // of the block-diagonal matrix AA and N the number of columns of B.
    void linalg_matrix_block_dot(
        ub::matrix<double> &A, 
        ub::matrix<double> &B, 
        ub::matrix<double> &C,
        int i_off, int j_off);

    /**
     * \brief inverts A
     * @param A symmetric positive definite matrix
     * @param V inverse matrix
     *
     * This function wraps the inversion of a matrix
     */
    void linalg_invert( ub::matrix<double> &A, ub::matrix<double> &V );
 
    /**
     * \brief determines Cholesky decomposition of matrix A
     * @param A symmetric positive definite matrix
     *
     * This function wraps the Cholesky decomposition
     */
    void linalg_cholesky_decompose( ub::matrix<double> &A );
    
    /**
     * \brief solves A*x=b
     * @param x storage for x
     * @param A symmetric positive definite matrix for linear system
     * @param b inhomogeniety
     * @param if A is not symmetric positive definite throws error code 
     *
     * This function wraps the cholesky linear system solver
     */
    void linalg_cholesky_solve(ub::vector<double> &x, ub::matrix<double> &A, ub::vector<double> &b);

    /**
     * \brief solves A*x=b
     * @param x storage for x
     * @param A matrix for linear equation system
     * @param b inhomogenity
     * @param residual if non-zero, residual will be stored here
     *
     * This function wrapps the qrsolver
     */
    void linalg_qrsolve(ub::vector<double> &x, ub::matrix<double> &A, ub::vector<double> &b, ub::vector<double> *residual=NULL);

    /**
     * \brief solves A*x=b under the constraint B*x = 0
     * @param x storage for x
     * @param A matrix for linear equation system
     * @param b inhomogenity
     * @param constr constrained condition B (or is it the transposed one? check that)
     *
     * This function wraps the qrsolver under constraints
     */
    void linalg_constrained_qrsolve(ub::vector<double> &x, ub::matrix<double> &A, ub::vector<double> &b, ub::matrix<double> &constr);

    /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x
     * @param A symmetric matrix 
     * @param E vector of eigenvalues
     * @param V matrix of eigenvalues
     * 
     * This function wraps gsl_eigen_symmv / DSYEV
     * note that the eigenvalues/eigenvectors are UNSORTED 
     * 
     */
    bool linalg_eigenvalues_symmetric( ub::symmetric_matrix<double> &A, ub::vector<double> &E, ub::matrix<double> &V );
    
   /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x
     * @param A matrix 
     * @param E vector of eigenvalues
     * @param V matrix of eigenvalues
     * 
     * This function wraps gsl_eigen_symmv / DSYEV
     * 
     */
    bool linalg_eigenvalues( ub::matrix<double> &A, ub::vector<double> &E, ub::matrix<double> &V );
    
    
   /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x
     * @param E vector of eigenvalues
     * @param V input: matrix to diagonalize
     * @param V output: eigenvectors      
     * 
     * This function wrapps gsl_eigen_symmv / DSYEV
     * 
     */
    bool linalg_eigenvalues( ub::vector<double> &E, ub::matrix<double> &V );
    
    
       /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x
     * @param E vector of eigenvalues
     * @param V input: matrix to diagonalize
     * @param V output: eigenvectors      
     * 
     * This function wrapps gsl_eigen_symmv / DSYEV
     * 
     */
    bool linalg_eigenvalues( ub::vector<float> &E, ub::matrix<float> &V );
    
   /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x
     * @param E vector of eigenvalues
     * @param V input: matrix to diagonalize
     * @param V output: eigenvectors      
     * 
     * This function wrapps gsl_eigen_symmv / DSYEV
     * 
     */
    bool linalg_eigenvalues( ub::matrix<double> &A, ub::vector<double> &E, ub::matrix<double> &V , int nmax );
    
      /**
     * \brief eigenvalues of a symmetric matrix A*x=E*x single precision
     * @param E vector of eigenvalues
     * @param V input: matrix to diagonalize
     * @param V output: eigenvectors      
     * 
     * This function wrapps gsl_eigen_symmv / DSYEV
     * 
     */
    bool linalg_eigenvalues( ub::matrix<float> &A, ub::vector<float> &E, ub::matrix<float> &V , int nmax );
    
     /**
     * \brief eigenvalues of a symmetric matrix A*x=E*B*x double precision
     * @param E vector of eigenvalues
     * @param A input: matrix to diagonalize
     * @param B input: overlap matrix
     * @param V output: eigenvectors      
     * 
     * This function wrapps eigen_gensymmv / dsygv
     * 
     */
    bool linalg_eigenvalues_general( ub::matrix<double> &A,ub::matrix<double> &B, ub::vector<double> &E, ub::matrix<double> &V);
    
    
    
}}



#endif	/* __VOTCA_TOOLS_LINALG_H */

