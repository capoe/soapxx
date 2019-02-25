#ifndef _SOAP_LINALG_KERNEL_H
#define	_SOAP_LINALG_KERNEL_H

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"

namespace soap { namespace linalg {

namespace ub = boost::numeric::ublas;


//double kernel_rematch(ub::matrix<double> K, double gamma, double eps) {
//    int nx = K.size1();
//    int ny = K.size2();
//    ub::vector<double> u(nx, 1.0);
//    ub::vector<double> ou(nx);
//    ub::vector<double> v(ny, 1.0);
//    double ax = 1./nx;
//    double ay = 1./ny;
//    double lambda = 1./gamma;
//    double terr = eps*eps;
//    double derr = 0.0;
//    ub::matrix<double> Kg(nx,ny);
//    //for (int i=0; i<nx; ++i) u(i) = 1.0;
//    //for (int j=0; j<ny; ++j) v(j) = 1.0;
//    for (int i=0; i<nx; ++i) {
//        for (int j=0; j<ny; ++j) {
//            Kg(i,j) = std::exp(-(1-K(i,j))*lambda);
//        }
//    }
//    do {
//        ou = u;
//
//        for (int i=0; i<nx; ++i) u(i) = 0.0;
//        soap::linalg::linalg_matrix_vector_dot(Kg, v, u, false);
//        derr = 0.0;
//        for (int i=0; i<nx; ++i) derr+=(ax-ou(i)*u(i))*(ax-ou(i)*u(i));        
//        for (int i=0; i<nx; ++i) u(i)=ax/u(i);
//
//        for (int i=0; i<ny; ++i) v(i) = 0.0;
//        soap::linalg::linalg_matrix_vector_dot(Kg, u, v, true);
//        for (int i=0; i<ny; ++i) v(i)=ay/v(i);
//    } while (derr>terr);
//
//    double k = 0.0;
//    ub::matrix<double> Pij(nx, ny);
//    for (int i=0; i<nx; ++i) {
//        for (int j=0; j<ny; ++j) {
//            Pij(i,j) = u(i)*Kg(i,j)*v(j);
//            k += Pij(i,j)*K(i,j);
//        }
//    }
//
//    return k;
//}

boost::python::object kernel_rematch_atomic(
        boost::python::object &kmat_npy, double gamma, double eps) {
    // CONVERT NUMPY KERNEL MATRIX TO UBLAS
    soap::linalg::numpy_converter npc("float64");
    ub::matrix<double> kmat;
    npc.numpy_to_ublas<double>(kmat_npy, kmat);
    // INITIALIZE OPTIMIZATION
    int nx = kmat.size1();
    int ny = kmat.size2();
    std::vector<double> u(nx), ou(nx), v(ny);
    double ax = 1.0/nx, ay=1.0/ny;
    ub::matrix<double> Kg(nx,ny);
    for (int i=0; i<nx; ++i) u[i]=1.0;
    for (int i=0; i<ny; ++i) v[i]=1.0;
    double lambda=1.0/gamma, terr=eps*eps, derr;
    for (int i=0; i<nx; ++i) 
        for (int j=0; j<ny; ++j) 
            Kg(i,j)=std::exp(-(1-kmat(i,j))*lambda);
    // OPTIMIZE 
    do {
        // u<-1.0/Kg.v
        for (int i=0; i<nx; ++i) { ou[i]=u[i]; u[i]=0.0; }
        for (int i=0; i<nx; ++i) for (int j=0; j<ny; ++j) u[i]+=Kg(i,j)*v[j];
        // at this point we can compute how far off unity we are
        derr = 0.0;
        for (int i=0; i<nx; ++i) derr+=(ax-ou[i]*u[i])*(ax-ou[i]*u[i]);        
        for (int i=0; i<nx; ++i) u[i]=ax/u[i];
        // v<-1.0/Kg.u
        for (int i=0; i<ny; ++i) v[i]=0.0; 
        for (int i=0; i<ny; ++i) for (int j=0; j<nx; ++j) v[i]+=Kg(j,i)*u[j];
        for (int i=0; i<ny; ++i) v[i]=ay/v[i];
        //std::cerr<<derr<<"\n";
    } while (derr>terr);
    /* 
    // COMPUTE KERNEL VALUE
    double rval=0, rrow; 
    for (int i=0; i<nx; ++i) 
    {
       std::cout << "Row " << i << std::endl;
       rrow=0;
       for (int j=0; j<ny; ++j) {
          rrow+=Kg(i,j)*kmat(i,j)*v[j];
          std::cout << Kg(i,j) << "   " << kmat(i,j) << "   " << v[j] << " u*K*v " << u[i]*Kg(i,j)*v[j] << std::endl;
       }
       rval+=u[i]*rrow;
    }   
    */
    // RETURN REGULARIZED PERMUTATION MATRIX
    ub::matrix<double> Pij(nx, ny);
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            Pij(i,j) = u[i]*Kg(i,j)*v[j];
        }
    }
    return npc.ublas_to_numpy<double>(Pij);
};

class KernelModule {
public:
    KernelModule();
    static void registerPython() {
        using namespace boost::python;
        def("kernel_rematch_atomic", kernel_rematch_atomic);
    }
private:
};

}}

#endif	

