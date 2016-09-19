#include "soap/boundary.hpp"
#include "soap/globals.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace soap {


soap::vec BoundaryTriclinic::connect(const vec &r_i, const vec &r_j) const {
    /*
    // This only works if a = (*,0,0), b = (*,*,0), c = (*,*,*) => e.g., GROMACS
    vec r_tp, r_dp, r_sp, r_ij;
    vec a = _box.getCol(0); vec b = _box.getCol(1); vec c = _box.getCol(2);
    r_tp = r_j - r_i;
    r_dp = r_tp - c*round(r_tp.getZ()/c.getZ());
    r_sp = r_dp - b*round(r_dp.getY()/b.getY());
    r_ij = r_sp - a*round(r_sp.getX()/a.getX());
    return r_ij;
    */

    vec a = _box.getCol(0);
    vec b = _box.getCol(1);
    vec c = _box.getCol(2);
    //GLOG() << "Box vectors: " << a << " " << b << " " << c << std::endl;

    vec u = _inv_box.getCol(0);
    vec v = _inv_box.getCol(1);
    vec w = _inv_box.getCol(2);
    //GLOG() << "Inverse box vectors: " << u << " " << v << " " << w << std::endl;

    vec dr = r_j - r_i;
    //GLOG() << "dr " << dr << std::endl;
    //GLOG() << "u " << std::floor(u*dr) << std::endl;
    dr = dr - std::floor(u*dr)*a;
    //GLOG() << "v " << std::floor(v*dr) << std::endl;
    dr = dr - std::floor(v*dr)*b;
    //GLOG() << "w " << std::floor(w*dr) << std::endl;
    dr = dr - std::floor(w*dr)*c;
    //GLOG() << "in first quadrant " << dr << std::endl;

    vec dr_min = dr;
    double d_min = soap::linalg::abs(dr);

    for (int i=0; i < 2; ++i) {
    for (int j=0; j < 2; ++j) {
    for (int k=0; k < 2; ++k) {
        vec dr_ijk = dr - i*a - j*b - k*c;
        double d_ijk = soap::linalg::abs(dr_ijk);
        //GLOG() << "ijk " << i << " " << j << " " << k << " " << dr << " " << d_ijk << " " << dr_ijk << std::endl;
        if (d_ijk < d_min) {
            d_min = d_ijk;
            dr_min = dr_ijk;
        }
    }}}

    return dr_min;
}

std::vector<int> BoundaryTriclinic::calculateRepetitions(double cutoff) {

    vec a = _box.getCol(0);
    vec b = _box.getCol(1);
    vec c = _box.getCol(2);

    vec u = _inv_box.getCol(0);
    vec v = _inv_box.getCol(1);
    vec w = _inv_box.getCol(2);

    //std::cout << a << std::endl;
    //std::cout << b << std::endl;
    //std::cout << c << std::endl;

    //std::cout << u << std::endl;
    //std::cout << v << std::endl;
    //std::cout << w << std::endl;

    double da = std::abs(u*a/soap::linalg::abs(u));
    double db = std::abs(v*b/soap::linalg::abs(v));
    double dc = std::abs(w*c/soap::linalg::abs(w));

    //std::cout << da << " " << db << " " << dc << std::endl;

    int na = int(1+(cutoff-0.5*da)/da);
    int nb = int(1+(cutoff-0.5*db)/db);
    int nc = int(1+(cutoff-0.5*dc)/dc);

    std::vector<int> na_nb_nc = { na, nb, nc };
    return na_nb_nc;
}

std::vector<int> BoundaryOrthorhombic::calculateRepetitions(double cutoff) {
    vec a = _box.getCol(0);
    vec b = _box.getCol(1);
    vec c = _box.getCol(2);

    double da = soap::linalg::abs(a);
    double db = soap::linalg::abs(b);
    double dc = soap::linalg::abs(c);

    int na = int(1+(cutoff-0.5*da)/da);
    int nb = int(1+(cutoff-0.5*db)/db);
    int nc = int(1+(cutoff-0.5*dc)/dc);

    std::vector<int> na_nb_nc = { na, nb, nc };
    return na_nb_nc;
}


}


BOOST_CLASS_EXPORT_IMPLEMENT(soap::BoundaryOpen);
BOOST_CLASS_EXPORT_IMPLEMENT(soap::BoundaryOrthorhombic);
BOOST_CLASS_EXPORT_IMPLEMENT(soap::BoundaryTriclinic);
