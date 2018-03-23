#include <fstream>
#include <boost/format.hpp>

#include "soap/mol2d.hpp"

namespace soap {

Mol2D::Mol2D(Structure &structure) :
    _structure(&structure) {
	GLOG() << "Configuring mol2d ..." << std::endl;
}

Mol2D::~Mol2D() {
}

double Mol2D::computeSurface(double res) {

    auto center = _structure->particles()[0];
    bool debug = false;

    double min_x = 0.0;
    double max_x = 0.0;
    double min_y = 0.0;
    double max_y = 0.0;
    double min_z = 0.0;
    double max_z = 0.0;

	Structure::particle_it_t pit;
	for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
		 vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
         if (dr.getX() < min_x) min_x = dr.getX();
         if (dr.getX() > max_x) max_x = dr.getX();
         if (dr.getY() < min_y) min_y = dr.getY();
         if (dr.getY() > max_y) max_y = dr.getY();
         if (dr.getZ() < min_z) min_z = dr.getZ();
         if (dr.getZ() > max_z) max_z = dr.getZ();
	}
    GLOG() << "x-range: " << min_x << "  " << max_x << std::endl;
    GLOG() << "y-range: " << min_y << "  " << max_y << std::endl;
    GLOG() << "z-range: " << min_z << "  " << max_z << std::endl;

    double dx = res;
    double dy = res;
    double dz = res;
    int nx = int((max_x-min_x+4)/dx)+1;
    int ny = int((max_y-min_y+4)/dy)+1;
    int nz = int((max_z-min_z+4)/dz)+1;
    GLOG() << nx << " x " << ny << " x " << nz << std::endl;
    vec r0 = center->getPos() + vec(min_x-2, min_y-2, min_z-2);

    int n_surface = 0;
    int n_off_surface = 0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j <= ny; ++j) {
			for (int k = 0; k <= nz; ++k) {
				vec dr(i*dx, j*dy, k*dz);
                vec ri = r0 + dr;

                bool on_surface = false;
                bool outside = true;
                for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
                     vec rij = _structure->connect(ri, (*pit)->getPos());
                     double dij = soap::linalg::abs(rij);
                     double s = dij - (*pit)->getSigma();
                     if (s*s <= dx*dx) {
                        on_surface = true;
                     }
                     else if (s < 0) {
                        outside = false;
                     }
                }
                if (on_surface && outside) {
                    if (debug) GLOG() << "C " << i*dx << " " << j*dy << " " << k*dz << std::endl;
                    n_surface += 1;
                } else n_off_surface += 1;
    }}}

    double A = 0.5*n_surface*dx*dy;
    return A;
}

double Mol2D::computeVolume(double res) {

    double min_x = 0.0;
    double max_x = 0.0;
    double min_y = 0.0;
    double max_y = 0.0;
    double min_z = 0.0;
    double max_z = 0.0;

    auto center = _structure->particles()[0];
    bool debug = false;

	Structure::particle_it_t pit;
	for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
		 vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
         if (dr.getX() < min_x) min_x = dr.getX();
         if (dr.getX() > max_x) max_x = dr.getX();
         if (dr.getY() < min_y) min_y = dr.getY();
         if (dr.getY() > max_y) max_y = dr.getY();
         if (dr.getZ() < min_z) min_z = dr.getZ();
         if (dr.getZ() > max_z) max_z = dr.getZ();
	}
    GLOG() << "x-range: " << min_x << "  " << max_x << std::endl;
    GLOG() << "y-range: " << min_y << "  " << max_y << std::endl;
    GLOG() << "z-range: " << min_z << "  " << max_z << std::endl;

    double dx = res;
    double dy = res;
    double dz = res;
    int nx = int((max_x-min_x+4)/dx)+1;
    int ny = int((max_y-min_y+4)/dy)+1;
    int nz = int((max_z-min_z+4)/dz)+1;
    GLOG() << nx << " x " << ny << " x " << nz << std::endl;
    vec r0 = center->getPos() + vec(min_x-2, min_y-2, min_z-2);

    int n_out = 0;
    int n_in = 0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j <= ny; ++j) {
			for (int k = 0; k <= nz; ++k) {
				vec dr(i*dx, j*dy, k*dz);
                vec ri = r0 + dr;

                bool outside = true;
                for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
                     vec rij = _structure->connect(ri, (*pit)->getPos());
                     double dij = soap::linalg::abs(rij);
                     if (dij <= (*pit)->getSigma()) {
                        outside = false;
                        break;
                     } else ;
                }
                if (outside) n_out += 1;
                else {
                    n_in += 1;
                    if (debug) GLOG() << "C " << i*dx << " " << j*dy << " " << k*dz << std::endl;
                }
    }}}

    double V = n_in*dx*dy*dz;
    return V;
}

void Mol2D::registerPython() {
    using namespace boost::python;

    class_<Mol2D>("Mol2D", init<Structure &>())
		.def("computeVolume", &Mol2D::computeVolume)
		.def("computeSurface", &Mol2D::computeSurface);
}

}
