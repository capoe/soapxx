#include <fstream>
#include <boost/format.hpp>

#include "soap/mol2d.hpp"
#include "soap/base/exceptions.hpp"

namespace soap {

Mol2D::Mol2D(Structure &structure) :
    _structure(&structure) {
	GLOG() << "Configuring mol2d ..." << std::endl;
}

Mol2D::~Mol2D() {
}

double Mol2D::computeFreeVolumeFraction(
        boost::python::numeric::array &np_centre, 
        double probe_radius, 
        double res) {
    bool verbose = false;
    vec centre(np_centre);
   
    double min_x = centre.getX()-probe_radius;
    double max_x = centre.getX()+probe_radius;
    double min_y = centre.getY()-probe_radius;
    double max_y = centre.getY()+probe_radius;
    double min_z = centre.getZ()-probe_radius;
    double max_z = centre.getZ()+probe_radius;
    
    int nx = int((max_x-min_x)/res)+1;
    int ny = int((max_y-min_y)/res)+1;
    int nz = int((max_z-min_z)/res)+1;
    GLOG() << nx << " x " << ny << " x " << nz << std::endl;
    vec r0 = vec(min_x, min_y, min_z);

    int n_sph = 0;
    int n_exc = 0;
    int n_tot = 0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			for (int k = 0; k < nz; ++k) {
                n_tot += 1;
                bool excluded = false;
				vec dr(i*res, j*res, k*res);
                vec ri = r0 + dr;
                if (soap::linalg::abs(ri-centre) > probe_radius) {
                    if (verbose) GLOG() << "N " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                    continue;
                }
                for (auto pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
                     vec rij = _structure->connect(ri, (*pit)->getPos());
                     double dij = soap::linalg::abs(rij);
                     if (dij <= (*pit)->getSigma()) {
                        excluded = true;
                        break;
                     } else ;
                }
                if (excluded) {
                    n_exc += 1;
                    if (verbose) GLOG() << "O " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                }
                else {
                    if (verbose) GLOG() << "C " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                }
                n_sph += 1;
                
    }}}

    double f_sph = double(n_sph)/n_tot;
    double f_free = 1. - double(n_exc)/n_sph;
    GLOG() << "f_sph  = " << f_sph << std::endl;
    GLOG() << "f_free = " << f_free << std::endl;
    return f_free;
}

double Mol2D::computeTPSA(double res) {

    auto center = _structure->particles()[0];
    bool verbose = false;

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

    if (verbose) {
        GLOG() << "x-range: " << min_x << "  " << max_x << std::endl;
        GLOG() << "y-range: " << min_y << "  " << max_y << std::endl;
        GLOG() << "z-range: " << min_z << "  " << max_z << std::endl;
    }

    double dx = res;
    double dy = res;
    double dz = res;
    int nx = int((max_x-min_x+4)/dx)+1;
    int ny = int((max_y-min_y+4)/dy)+1;
    int nz = int((max_z-min_z+4)/dz)+1;
    GLOG() << nx << " x " << ny << " x " << nz << std::endl;
    vec r0 = center->getPos() + vec(min_x-2, min_y-2, min_z-2);

    double total_weight = 0.0;
    int n_surface = 0;
    int n_off_surface = 0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j <= ny; ++j) {
			for (int k = 0; k <= nz; ++k) {
				vec dr(i*dx, j*dy, k*dz);
                vec ri = r0 + dr;

                bool on_surface = false;
                bool outside = true;
                double weight = 0.0;
                for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
                     vec rij = _structure->connect(ri, (*pit)->getPos());
                     double dij = soap::linalg::abs(rij);
                     double s = dij - (*pit)->getSigma();
                     if (s*s <= dx*dx) {
                        on_surface = true;
                        weight = (*pit)->getWeight();
                     }
                     else if (s < 0) {
                        outside = false;
                     }
                }
                if (on_surface && outside && std::abs(weight) > 1e-10) {
                    n_surface += 1;
                    total_weight += weight;
                    if (verbose && weight > 1e-10) GLOG() << "O " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                    else if (verbose && weight < -1e-10) GLOG() << "N " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                }
                else if (on_surface && outside) {
                    if (verbose) GLOG() << "C " << ri.getX() << " " << ri.getY() << " " << ri.getZ() << std::endl;
                    n_surface += 1;
                } else n_off_surface += 1;
    }}}

    double A = 0.5*total_weight*dx*dy;
    return A;
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

    if (debug) {
        GLOG() << "x-range: " << min_x << "  " << max_x << std::endl;
        GLOG() << "y-range: " << min_y << "  " << max_y << std::endl;
        GLOG() << "z-range: " << min_z << "  " << max_z << std::endl;
    }

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
        .def("computeFreeVolumeFraction", &Mol2D::computeFreeVolumeFraction)
        .def("computeTPSA", &Mol2D::computeTPSA)
		.def("computeVolume", &Mol2D::computeVolume)
		.def("computeSurface", &Mol2D::computeSurface);
}

}
