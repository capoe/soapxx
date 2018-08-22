#ifndef _SOAP_MOL2D_HPP
#define _SOAP_MOL2D_HPP

#include <assert.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/base_object.hpp>

#include "soap/base/logger.hpp"
#include "soap/types.hpp"
#include "soap/globals.hpp"
#include "soap/structure.hpp"


namespace soap {


class Mol2D
{
public:
	Mol2D(Structure &structure);
   ~Mol2D();
	double computeVolume(double res);
    double computeSurface(double res);
    double computeTPSA(double res);
    double computeFreeVolumeFraction(boost::python::numeric::array &centre, double probe_radius, double res);
	static void registerPython();
private:
    Structure *_structure;
};

}

#endif /* _SOAP_MOL2D_HPP_ */
