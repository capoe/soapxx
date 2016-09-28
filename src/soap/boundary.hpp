#ifndef _SOAP_BOUNDARY_HPP
#define	_SOAP_BOUNDARY_HPP

#include <cmath>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "soap/types.hpp"
#include "soap/base/exceptions.hpp"

namespace soap {

class BoundaryOpen;
class BoundaryOrthorhombic;
class BoundaryTriclinic;

class Boundary
{
public:

	enum eBoxType {
		typeOpen = 0, typeOrthorhombic, typeTriclinic
	};

	Boundary() {
		_type = Boundary::typeOpen;
		_box.ZeroMatrix();
	}
    virtual ~Boundary() {;}

    void setBox(const matrix &box) {
        _box = box;
    }

    const matrix &getBox() { return _box; };
    virtual double BoxVolume() {
		vec a = _box.getCol(0);
		vec b = _box.getCol(1);
		vec c = _box.getCol(2);
		return (a^b)*c;
    }
    virtual vec connect(const vec &r_i, const vec &r_j) const {
        std::cout << "connect default" << std::endl;
    	return r_j - r_i;
    }

    virtual std::vector<int> calculateRepetitions(double cutoff) {
        std::vector<int> na_nb_nc = { 0, 0, 0 };
        return na_nb_nc;
    }

    virtual eBoxType getBoxType() { return _type; }

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _box;
    	arch & _type;
    }

protected:
    matrix _box;
    eBoxType _type;
};


class BoundaryOpen : public Boundary
{
public:
	BoundaryOpen(const matrix &box) {
		 _type = Boundary::typeOpen;
		 _box = box;
	}
	BoundaryOpen() {
		_type = Boundary::typeOpen;
		_box.ZeroMatrix();
	}
    vec connect(const vec &r_i, const vec &r_j) const {
    	return r_j - r_i;
    }
    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & boost::serialization::base_object<Boundary>(*this);
	}
};

class BoundaryOrthorhombic : public Boundary
{
public:
	BoundaryOrthorhombic(const matrix &box) {
		 _type = Boundary::typeOrthorhombic;
		 _box = box;
	}
	BoundaryOrthorhombic() {
		_type = Boundary::typeOrthorhombic;
		_box.UnitMatrix();
	}
	vec connect(const vec &r_i, const vec &r_j) const {
	    std::cout << "Connect ortho" << std::endl;
		vec r_ij;
		double a = _box.get(0,0); double b = _box.get(1,1); double c = _box.get(2,2);
		r_ij = r_j - r_i;
		r_ij.setZ( r_ij.getZ() - c*round(r_ij.getZ()/c) );
		r_ij.setY( r_ij.getY() - b*round(r_ij.getY()/b) );
		r_ij.setX( r_ij.getX() - a*round(r_ij.getX()/a) );
		return r_ij;
	}

	virtual std::vector<int> calculateRepetitions(double cutoff);

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & boost::serialization::base_object<Boundary>(*this);
	}
};

class BoundaryTriclinic : public Boundary
{
public:
	BoundaryTriclinic(const matrix &box) {
		_type = Boundary::typeTriclinic;
		_box = box;

        // Set-up inverse box
	    vec a = _box.getCol(0); 
        vec b = _box.getCol(1); 
        vec c = _box.getCol(2);
        double V = this->BoxVolume();
        vec a_inv = b ^ c / V;
        vec b_inv = c ^ a / V;
        vec c_inv = a ^ b / V;
        _inv_box = matrix(a_inv, b_inv, c_inv);
	}
	BoundaryTriclinic() {
		_type = Boundary::typeTriclinic;
		_box.UnitMatrix();
	}
	virtual vec connect(const vec &r_i, const vec &r_j) const;

	virtual std::vector<int> calculateRepetitions(double cutoff);

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & boost::serialization::base_object<Boundary>(*this);
	}
private:
    matrix _inv_box;
};

}

BOOST_CLASS_EXPORT_KEY(soap::Boundary);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryOpen);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryOrthorhombic);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryTriclinic);

#endif
