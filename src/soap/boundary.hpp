#ifndef _SOAP_BOUNDARY_HPP
#define	_SOAP_BOUNDARY_HPP

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "soap/types.hpp"

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
    virtual vec connect(const vec &r_i, const vec &r_j) {
    	return r_j - r_i;
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
		vec r_ij;
		double a = _box.get(0,0); double b = _box.get(1,1); double c = _box.get(2,2);
		r_ij = r_j - r_i;
		r_ij.setZ( r_ij.getZ() - c*round(r_ij.getZ()/c) );
		r_ij.setY( r_ij.getY() - b*round(r_ij.getY()/b) );
		r_ij.setX( r_ij.getX() - a*round(r_ij.getX()/a) );
		return r_ij;
	}

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
	}
	BoundaryTriclinic() {
		_type = Boundary::typeTriclinic;
		_box.UnitMatrix();
	}
	vec connect(const vec &r_i, const vec &r_j) const {
	    vec r_tp, r_dp, r_sp, r_ij;
	    vec a = _box.getCol(0); vec b = _box.getCol(1); vec c = _box.getCol(2);
	    r_tp = r_j - r_i;
	    r_dp = r_tp - c*round(r_tp.getZ()/c.getZ());
	    r_sp = r_dp - b*round(r_dp.getY()/b.getY());
	    r_ij = r_sp - a*round(r_sp.getX()/a.getX());
	    return r_ij;
	}

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & boost::serialization::base_object<Boundary>(*this);
	}
};

}

BOOST_CLASS_EXPORT_KEY(soap::Boundary);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryOpen);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryOrthorhombic);
BOOST_CLASS_EXPORT_KEY(soap::BoundaryTriclinic);

#endif
