#ifndef _SOAP_BOUNDARY_HPP
#define	_SOAP_BOUNDARY_HPP

#include "linalg/matrix.hpp"

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
    virtual vec connect(const vec &r_i, const vec &r_j) const = 0;

    virtual eBoxType getBoxType() { return _type; }

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
    vec connect(const vec &r_i, const vec &r_j) const {
    	return r_j - r_i;
    }
};

class BoundaryOrthorhombic : public Boundary
{
public:
	BoundaryOrthorhombic(const matrix &box) {
		 _type = Boundary::typeOrthorhombic;
		 _box = box;
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
};

class BoundaryTriclinic : public Boundary
{
public:
	BoundaryTriclinic(const matrix &box) {
		_type = Boundary::typeTriclinic;
		_box = box;
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
};

}

#endif

