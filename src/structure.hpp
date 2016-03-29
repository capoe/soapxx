#ifndef _SOAP_STRUCTURE_HPP
#define _SOAP_STRUCTURE_HPP

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>
#include "types.hpp"
#include "boundary.hpp"

namespace soap {


class Segment;
class Structure;

class Particle
{
public:
    Particle(int id) 
        : _id(id), _pos(vec(0,0,0)), _name(""), _type_id(-1), _type(""), _mass(.0), _weight(.0), _sigma(.0),
		  _segment(NULL)
        { ; }
   ~Particle() {
	   ;
    }

    static void registerPython() {
        using namespace boost::python;
        class_<Particle, Particle*>("Particle", init<int>())
            .add_property("pos", &Particle::getPosNumeric, &Particle::setPosNumeric)
			.add_property("id", &Particle::getId, &Particle::setId)
            .add_property("name", make_function(&Particle::getName, copy_non_const()), &Particle::setName)
            .add_property("type", make_function(&Particle::getType, copy_non_const()), &Particle::setType)
            .add_property("type_id", make_function(&Particle::getTypeId, copy_non_const()), &Particle::setTypeId)
            .add_property("mass", make_function(&Particle::getMass, copy_non_const()), &Particle::setMass)
            .add_property("weight", make_function(&Particle::getWeight, copy_non_const()), &Particle::setWeight)
            .add_property("sigma", make_function(&Particle::getSigma, copy_non_const()), &Particle::setSigma);
    }

    void setPos(vec &pos) { _pos = pos; }
    void setPos(double x, double y, double z) { _pos = vec(x,y,z); }
    void setPosNumeric(const boost::python::numeric::array &pos) { _pos = vec(pos); }
    vec &getPos() { return _pos; }
    boost::python::numeric::array getPosNumeric() { boost::python::numeric::array pos(boost::python::make_tuple(_pos.x(), _pos.y(), _pos.z())); return pos; }

    void setName(std::string name) { _name = name; }
    std::string &getName() { return _name; }

    void setId(int id) { _id = id; }
    int getId() { return _id; }

    void setTypeId(int id) { _type_id = id; }
    int &getTypeId() { return _type_id; }

    void setType(std::string type) { _type = type; }
    std::string &getType() { return _type; }

    void setMass(double mass) { _mass = mass; }
    double &getMass() { return _mass; }

    void setWeight(double weight) { _weight = weight; }
    double &getWeight() { return _weight; }

    void setSigma(double sigma) { _sigma = sigma; }
    double &getSigma() { return _sigma; }
    
private:
    Segment *_segment;
    // Book-keeping
    int _id;
    std::string _name;
    int _type_id;
    std::string _type;
    // Observables
    vec _pos;
    double _mass;
    double _weight;
    double _sigma;
};


class Segment
{
public:
    Segment(int id) : _id(id) { ; }
   ~Segment() {
	   _particles.clear();
    }
    static void registerPython() {
           using namespace boost::python;
           class_<Segment>("Segment", init<int>())
               .def("addParticle", &Segment::addParticle, return_value_policy<reference_existing_object>());
    }
    Particle &addParticle(Particle *new_part) {
        _particles.push_back(new_part);
        return *new_part;
    }

private:
    int _id;
    std::vector<Particle*> _particles;
};


class Structure
{
public:
	typedef std::vector<Particle*>::iterator particle_it_t;

    Structure(std::string label) : _id(-1), _label(label), _box(NULL) {
    	matrix box;
    	box.ZeroMatrix();
    	this->setBoundary(box);
    }
   ~Structure() {
        delete _box;
        _box = NULL;
        std::vector<Segment*>::iterator sit;
        for (sit = _segments.begin(); sit != _segments.end(); ++sit) {
        	delete *sit;
        }
        _segments.clear();
        std::vector<Particle*>::iterator pit;
        for (pit = _particles.begin(); pit != _particles.end(); ++pit) {
        	delete *pit;
        }
        _particles.clear();
    }
    static void registerPython() {
           using namespace boost::python;
           class_<Structure>("Structure", init<std::string>())
               .def("addSegment", &Structure::addSegment, return_value_policy<reference_existing_object>())
               .def("addParticle", &Structure::addParticle, return_value_policy<reference_existing_object>())
			   .def("__iter__", range<return_value_policy<reference_existing_object> >(&Structure::beginParticles, &Structure::endParticles))
			   .add_property("particles", range<return_value_policy<reference_existing_object> >(&Structure::beginParticles, &Structure::endParticles))
   			   .def("connect", &Structure::connectNumeric)
               .add_property("box", &Structure::getBoundaryNumeric, &Structure::setBoundaryNumeric)
			   .add_property("label", make_function(&Structure::getLabel, copy_non_const()), &Structure::setLabel);
           class_< std::vector<Particle*> >("ParticleContainer")
        	   .def(vector_indexing_suite<std::vector<Particle*> >());
    }
    std::vector<Particle*>::iterator beginParticles() { return _particles.begin(); }
    std::vector<Particle*>::iterator endParticles() { return _particles.end(); }
    std::string &getLabel() { return _label; }
    void setLabel(std::string label) { _label = label; }
    // PARTICLE CREATION & INTERFACE
    Segment &addSegment() {
        int id = _segments.size()+1;
        Segment *new_seg = new Segment(id);
        _segments.push_back(new_seg);
        return *new_seg;
    }
    Particle &addParticle(Segment &seg) {
        int id = _particles.size()+1;
        Particle *new_part = new Particle(id);
        _particles.push_back(new_part);
        seg.addParticle(new_part);
        return *new_part;
    }
    // BOUNDARY CREATION & INTERFACE
    vec connect(const vec &r1, const vec &r2) {
    	return _box->connect(r1, r2); // Points from r1 to r2
    }
    boost::python::numeric::array connectNumeric(
        const boost::python::numeric::array &a1,
		const boost::python::numeric::array &a2) {
    	vec r1(a1);
    	vec r2(a2);
    	vec dr = this->connect(r1, r2);
		return boost::python::numeric::array(boost::python::make_tuple(
			dr.x(), dr.y(), dr.z()));
	}
    void setBoundary(const matrix &box) {
    	delete _box;
		if(box.get(0,0)==0 && box.get(0,1)==0 && box.get(0,2)==0 &&
		   box.get(1,0)==0 && box.get(1,1)==0 && box.get(1,2)==0 &&
		   box.get(2,0)==0 && box.get(2,1)==0 && box.get(2,2)==0) {
				_box = new BoundaryOpen(box);
		}
		else if(box.get(0,1)==0 && box.get(0,2)==0 &&
				box.get(1,0)==0 && box.get(1,2)==0 &&
				box.get(2,0)==0 && box.get(2,1)==0) {
				_box = new BoundaryOrthorhombic(box);
		}
		else {
			_box = new BoundaryTriclinic(box);
		}
		return;
	}
    void setBoundaryNumeric(const boost::python::numeric::array &m) {
    	matrix box(m);
    	this->setBoundary(box);
    }
    boost::python::numeric::array getBoundaryNumeric() {
    	matrix box = _box->getBox();
    	boost::python::numeric::array box_np(boost::python::make_tuple(
            box.get(0,0), box.get(0,1), box.get(0,2),
			box.get(1,0), box.get(1,1), box.get(1,2),
			box.get(2,0), box.get(2,1), box.get(2,2)));
    	box_np.resize(3,3);
    	return box_np;
    }

private:
    int _id;
    std::string _label;
    std::vector<Segment*> _segments;
    std::vector<Particle*> _particles;

    Boundary *_box;
};


} // soap namespace

/*
#include <boost/python.hpp>

BOOST_PYTHON_MODULE(_soapxx)
{
    using namespace boost::python;
    soap::Particle::registerPython();
    soap::Segment::registerPython();
    soap::Structure::registerPython();
}
*/

#endif
