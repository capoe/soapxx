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
    Particle(int id) { this->null(); _id = id; }
    Particle() { this->null(); }
   ~Particle() {;}
    void null();
    // Position
    void setPos(vec &pos) { _pos = pos; }
    void setPos(double x, double y, double z) { _pos = vec(x,y,z); }
    vec &getPos() { return _pos; }
    void setPosNumeric(const boost::python::numeric::array &pos) { _pos = vec(pos); }
    boost::python::numeric::array getPosNumeric();
    // Name
    void setName(std::string name) { _name = name; }
    std::string &getName() { return _name; }
    // Id
    void setId(int id) { _id = id; }
    int getId() { return _id; }
    // Type Id
    void setTypeId(int id) { _type_id = id; }
    int &getTypeId() { return _type_id; }
    // Type
    void setType(std::string type) { _type = type; }
    std::string &getType() { return _type; }
    // Mass
    void setMass(double mass) { _mass = mass; }
    double &getMass() { return _mass; }
    // Weight
    void setWeight(double weight) { _weight = weight; }
    double &getWeight() { return _weight; }
    // Sigma
    void setSigma(double sigma) { _sigma = sigma; }
    double &getSigma() { return _sigma; }
    
    static void registerPython();

    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
    	arch & _segment;
    	arch & _id;
    	arch & _name;
    	arch & _type_id;
    	arch & _type;
    	arch & _pos;
    	arch & _mass;
    	arch & _weight;
    	arch & _sigma;
		return;
	}

private:
    Segment *_segment;
    // Labels
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
	typedef std::vector<Particle*> particle_array_t;

    Segment(int id) : _id(id) { ; }
    Segment() : _id(-1) { ; }
   ~Segment() { _particles.clear(); }

    Particle &addParticle(Particle *new_part);

    static void registerPython();

    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
    	arch & _id;
    	arch & _particles;
		return;
	}

private:
    int _id;
    particle_array_t _particles;
};


class Structure
{
public:
	typedef std::vector<Particle*> particle_array_t;
	typedef std::vector<Particle*>::iterator particle_it_t;
	typedef std::vector<Segment*> segment_array_t;
	typedef std::vector<Segment*>::iterator segment_it_t;

    Structure(std::string label);
    Structure();
   ~Structure();
    void null();

    particle_array_t &particles() { return _particles; }
    particle_it_t beginParticles() { return _particles.begin(); }
    particle_it_t endParticles() { return _particles.end(); }

    std::string &getLabel() { return _label; }
    void setLabel(std::string label) { _label = label; }

    // PARTICLE CREATION & INTERFACE
    Segment &addSegment();
    Particle &addParticle(Segment &seg);

    // BOUNDARY CREATION & INTERFACE
    void setBoundary(const matrix &box);
    vec connect(const vec &r1, const vec &r2) { return _box->connect(r1, r2); /* 1->2 */ }
    void setBoundaryNumeric(const boost::python::numeric::array &m);
    boost::python::numeric::array getBoundaryNumeric();
	boost::python::numeric::array connectNumeric(
		const boost::python::numeric::array &a1,
		const boost::python::numeric::array &a2);

    static void registerPython();

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _id;
    	arch & _label;
    	arch & _segments;
    	arch & _particles;
    	arch & _box;
    	return;
    }

private:
    int _id;
    std::string _label;
    segment_array_t _segments;
    particle_array_t _particles;

    Particle* _center;
    bool _has_center;

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
