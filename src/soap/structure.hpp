#ifndef _SOAP_STRUCTURE_HPP
#define _SOAP_STRUCTURE_HPP

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>

#include "soap/base/exceptions.hpp"
#include "soap/types.hpp"
#include "soap/boundary.hpp"

namespace soap {


class Segment;
class Structure;

struct Multitype
{
    typedef std::map<std::string, double> typemap_t;
    typedef std::map<std::string, double>::iterator typemap_it_t;
    Multitype() {};
   ~Multitype() {};
    void clear() { _typemap.clear(); _typestr = ""; }
    // Single-type interface
    void set(std::string type) { this->clear(); _typemap[type] = 1.0; _typestr = type; }
    std::string &getString() { return _typestr; }
    // Multi-type interface
    void add(std::string type, double weight) { _typemap[type] = weight; _typestr += type; }
    double getWeight(std::string type) { return _typemap[type]; }
    int size() { return _typemap.size(); }
    typemap_it_t begin() { return _typemap.begin(); }
    typemap_it_t end() { return _typemap.end(); }
    typemap_t _typemap;
    std::string _typestr;
    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
    	arch & _typemap;
		return;
	}
};

class Particle
{
public:
    explicit Particle(int id) { this->null(); _id = id; }
    Particle() { this->null(); }
   ~Particle() {;}
    void null();
    void model(Particle &model);
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
    void clearType() { _mtype.clear(); }
    void setType(std::string type) { _mtype.set(type); }
    void addType(std::string t, double w) { _mtype.add(t, w); }
    std::string &getType() { return _mtype.getString(); }
    double getTypeWeight(std::string t) { return _mtype.getWeight(t); }
    Multitype &getMultitype() { return _mtype; }
    // Mass
    void setMass(double mass) { _mass = mass; }
    double &getMass() { return _mass; }
    // Weight
    void setWeight(double weight) { _weight = weight; }
    double &getWeight() { return _weight; }
    // Sigma
    void setSigma(double sigma) { _sigma = sigma; }
    double &getSigma() { return _sigma; }
    // Segment
    Segment *getSegment() { return _segment; }

    static void registerPython();

    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
    	arch & _segment;
    	arch & _id;
    	arch & _name;
    	arch & _type_id;
        arch & _mtype;
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
    Multitype _mtype;
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
	typedef particle_array_t::iterator particle_it_t;

    explicit Segment(int id) : _id(id), _name("?"), _type("?") { ; }
    Segment() : _id(-1), _name("?"), _type("?") { ; }
   ~Segment() { _particles.clear(); }

    // PARTICLE CONTAINER
    Particle &addParticle(Particle *new_part);
    particle_array_t &particles() { return _particles; }
    particle_it_t beginParticles() { return _particles.begin(); }
    particle_it_t endParticles() { return _particles.end(); }

    // Name
    void setName(std::string name) { _name = name; }
    std::string &getName() { return _name; }
    // Id
    void setId(int id) { _id = id; }
    int getId() { return _id; }
    // Type
    void setType(std::string type) { _type = type; }
    std::string &getType() { return _type; }

    static void registerPython();

    template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
    	arch & _id;
    	arch & _name;
    	arch & _type;
    	arch & _particles;
		return;
	}

private:
    int _id;
    std::string _name;
    std::string _type;
    particle_array_t _particles;
};


class Structure
{
public:
	typedef std::vector<Particle*> particle_array_t;
	typedef std::vector<Particle*>::iterator particle_it_t;
	typedef std::vector<Segment*> segment_array_t;
	typedef std::vector<Segment*>::iterator segment_it_t;
    typedef double dtype_t;
    typedef boost::numeric::ublas::matrix<dtype_t> laplace_t;

    explicit Structure(std::string label);
    Structure(const Structure &structure);
    Structure();
   ~Structure();
    void null();
    void model(Structure &structure);

    // PARTICLE CONTAINER
    particle_array_t &particles() { return _particles; }
    particle_it_t beginParticles() { return _particles.begin(); }
    particle_it_t endParticles() { return _particles.end(); }
    int getNumberOfParticles() { return _particles.size(); }
    Particle *getParticle(int pid) { return _particles[pid-1]; }

    // SEGMENT CONTAINER
    segment_array_t &segments() { return _segments; }
    segment_it_t beginSegments() { return _segments.begin(); }
    segment_it_t endSegments() { return _segments.end(); }

    Segment *getSegment(int id) { if (id > _segments.size()) throw soap::base::OutOfRange("getSegment"); return _segments[id-1]; }

    std::string &getLabel() { return _label; }
    void setLabel(std::string label) { _label = label; }

    // PARTICLE CREATION & INTERFACE
    Segment &addSegment();
    Particle &addParticle(Segment &seg);

    // LAPLACIAN
    void setLaplacian(boost::python::object &np_laplacian, std::string np_dtype);
    boost::python::object getLaplacianNumpy(std::string np_dtype);
    bool hasLaplacian();
    laplace_t &getLaplacian();

    // BOUNDARY CREATION & INTERFACE
    Boundary *getBoundary() { return _box; }
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
        arch & _laplacian;
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

    laplace_t *_laplacian; 
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
