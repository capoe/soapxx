#include <boost/version.hpp>
#if BOOST_VERSION >= 106400
#define BOOST_PYTHON_STATIC_LIB  
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
#endif
#include "soap/structure.hpp"


namespace soap {

// ========
// PARTICLE
// ========

void Particle::null() {
	_id = -1;
	_pos = vec(0,0,0);
	_name = "?";
	_type_id = -1;
	_type = "?";
	_mass = 0.;
	_weight = 0.;
	_sigma = 0.;
	_segment = NULL;
}

void Particle::model(Particle &model) {
    // ID assigned internally, so is Segment pointer
    _pos = model._pos;
    _name = model._name;
    _type_id = model._type_id;
    _type = model._type;
    _mass = model._mass;
    _weight = model._weight;
    _sigma = model._sigma;
}

#if BOOST_VERSION >= 106400
boost::python::numpy::ndarray Particle::getPosNumeric() {
	boost::python::numpy::ndarray pos(boost::python::numpy::array(boost::python::make_tuple(_pos.x(), _pos.y(), _pos.z()))); return pos;
}
#else
boost::python::numeric::array Particle::getPosNumeric() {
	boost::python::numeric::array pos(boost::python::make_tuple(_pos.x(), _pos.y(), _pos.z())); return pos;
}
#endif

void Particle::registerPython() {
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

// =======
// SEGMENT
// =======

Particle &Segment::addParticle(Particle *new_part) {
	_particles.push_back(new_part);
	return *new_part;
}

void Segment::registerPython() {
	using namespace boost::python;
	class_<Segment, Segment*>("Segment", init<int>())
        .add_property("id", &Segment::getId, &Segment::setId)
	    .add_property("name", make_function(&Segment::getName, copy_non_const()), &Segment::setName)
	    .add_property("type", make_function(&Segment::getType, copy_non_const()), &Segment::setType)
	    .add_property("particles", range<return_value_policy<reference_existing_object> >(&Segment::beginParticles, &Segment::endParticles))
	    .def("addParticle", &Segment::addParticle, return_value_policy<reference_existing_object>());
}


// =========
// STRUCTURE
// =========

Structure::Structure(std::string label) {
	this->null();
	_label = label;

}

Structure::Structure() {
	this->null();
}

Structure::Structure(const Structure &structure) {
    this->null();
    assert(false && "COPY CONSTRUCTOR NOT AVAILABLE. USE ::model(Structure &) instead.");
}

void Structure::model(Structure &structure) {
    this->null();
    // ID, label, box
    _id = structure._id;
    _label = structure._label;
    this->setBoundary(structure._box->getBox());
    // Segments, particles 
    for (segment_it_t sit = structure.beginSegments(); sit != structure.endSegments(); ++sit) {
        Segment &new_seg = this->addSegment();
        for (particle_it_t pit = (*sit)->beginParticles(); pit != (*sit)->endParticles(); ++pit) {
            Particle &new_part = this->addParticle(new_seg);
            new_part.model(*(*pit));
        }
    }
}

void Structure::null() {
	_id = -1;
	_label = "?";
	_box = NULL;
	_center = NULL;
	_has_center = false;
	matrix box;
	box.ZeroMatrix();
	this->setBoundary(box);
}

Structure::~Structure() {
	delete _box;
	_box = NULL;
	segment_it_t sit;
	for (sit = _segments.begin(); sit != _segments.end(); ++sit) {
		delete *sit;
	}
	_segments.clear();
	particle_it_t pit;
	for (pit = _particles.begin(); pit != _particles.end(); ++pit) {
		delete *pit;
	}
	_particles.clear();
}

Segment &Structure::addSegment() {
	int id = _segments.size()+1;
	Segment *new_seg = new Segment(id);
	_segments.push_back(new_seg);
	return *new_seg;
}

Particle &Structure::addParticle(Segment &seg) {
	int id = _particles.size()+1;
	Particle *new_part = new Particle(id);
	_particles.push_back(new_part);
	seg.addParticle(new_part);
	return *new_part;
}

#if BOOST_VERSION >= 106400
boost::python::numpy::ndarray Structure::connectNumeric(
	const boost::python::numpy::ndarray &a1,
	const boost::python::numpy::ndarray &a2) {
	vec r1(a1);
	vec r2(a2);
	vec dr = this->connect(r1, r2);
	return boost::python::numpy::ndarray(boost::python::numpy::array(boost::python::make_tuple(
		dr.x(), dr.y(), dr.z())));
}
#else
boost::python::numeric::array Structure::connectNumeric(
	const boost::python::numeric::array &a1,
	const boost::python::numeric::array &a2) {
	vec r1(a1);
	vec r2(a2);
	vec dr = this->connect(r1, r2);
	return boost::python::numeric::array(boost::python::make_tuple(
	        dr.x(), dr.y(), dr.z()));
}
#endif

void Structure::setBoundary(const matrix &box) {
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

#if BOOST_VERSION >= 106400
void Structure::setBoundaryNumeric(const boost::python::numpy::ndarray &m) {
	matrix box(m);
	this->setBoundary(box);
}
#else
void Structure::setBoundaryNumeric(const boost::python::numeric::array &m) {
	matrix box(m);
	this->setBoundary(box);
}
#endif

#if BOOST_VERSION >= 106400
boost::python::numpy::ndarray Structure::getBoundaryNumeric() {
	matrix box = _box->getBox();
	boost::python::numpy::ndarray box_np(boost::python::numpy::array(boost::python::make_tuple(
		box.get(0,0), box.get(0,1), box.get(0,2),
		box.get(1,0), box.get(1,1), box.get(1,2),
		box.get(2,0), box.get(2,1), box.get(2,2))));
	box_np.reshape(boost::python::make_tuple(3,3));
	return box_np;
}
#else
boost::python::numeric::array Structure::getBoundaryNumeric() {
	matrix box = _box->getBox();
	boost::python::numeric::array box_np(boost::python::make_tuple(
		box.get(0,0), box.get(0,1), box.get(0,2),
		box.get(1,0), box.get(1,1), box.get(1,2),
		box.get(2,0), box.get(2,1), box.get(2,2)));
	box_np.resize(3,3);
	return box_np;
}
#endif

void Structure::registerPython() {
	using namespace boost::python;
	class_<Structure>("Structure", init<std::string>())
       .def("model", &Structure::model)
	   .def("addSegment", &Structure::addSegment, return_value_policy<reference_existing_object>())
	   .def("getSegment", &Structure::getSegment, return_value_policy<reference_existing_object>())
	   .def("addParticle", &Structure::addParticle, return_value_policy<reference_existing_object>())
	   .def("getParticle", &Structure::getParticle, return_value_policy<reference_existing_object>())
	   .def("__iter__", range<return_value_policy<reference_existing_object> >(&Structure::beginParticles, &Structure::endParticles))
	   .add_property("particles", range<return_value_policy<reference_existing_object> >(&Structure::beginParticles, &Structure::endParticles))
	   .add_property("segments", range<return_value_policy<reference_existing_object> >(&Structure::beginSegments, &Structure::endSegments))
       .add_property("n_particles", &Structure::getNumberOfParticles)
	   .def("connect", &Structure::connectNumeric)
	   .add_property("box", &Structure::getBoundaryNumeric, &Structure::setBoundaryNumeric)
	   .add_property("label", make_function(&Structure::getLabel, copy_non_const()), &Structure::setLabel);
	class_<particle_array_t>("ParticleContainer")
	   .def(vector_indexing_suite<particle_array_t>());
	class_<segment_array_t>("SegmentContainer")
	    .def(vector_indexing_suite<segment_array_t>());
}

} /* CLOSE NAMESPACE */
