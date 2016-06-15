#ifndef _SOAP_CUTOFF_HPP
#define _SOAP_CUTOFF_HPP

#include <string>
#include <math.h>
#include <vector>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "soap/base/exceptions.hpp"
#include "soap/base/objectfactory.hpp"
#include "soap/globals.hpp"
#include "soap/options.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

class CutoffFunction
{
public:
	std::string &identify() { return _type; }
	const double &getCutoff() { return _Rc; }
    const double &getCutoffWidth() { return _Rc_width; }
    const double &getCenterWeight() { return _center_weight; }
    CutoffFunction() : 
        _type("shifted-cosine"), 
        _Rc(-1), 
        _Rc_width(-1.),
        _center_weight(-1.) {;}
    virtual ~CutoffFunction() {;}
    virtual void configure(Options &options);
    virtual bool isWithinCutoff(double r);
    virtual double calculateWeight(double r);
    virtual vec calculateGradientWeight(double r, vec d);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _type;
    	arch & _Rc;
    	arch & _Rc_width;
    	arch & _center_weight;
    }

protected:
    std::string _type;
    double _Rc;
    double _Rc_width;
    double _center_weight;

public:
    static constexpr double WEIGHT_ZERO = 1e-10;
};

class CutoffFunctionHeaviside : public CutoffFunction
{
public:
    CutoffFunctionHeaviside() { _type = "heaviside"; _Rc = -1; }
    void configure(Options &options);
    bool isWithinCutoff(double r);
    double calculateWeight(double r);
    virtual vec calculateGradientWeight(double r, vec d);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & boost::serialization::base_object<CutoffFunction>(*this);
    }
};


class CutoffFunctionFactory
    : public soap::base::ObjectFactory<std::string, CutoffFunction>
{
private:
    CutoffFunctionFactory() {}
public:
    static void registerAll(void);
    CutoffFunction *create(const std::string &key);
    friend CutoffFunctionFactory &CutoffFunctionOutlet();
};

inline CutoffFunctionFactory &CutoffFunctionOutlet() {
    static CutoffFunctionFactory _instance;
    return _instance;
}

inline CutoffFunction *CutoffFunctionFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        CutoffFunction *basis = (it->second)();
        return basis;
    }
    else {
        throw std::runtime_error("Factory key " + key + " not found.");
    }
}

} /* CLOSE NAMESPACE */

BOOST_CLASS_EXPORT_KEY(soap::CutoffFunction);
BOOST_CLASS_EXPORT_KEY(soap::CutoffFunctionHeaviside);

#endif
