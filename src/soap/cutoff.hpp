#ifndef _SOAP_CUTOFF_HPP
#define _SOAP_CUTOFF_HPP

#include <string>
#include <math.h>
#include <vector>

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
    virtual double calculateWeight(double r) {
        double weight_at_r = 1.;
        if (r > _Rc) {
            weight_at_r = -1.;
        }
        else if (r <= _Rc - _Rc_width) {
            weight_at_r = 1.;
        }
        else {
            weight_at_r = 0.5*(1+cos(M_PI*(r-_Rc+_Rc_width)/_Rc_width));
        }
        return weight_at_r;
    }

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

}

#endif
