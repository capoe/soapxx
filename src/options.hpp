#ifndef _SOAP_OPTIONS_HPP
#define _SOAP_OPTIONS_HPP
#include <boost/format.hpp>
#include "types.hpp"
#include <map>

namespace soap {

class Options
{
public:
	Options()
		: _N_real(8), _L_real(6), _Rc_real(5.),
		  _N_recip(8), _L_recip(6), _Rc_recip(5.),
		  _b1(vec(1,0,0)), _b2(vec(0,1,0)), _b3(vec(0,0,1)),
		  _center_w0(1.), _center_excludes(boost::python::list()) {
		// SET DEFAULTS
		this->set("radialbasis.integration_steps", 15);
		this->set("radialbasis.mode", "equispaced");
	}
   ~Options() {}

    template<typename return_t>
    return_t get(std::string key) {
        return soap::lexical_cast<return_t, std::string>(_key_value_map[key], "wrong or missing type in " + key);
    }

    void set(std::string key, std::string value) { _key_value_map[key] = value; }
    void set(std::string key, int value) { this->set(key, boost::lexical_cast<std::string>(value)); }
    void set(std::string key, double value) { this->set(key, boost::lexical_cast<std::string>(value)); }

    /*
    template<class value_t>
    void set(std::string key, value_t value) {
    	std::cout << "value" << value << "_" << std::endl;
    	_key_value_map[key] = boost::lexical_cast<std::string>(value);
    }
    */

	void configureRealBasis(int N_real, int L_real, double Rc_real, std::string type) {
		_N_real = N_real;
		_L_real = L_real;
		_Rc_real = Rc_real;
		_type_real = type;
		_key_value_map["radialbasis.Rc"] = boost::lexical_cast<std::string>(_Rc_real);
		_key_value_map["radialbasis.L"] = boost::lexical_cast<std::string>(_L_real);
		_key_value_map["radialbasis.N"] = boost::lexical_cast<std::string>(_N_real);
		_key_value_map["radialbasis.type"] = boost::lexical_cast<std::string>(_type_real);
	}
	void configureReciprocalBasis(int N_recip, int L_recip, double Rc_recip) {
		_N_recip = N_recip;
		_L_recip = L_recip;
		_Rc_recip = Rc_recip;
	}
	void configureCenters(double center_w0, boost::python::list center_excludes) {
		_center_w0 = center_w0;
		_center_excludes = center_excludes;
	}
	void configureReciprocalLattice(vec b1, vec b2, vec b3) {
		_b1 = b1;
		_b2 = b2;
		_b3 = b3;
	}
	std::string summarizeOptions() {
		std::string info = "";
		info += "Options:\n";
//		info += "o Centers:                  ";
//		info += (boost::format("W0=%1% Excl#=%2%\n") % _center_w0 % boost::python::len(_center_excludes)).str();
//		info += "o Real-space basis:         ";
//		info += (boost::format("N=%1% L=%2% Rc=%3%\n") % _N_real % _L_real % _Rc_real).str();
//		info += "o Reciprocal-space basis:   ";
//		info += (boost::format("N=%1% L=%2% Rc=%3%\n") % _N_recip % _L_recip % _Rc_recip).str();
//		info += "o Reciprocal-space lattice: ";
//		info += (boost::format("b1x=%1% b1y=%2% b1z=%3% ") % _b1.x() % _b1.y() % _b1.z()).str();
//		info += (boost::format("b2x=%1% b2y=%2% b2z=%3% ") % _b2.x() % _b2.y() % _b2.z()).str();
//		info += (boost::format("b3x=%1% b3y=%2% b3z=%3% ") % _b3.x() % _b3.y() % _b3.z()).str();

		std::map<std::string, std::string>::iterator it;
		for (it = _key_value_map.begin(); it != _key_value_map.end(); ) {
			info += (boost::format(" o %1$-30s : %2$s") % it->first % it->second).str();
			if (++it != _key_value_map.end()) info += "\n";
		}
		return info;
	}
	static void registerPython();

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _key_value_map;
		return;
	}

private:
	// REAL SPACE
	int _N_real;
	int _L_real;
	double _Rc_real;
	std::string _type_real;
	// RECIPROCAL SPACE
	int _N_recip;
	int _L_recip;
	double _Rc_recip;
	vec _b1;
	vec _b2;
	vec _b3;
	// CENTERS
	double _center_w0;
	boost::python::list _center_excludes;

	std::map<std::string, std::string> _key_value_map;

};

}

#endif
