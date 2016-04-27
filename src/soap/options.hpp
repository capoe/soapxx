#ifndef _SOAP_OPTIONS_HPP
#define _SOAP_OPTIONS_HPP

#include <map>
#include <boost/format.hpp>

#include "soap/types.hpp"

namespace soap {

class Options
{
public:
	typedef std::map<std::string, std::string> map_options_t;
	typedef std::map<std::string, bool> map_exclude_t;

	// CONSTRUCT & DESTRUCT
	Options();
    ~Options() {}

    // GET & SET
    template<typename return_t>
    return_t get(std::string key) {
        return soap::lexical_cast<return_t, std::string>(_key_value_map[key], "wrong or missing type in " + key);
    }
    void set(std::string key, std::string value) { _key_value_map[key] = value; }
    void set(std::string key, int value) { this->set(key, boost::lexical_cast<std::string>(value)); }
    void set(std::string key, double value) { this->set(key, boost::lexical_cast<std::string>(value)); }
    //void set(std::string key, bool value) { this->set(key, boost::lexical_cast<std::string>(value)); }
	void configureCenters(boost::python::list center_excludes) { _center_excludes = center_excludes; }
	std::string summarizeOptions();

	// EXCLUSIONS TODO Move this to a distinct class
	void excludeCenters(boost::python::list &types);
	void excludeTargets(boost::python::list &types);
	bool doExcludeCenter(std::string &type);
	bool doExcludeTarget(std::string &type);

	// PYTHON
	static void registerPython();

	// SERIALIZATION
	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _key_value_map;
		arch & _exclude_center;
		arch & _exclude_target;
		return;
	}

private:
	boost::python::list _center_excludes;
	map_options_t _key_value_map;

	map_exclude_t _exclude_center;
	map_exclude_t _exclude_target;
};

}

#endif
