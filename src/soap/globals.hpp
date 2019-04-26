#ifndef _SOAP_GLOBALS_H
#define	_SOAP_GLOBALS_H

#include "soap/base/logger.hpp"

namespace soap {

    extern Logger GLOG;
    void GLOG_SET_SILENT(bool silent);
    void GLOG_SET_VERBOSE(bool verbose);
    void GLOG_TOGGLE_SILENCE();
    bool GLOG_IS_SILENT();
    void GLOG_SILENCE();

    namespace constants {
        const double ANGSTROM_TO_BOHR = 1./0.52917721067;
    }

}

#endif
