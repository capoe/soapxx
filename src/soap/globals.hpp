#ifndef _SOAP_GLOBALS_H
#define	_SOAP_GLOBALS_H

#include "soap/base/logger.hpp"

namespace soap {

    extern Logger GLOG;
    void GLOG_SILENCE();
    void GLOG_VERBOSE(bool verbose);

    namespace constants {
        const double ANGSTROM_TO_BOHR = 1./0.52917721067;
    }

}

#endif
