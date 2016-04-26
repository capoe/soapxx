#include "soap/globals.hpp"

namespace soap {

Logger GLOG;

void GLOG_SILENCE() {
    GLOG() << "Silencing logger ..." << std::endl;
    GLOG.silence();
    return;
}

}

