#include "soap/globals.hpp"

namespace soap {

Logger GLOG;

void GLOG_SILENCE() {
    //GLOG() << "Silencing logger ..." << std::endl;
    GLOG.silence();
    return;
}

void GLOG_TOGGLE_SILENCE() {
    GLOG.toggleSilence();
    return;
}

void GLOG_VERBOSE(bool verbose) {
    GLOG.setVerbose(verbose);
    return;
}

}
