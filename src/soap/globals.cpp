#include "soap/globals.hpp"

namespace soap {

Logger GLOG;

void GLOG_SET_SILENT(bool silent) {
    GLOG.setSilent(silent);
    return;
}

void GLOG_SET_VERBOSE(bool verbose) {
    GLOG.setVerbose(verbose);
    return;
}

void GLOG_TOGGLE_SILENCE() {
    GLOG.toggleSilence();
    return;
}

bool GLOG_IS_SILENT() {
    return GLOG.isSilent();
}

void GLOG_SILENCE() {
    GLOG.silence();
    return;
}

}
