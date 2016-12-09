#include "soap/contraction.hpp"

namespace soap {

EnergySpectrum::EnergySpectrum(Spectrum &spectrum, Options &options) : _global_map_powspec(NULL) {
    GLOG() << "Configuring energy spectrum ..." << std::endl;
    _options = &options;
    _spectrum = &spectrum;
    // Configure cut-off function
    Options options_tmp = Options();
    options_tmp.set("radialcutoff.Rc", _options->get<std::string>("energyspectrum.radialcutoff.Rc"));
    options_tmp.set("radialcutoff.Rc_width", _options->get<std::string>("energyspectrum.radialcutoff.Rc_width"));
    options_tmp.set("radialcutoff.center_weight", _options->get<std::string>("energyspectrum.radialcutoff.center_weight"));
    if (_options->hasKey("radialcutoff.Rc_heaviside")) {
        options_tmp.set("radialcutoff.Rc_heaviside", _options->get<std::string>("energyspectrum.radialcutoff.Rc_heaviside"));
    }
    _cutoff = CutoffFunctionOutlet().create(_options->get<std::string>("energyspectrum.radialcutoff.type"));
	_cutoff->configure(options_tmp);
    return;
}

EnergySpectrum::~EnergySpectrum() {
    // Cut-off function
    delete _cutoff;
    _cutoff = NULL;
    // Global spectrum
    if (_global_map_powspec) {
        for (auto it = _global_map_powspec->begin(); it != _global_map_powspec->end(); ++it) {
            delete it->second;
        }
        _global_map_powspec->clear();
    }
    return;
}

PowerExpansion *EnergySpectrum::getPowerGlobal(std::string s1, std::string s2) {
    assert(_global_map_powspec && "Compute global spectrum first.");
    AtomicSpectrum::type_pair_t types(s1, s2);
    auto it = _global_map_powspec->find(types);
    if (it == _global_map_powspec->end()) {
        return NULL;
    }
    else {
        return (*_global_map_powspec)[types];
    }
}

void EnergySpectrum::compute() {
    GLOG() << "Computing energy spectrum ..." << std::endl;
    // TODO Add images
    double R0 = _options->get<double>("energyspectrum.r0");
    double gamma = _options->get<double>("energyspectrum.gamma");
    bool norm_global = true; // Not used / not required (normalization on python level)

    // Allocate map for power spectra: type pair -> power spectrum
    if (_global_map_powspec) {
        for (auto it = _global_map_powspec->begin(); it != _global_map_powspec->end(); ++it) {
            delete it->second;
        }
        _global_map_powspec->clear();
    }
    this->_global_map_powspec = new AtomicSpectrum::map_xnkl_t;

    // Loop over pairs of expansion sites ...
    int pair_count = 0;
    double total_weight = 0.;
    for (Spectrum::atomic_it_t it = _spectrum->beginAtomic(); it != _spectrum->endAtomic(); ++it) {
        AtomicSpectrum *aspec = *it;
        AtomicSpectrum::map_qnlm_t map_qnlm_a = aspec->getQnlmMap();
        for (Spectrum::atomic_it_t jt = _spectrum->beginAtomic(); jt != _spectrum->endAtomic(); ++jt) {
            AtomicSpectrum *bspec = *jt;
            AtomicSpectrum::map_qnlm_t map_qnlm_b = bspec->getQnlmMap();

            // Distance
            vec dr_ab = _spectrum->getStructure()->connect(aspec->getCenterPos(), bspec->getCenterPos());
            double R_ab = soap::linalg::abs(dr_ab) + R0;
            // Cut-off check
            if (! _cutoff->isWithinCutoff(R_ab-R0)) continue;
            double weight_scale = _cutoff->calculateWeight(R_ab-R0);
            GLOG() << aspec->getCenterId() << ":" << bspec->getCenterId() << " R=" << R_ab << " w=" << weight_scale << std::endl;
            pair_count += 1;
            total_weight += weight_scale;

            // Contract nlm-expansions ...
            AtomicSpectrum::map_qnlm_t::iterator it1;
            AtomicSpectrum::map_qnlm_t::iterator it2;
            // Iterate over all type pairs
            for (it1 = map_qnlm_a.begin(); it1 != map_qnlm_a.end(); ++it1) {
                for (it2 = map_qnlm_b.begin(); it2 != map_qnlm_b.end(); ++it2) {
                    // Compute scale factor
                    double scale = (it1 == it2) ? 1./pow(R_ab, gamma) : 0.5/pow(R_ab, gamma);
                    scale *= weight_scale;
                    // Create new pair spectrum or add to existing
                    AtomicSpectrum::type_pair_t types(it1->first, it2->first);
                    GLOG() << " " << types.first << ":" << types.second << std::flush;
                    auto it = _global_map_powspec->find(types);
                    if (it == _global_map_powspec->end()) {
                        GLOG() << "*" << std::flush;
                        (*_global_map_powspec)[types] = new PowerExpansion(_spectrum->getBasis());
                        (*_global_map_powspec)[types]->computeCoefficientsHermConj(it1->second, it2->second, scale);
                    }
                    else {
                        GLOG() << "+" << std::flush;
                        PowerExpansion powex(_spectrum->getBasis());
                        powex.computeCoefficientsHermConj(it1->second, it2->second, scale);
                        (*_global_map_powspec)[types]->add(&powex);
                    }
                }
            }
            GLOG() << std::endl;
        }
    }
    return;
}

void EnergySpectrum::registerPython() {
    using namespace boost::python;
    class_<EnergySpectrum>("EnergySpectrum", init<Spectrum &, Options &>())
        .def("compute", &EnergySpectrum::compute)
        .def("getPowerGlobal", &EnergySpectrum::getPowerGlobal, return_value_policy<reference_existing_object>());
}

}
