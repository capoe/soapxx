#ifndef _SOAP_CONTRACTION_HPP
#define _SOAP_CONTRACTION_HPP

#include "soap/spectrum.hpp"

namespace soap {

class EnergySpectrum
{
public:
    EnergySpectrum(Spectrum &spectrum, Options &options);
   ~EnergySpectrum();
    void compute();
    PowerExpansion *getPowerGlobal(std::string s1, std::string s2);
    static void registerPython();
private:
    Spectrum *_spectrum;
    Options *_options;
    CutoffFunction *_cutoff;
    AtomicSpectrum::map_xnkl_t *_global_map_powspec;
};

}






#endif /* _SOAP_CONTRACTION_HPP */
