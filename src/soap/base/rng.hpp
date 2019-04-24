#ifndef SOAP_BASE_RNG_HPP 
#define SOAP_BASE_RNG_HPP 
#include <vector>

namespace soap { namespace base {

// MARSAGLIA pseudo random number generator
// See: G. Marsaglia and A. Zaman. Toward a universal random number generator,
// Statistics & Probability Letters, 9(1):35–39, 1990.
class RNG {
  public:
    RNG(){};
    ~RNG(){};
    void init(int nA1, int nA2, int nA3, int nB1);
    double uniform(void);
    int uniform_int(int max_int);
    double gaussian(double sigma);
  private:
    static const int MARS_FIELD_SIZE = 98;
    std::vector<double> MARSarray;
    double MARSc, MARScd, MARScm;
    int MARSi, MARSj;
};

}}

#endif
