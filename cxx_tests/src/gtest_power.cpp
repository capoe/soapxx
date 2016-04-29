#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <soap/power.hpp>
#include <soap/options.hpp>
#include "gtest_defines.hpp"

class TestPowerExpansionWithHeavisideCutoff : public ::testing::Test
{
public:

    soap::Options _options;
    soap::Basis *_basis;
    std::string _constructor_stdout;
    std::string _constructor_stdout_ref;
   
    virtual void SetUp() {
        
	    _options.set("radialbasis.type", "gaussian");
	    _options.set("radialbasis.mode", "adaptive");
	    _options.set("radialbasis.N", 9);
	    _options.set("radialbasis.sigma", 0.5);
	    _options.set("radialbasis.integration_steps", 15);
	    _options.set("radialcutoff.type", "heaviside");
	    _options.set("radialcutoff.Rc", 4.);
	    _options.set("radialcutoff.Rc_width", 0.5);
	    _options.set("radialcutoff.center_weight", 1.);
	    _options.set("angularbasis.type", "spherical-harmonic");
	    _options.set("angularbasis.L", 6);
        
        soap::RadialBasisFactory::registerAll();
        soap::AngularBasisFactory::registerAll();
        soap::CutoffFunctionFactory::registerAll();
        
        ::testing::internal::CaptureStdout();
        _basis = new soap::Basis(&_options);
        ::testing::internal::GetCapturedStdout();
    }

    virtual void TearDown() {
        delete _basis;
        _basis = NULL;
    }
};

TEST_F(TestPowerExpansionWithHeavisideCutoff, Gradients) {   

    typedef std::tuple<int, int, int, double, double, double, bool> nklxyzt_t;
    
    std::vector<nklxyzt_t> nklxyzt_list = {
        nklxyzt_t{0,0,0,1.,0.5,-0.5,true},
        nklxyzt_t{0,1,2,1.,0.5,-0.5,false},
        nklxyzt_t{0,3,1,-1.,-0.5,1.,true},
        nklxyzt_t{3,4,3,0.,0.3,-0.7,false},
        nklxyzt_t{7,2,4,-0.5,3.,4.2,true}
    };
    
    // Second set of positions if same_types = false
    double x_2 = 1.5;
    double y_2 = -0.5;
    double z_2 = 1.;
    soap::vec R_2(x_2, y_2, z_2);
    double r_2 = soap::linalg::abs(R_2);
    soap::vec d_2 = (r_2 > 0.) ? R_2/r_2 : soap::vec(0.,0.,1.);
    double weight0_2 = 1.;
    double weight_scale_2 = _basis->getCutoff()->calculateWeight(r_2);
    
    std::stringstream output;
    std::string output_ref = "n=0 k=0 l=0 x=+1.0000 y=+0.5000 z=-0.5000 X_re=+7.9659943e-02 dX_re=-3.1906261e-01 -1.5953130e-01 +1.5953130e-01 X_im=+0.0000000e+00 dX_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=0 k=1 l=2 x=+1.0000 y=+0.5000 z=-0.5000 x_2=+1.0000 y_2=+0.5000 z_2=-0.5000 X_re=+5.3867510e-02 dX_re=-6.0228017e-02 -3.0114009e-02 +3.0114009e-02 X_im=+0.0000000e+00 dX_im=+5.1701290e-18 +0.0000000e+00 -3.4467527e-18 n=0 k=3 l=1 x=-1.0000 y=-0.5000 z=+1.0000 X_re=+2.1246860e-02 dX_re=+2.0772436e-03 +1.0386218e-03 -2.0772436e-03 X_im=+0.0000000e+00 dX_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=3 k=4 l=3 x=+0.0000 y=+0.3000 z=-0.7000 x_2=+0.0000 y_2=+0.3000 z_2=-0.7000 X_re=+4.9622475e-06 dX_re=-1.3359712e-04 -1.1974146e-05 +5.6734884e-06 X_im=+4.4449428e-23 dX_im=-7.1119085e-21 +6.2229199e-21 +9.7788741e-21 n=7 k=2 l=4 x=-0.5000 y=+3.0000 z=+4.2000 X_re=+1.0393232e-11 dX_re=+1.3777689e-11 -8.2666133e-11 -1.1573259e-10 X_im=+0.0000000e+00 dX_im=-6.5423244e-29 +2.1028900e-28 +1.4953884e-28 ";
    
    for (auto it = nklxyzt_list.begin(); it != nklxyzt_list.end(); ++it) {
        // Extract parameters
        int n = std::get<0>(*it);
        int k = std::get<1>(*it);
        int l = std::get<2>(*it);
        double x = std::get<3>(*it);
        double y = std::get<4>(*it);
        double z = std::get<5>(*it);
        bool test_same_types = std::get<6>(*it);
        int nk = _basis->getRadBasis()->N()*n+k;
        
        // Distance, direction, weight
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = (r > 0.) ? R/r : soap::vec(0.,0.,1.);
    
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        
        // Adjust if Xnkl contracted from related densities qnlm (=> same_types = true)
        if (test_same_types) {
            R_2 = R;
            r_2 = r;
            d_2 = d;
            weight0_2 = weight0;
            weight_scale_2 = weight_scale;
        }
        
        // Compute
        ::testing::internal::CaptureStdout();
        
        soap::BasisExpansion basex1(_basis);
        basex1.computeCoefficients(r, d, weight0, weight_scale, sigma, true); // <- gradients = true
        
        soap::BasisExpansion basex2(_basis);
        basex2.computeCoefficients(r_2, d_2, weight0_2, weight_scale_2, sigma, false); // <- gradients = false
        
        soap::PowerExpansion powex(_basis);
        powex.computeCoefficients(&basex1, &basex2); // <- note argument duplication (gradients = false in both cases)
        
        soap::PowerExpansion powex_grad(_basis);
        powex_grad.computeCoefficientsGradients(&basex1, &basex2, test_same_types);
        
        ::testing::internal::GetCapturedStdout();
        
        // Extract
        soap::PowerExpansion::coeff_t &coeff = powex.getCoefficients();
        soap::PowerExpansion::coeff_t &coeff_grad_x = powex_grad.getCoefficientsGradX();
        soap::PowerExpansion::coeff_t &coeff_grad_y = powex_grad.getCoefficientsGradY();
        soap::PowerExpansion::coeff_t &coeff_grad_z = powex_grad.getCoefficientsGradZ();
        
        output
            << boost::format("n=%1$d k=%2$d l=%3$d ") % n % k % l
            << boost::format("x=%1$+1.4f y=%2$+1.4f z=%3$+1.4f ") % x % y % z
            << std::flush;
        if (!test_same_types) output
            << boost::format("x_2=%1$+1.4f y_2=%2$+1.4f z_2=%3$+1.4f ") % x % y % z
            << std::flush;
        output
            << boost::format("X_re=%1$+1.7e dX_re=%2$+1.7e %3$+1.7e %4$+1.7e ")
                % coeff(nk,l).real() % coeff_grad_x(nk,l).real() % coeff_grad_y(nk,l).real() % coeff_grad_z(nk,l).real()
            << boost::format("X_im=%1$+1.7e dX_im=%2$+1.7e %3$+1.7e %4$+1.7e ") 
                % coeff(nk,l).imag() % coeff_grad_x(nk,l).imag() % coeff_grad_y(nk,l).imag() % coeff_grad_z(nk,l).imag()
            << std::flush;
    }
    
    // VERIFY
    EXPECT_EQ(output.str(), output_ref);
 
    /*
    // MANUAL TESTING
    int N = 150;
    double dx = 0.05;
    
    bool test_same_types = true;
    soap::vec R_2(1.5, -0.5, 1.);
    double r_2 = soap::linalg::abs(R_2);
    soap::vec d_2 = (r_2 > 0.) ? R_2/r_2 : soap::vec(0.,0.,1.);
    double weight0_2 = 1.;
    double weight_scale_2 = _basis->getCutoff()->calculateWeight(r_2);
    
    std::string logfile = "tmp";
    std::ofstream ofs;
    ofs.open(logfile.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Bad file handle: " + logfile);
    }
    
    for (int i = 0; i < N; ++i) {        
        
        double x = i*dx;
        double y = 0.5;
        double z = -0.2;
    
        // Distance, direction, weight
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = (r > 0.) ? R/r : soap::vec(0.,0.,1.);
    
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        
        // Adjust if Xnkl contracted from related densities qnlm (=> same_types = true)
        if (test_same_types) {
            R_2 = R;
            r_2 = r;
            d_2 = d;
            weight0_2 = weight0;
            weight_scale_2 = weight_scale;
        }
        
        // Compute
        ::testing::internal::CaptureStdout();
        
        soap::BasisExpansion basex1(_basis);
        basex1.computeCoefficients(r, d, weight0, weight_scale, sigma, true); // <- gradients = true
        
        soap::BasisExpansion basex2(_basis);
        basex2.computeCoefficients(r_2, d_2, weight0_2, weight_scale_2, sigma, false); // <- gradients = false
        
        soap::PowerExpansion powex(_basis);
        powex.computeCoefficients(&basex1, &basex2); // <- note argument duplication (gradients = false in both cases)
        
        soap::PowerExpansion powex_grad(_basis);
        powex_grad.computeCoefficientsGradients(&basex1, &basex2, test_same_types);
        
        ::testing::internal::GetCapturedStdout();
        
        // Extract
        soap::PowerExpansion::coeff_t &coeff = powex.getCoefficients();
        soap::PowerExpansion::coeff_t &coeff_grad_x = powex_grad.getCoefficientsGradX();
        soap::PowerExpansion::coeff_t &coeff_grad_y = powex_grad.getCoefficientsGradY();
        soap::PowerExpansion::coeff_t &coeff_grad_z = powex_grad.getCoefficientsGradZ();
        
        int n = 7;
        int k = 5;
        int l = 1;
        int nk = _basis->getRadBasis()->N()*n+k;
        
        // x (#1)  Qnlm_re (#2)  dQnlm_dx_re (#4)
        ofs << boost::format("%1$+1.4f %2$+1.7e %3$+1.7e %4$+1.7e %5$+1.7e %6$+1.7e %7$+1.7e %8$+1.7e %9$+1.7e") 
            % R.getX() % coeff(nk,l).real() % coeff(nk,l).imag()
            % coeff_grad_x(nk,l).real() % coeff_grad_x(nk,l).imag()
            % coeff_grad_y(nk,l).real() % coeff_grad_y(nk,l).imag()
            % coeff_grad_z(nk,l).real() % coeff_grad_z(nk,l).imag()
             << std::endl;
    }
    
    ofs.close();
    */
}








