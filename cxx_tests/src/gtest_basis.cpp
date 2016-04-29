#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <soap/basis.hpp>
#include <soap/options.hpp>
#include "gtest_defines.hpp"

class TestBasisExpansion : public ::testing::Test
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
	    _options.set("radialcutoff.type", "shifted-cosine");
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

TEST_F(TestBasisExpansion, Gradients) {
    
    typedef std::tuple<int, int, int, double, double, double> nlmxyz_t;
    
    std::vector<nlmxyz_t> nlmxyz_list = {
        nlmxyz_t{0,0,0,1.,0.5,-0.5},
        nlmxyz_t{0,1,0,1.,0.5,-0.5},
        nlmxyz_t{0,3,-1,-1.,-0.5,1.},
        nlmxyz_t{3,4,1,0.,0.3,-0.7},
        nlmxyz_t{7,2,-1,-0.5,3.,4.}
    };
    
    std::stringstream output;
    std::string output_ref = "n=0 l=0 m=0 x=+1.0000 y=+0.5000 z=-0.5000 Q_re=+9.4683121e-02 dQ_re=-1.8961753e-01 -9.4808765e-02 +9.4808765e-02 Q_im=+0.0000000e+00 dQ_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=0 l=1 m=0 x=+1.0000 y=+0.5000 z=-0.5000 Q_re=-4.7738237e-02 dQ_re=+1.0923622e-01 +5.4618111e-02 +4.0858364e-02 Q_im=-0.0000000e+00 dQ_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=0 l=3 m=-1 x=-1.0000 y=-0.5000 z=+1.0000 Q_re=-1.2554214e-02 dQ_re=-2.8055842e-02 -2.0305028e-02 -5.0416312e-03 Q_im=+6.2771069e-03 dQ_im=+2.0305028e-02 -2.4017000e-03 +2.5208156e-03 n=3 l=4 m=1 x=+0.0000 y=+0.3000 z=-0.7000 Q_re=-2.1144554e-20 dQ_re=-1.1510939e-03 +8.4096906e-20 -4.4449962e-19 Q_im=-3.4532818e-04 dQ_im=+2.2521571e-20 +1.6936152e-03 -4.1418340e-03 n=7 l=2 m=-1 x=-0.5000 y=+3.0000 z=+4.0000 Q_re=+7.0360862e-15 dQ_re=-1.1432964e-14 -1.5835248e-14 -1.9354642e-14 Q_im=+4.2216517e-14 dQ_im=+1.5835248e-14 -8.0939315e-14 -1.1612785e-13 "; 
    
    for (auto it = nlmxyz_list.begin(); it != nlmxyz_list.end(); ++it) {
        // Extract parameters
        int n = std::get<0>(*it);
        int l = std::get<1>(*it);
        int m = std::get<2>(*it);
        double x = std::get<3>(*it);
        double y = std::get<4>(*it);
        double z = std::get<5>(*it);
        int lm = l*l+l+m;
        
        // Position, radius, direction
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = R/r;
    
        // Weights
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        bool gradients = true;
        
        // Compute gradients
        ::testing::internal::CaptureStdout();
        soap::BasisExpansion basex(_basis);
        basex.computeCoefficients(r, d, weight0, weight_scale, sigma, gradients);
        ::testing::internal::GetCapturedStdout();
        
        // Extract results
        soap::BasisExpansion::coeff_t &coeff = basex.getCoefficients();
        soap::BasisExpansion::coeff_t &coeff_grad_x = basex.getCoefficientsGradX();
        soap::BasisExpansion::coeff_t &coeff_grad_y = basex.getCoefficientsGradY();
        soap::BasisExpansion::coeff_t &coeff_grad_z = basex.getCoefficientsGradZ();
        
        
        output
            << boost::format("n=%1$d l=%2$d m=%3$d ") % n % l % m
            << boost::format("x=%1$+1.4f y=%2$+1.4f z=%3$+1.4f ") % x % y % z
            << boost::format("Q_re=%1$+1.7e dQ_re=%2$+1.7e %3$+1.7e %4$+1.7e ")
                % coeff(n,lm).real() % coeff_grad_x(n,lm).real() % coeff_grad_y(n,lm).real() % coeff_grad_z(n,lm).real()
            << boost::format("Q_im=%1$+1.7e dQ_im=%2$+1.7e %3$+1.7e %4$+1.7e ") 
                % coeff(n,lm).imag() % coeff_grad_x(n,lm).imag() % coeff_grad_y(n,lm).imag() % coeff_grad_z(n,lm).imag()
            << std::flush;
    }
    
    // VERIFY
    EXPECT_EQ(output.str(), output_ref);    
    
    // GENERATE REFERENCE
    // std::cout << "\"" << output.str() << "\"" << std::endl;
    
    /*
    // MANUAL TESTING
    int N = 200;
    double dx = 0.05;    
    
    std::string logfile = "tmp";
    std::ofstream ofs;
    ofs.open(logfile.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Bad file handle: " + logfile);
    }
    
    for (int i = 0; i < N; ++i) {        
        
        double z = -i*dx+5.;
        double x = 0.2;
        double y = -3.;
    
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = R/r;
    
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        bool gradients = true;        
        
        // Compute
        ::testing::internal::CaptureStdout();
        soap::BasisExpansion basex(_basis);
        basex.computeCoefficients(r, d, weight0, weight_scale, sigma, gradients);
        ::testing::internal::GetCapturedStdout();
        
        // Extract
        soap::BasisExpansion::coeff_t &coeff = basex.getCoefficients();
        soap::BasisExpansion::coeff_t &coeff_grad_x = basex.getCoefficientsGradX();
        soap::BasisExpansion::coeff_t &coeff_grad_y = basex.getCoefficientsGradY();
        soap::BasisExpansion::coeff_t &coeff_grad_z = basex.getCoefficientsGradZ();
        
        int n = 3;
        int l = 5;
        int m = -4;
        int lm = l*l+l+m;
        
        ofs << boost::format("%1$+1.4f %2$+1.7e %3$+1.7e %4$+1.7e %5$+1.7e %6$+1.7e %7$+1.7e %8$+1.7e %9$+1.7e") 
            % R.getZ() % coeff(n,lm).real() % coeff(n,lm).imag()
            % coeff_grad_z(n,lm).real() % coeff_grad_z(n,lm).imag()
            % coeff_grad_y(n,lm).real() % coeff_grad_y(n,lm).imag()
            % coeff_grad_x(n,lm).real() % coeff_grad_x(n,lm).imag()
             << std::endl;        
    }    
    
    ofs.close();
    */
}

class TestBasisExpansionWithHeavisideCutoff : public ::testing::Test
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

TEST_F(TestBasisExpansionWithHeavisideCutoff, AdaptiveHeavisideCutoff) { 
    // CHECK WHETHER CUTOFF WAS ADJUSTED TO ADAPTIVE GAUSSIAN RADIAL BASIS
    double rc_heaviside = _basis->getCutoff()->getCutoff();
    EXPECT_DOUBLE_EQ(rc_heaviside, 6.3480129963106151);
}

TEST_F(TestBasisExpansionWithHeavisideCutoff, Gradients) {    
    
    typedef std::tuple<int, int, int, double, double, double> nlmxyz_t;
    
    std::vector<nlmxyz_t> nlmxyz_list = {
        nlmxyz_t{0,0,0,1.,0.5,-0.5},
        nlmxyz_t{0,1,0,1.,0.5,-0.5},
        nlmxyz_t{0,3,-1,-1.,-0.5,1.},
        nlmxyz_t{3,4,1,0.,0.3,-0.7},
        nlmxyz_t{7,2,-1,-0.5,3.,4.}
    };
    
    std::stringstream output;
    std::string output_ref = "n=0 l=0 m=0 x=+1.0000 y=+0.5000 z=-0.5000 Q_re=+9.4683121e-02 dQ_re=-1.8961753e-01 -9.4808765e-02 +9.4808765e-02 Q_im=+0.0000000e+00 dQ_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=0 l=1 m=0 x=+1.0000 y=+0.5000 z=-0.5000 Q_re=-4.7738237e-02 dQ_re=+1.0923622e-01 +5.4618111e-02 +4.0858364e-02 Q_im=-0.0000000e+00 dQ_im=+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 n=0 l=3 m=-1 x=-1.0000 y=-0.5000 z=+1.0000 Q_re=-1.2554214e-02 dQ_re=-2.8055842e-02 -2.0305028e-02 -5.0416312e-03 Q_im=+6.2771069e-03 dQ_im=+2.0305028e-02 -2.4017000e-03 +2.5208156e-03 n=3 l=4 m=1 x=+0.0000 y=+0.3000 z=-0.7000 Q_re=-2.1144554e-20 dQ_re=-1.1510939e-03 +8.4096906e-20 -4.4449962e-19 Q_im=-3.4532818e-04 dQ_im=+2.2521571e-20 +1.6936152e-03 -4.1418340e-03 n=7 l=2 m=-1 x=-0.5000 y=+3.0000 z=+4.0000 Q_re=-7.0360862e-05 dQ_re=+1.1432964e-04 +1.5835248e-04 +1.9354642e-04 Q_im=-4.2216517e-04 dQ_im=-1.5835248e-04 +8.0939315e-04 +1.1612785e-03 ";
    for (auto it = nlmxyz_list.begin(); it != nlmxyz_list.end(); ++it) {
        // Extract parameters
        int n = std::get<0>(*it);
        int l = std::get<1>(*it);
        int m = std::get<2>(*it);
        double x = std::get<3>(*it);
        double y = std::get<4>(*it);
        double z = std::get<5>(*it);
        int lm = l*l+l+m;
        
        // Position, radius, direction
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = R/r;
    
        // Weights
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        bool gradients = true;
        
        // Compute gradients
        ::testing::internal::CaptureStdout();
        soap::BasisExpansion basex(_basis);
        basex.computeCoefficients(r, d, weight0, weight_scale, sigma, gradients);
        ::testing::internal::GetCapturedStdout();
        
        // Extract results
        soap::BasisExpansion::coeff_t &coeff = basex.getCoefficients();
        soap::BasisExpansion::coeff_t &coeff_grad_x = basex.getCoefficientsGradX();
        soap::BasisExpansion::coeff_t &coeff_grad_y = basex.getCoefficientsGradY();
        soap::BasisExpansion::coeff_t &coeff_grad_z = basex.getCoefficientsGradZ();
        
        
        output
            << boost::format("n=%1$d l=%2$d m=%3$d ") % n % l % m
            << boost::format("x=%1$+1.4f y=%2$+1.4f z=%3$+1.4f ") % x % y % z
            << boost::format("Q_re=%1$+1.7e dQ_re=%2$+1.7e %3$+1.7e %4$+1.7e ")
                % coeff(n,lm).real() % coeff_grad_x(n,lm).real() % coeff_grad_y(n,lm).real() % coeff_grad_z(n,lm).real()
            << boost::format("Q_im=%1$+1.7e dQ_im=%2$+1.7e %3$+1.7e %4$+1.7e ") 
                % coeff(n,lm).imag() % coeff_grad_x(n,lm).imag() % coeff_grad_y(n,lm).imag() % coeff_grad_z(n,lm).imag()
            << std::flush;
    }
    
    // VERIFY
    EXPECT_EQ(output.str(), output_ref);    
    
    // GENERATE REFERENCE
    //std::cout << "\"" << output.str() << "\"" << std::endl;
    
    /*
    // MANUAL TESTING
    int N = 150;
    double dx = 0.05;
    
    std::string logfile = "tmp";
    std::ofstream ofs;
    ofs.open(logfile.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Bad file handle: " + logfile);
    }
    
    for (int i = 0; i < N; ++i) {        
        
        double x = i*dx;
        double y = 0.0;
        double z = 0.0;
    
        soap::vec R(x, y, z);
        double r = soap::linalg::abs(R);
        soap::vec d = R/r;
    
        double weight0 = 1.;
        double weight_scale = _basis->getCutoff()->calculateWeight(r);
        double sigma = 0.5;
        bool gradients = true;        
        
        // Compute
        ::testing::internal::CaptureStdout();
        soap::BasisExpansion basex(_basis);
        basex.computeCoefficients(r, d, weight0, weight_scale, sigma, gradients);
        ::testing::internal::GetCapturedStdout();
        
        // Extract
        soap::BasisExpansion::coeff_t &coeff = basex.getCoefficients();
        soap::BasisExpansion::coeff_t &coeff_grad_x = basex.getCoefficientsGradX();
        soap::BasisExpansion::coeff_t &coeff_grad_y = basex.getCoefficientsGradY();
        soap::BasisExpansion::coeff_t &coeff_grad_z = basex.getCoefficientsGradZ();
        
        int n = 5;
        int l = 1;
        int m = -1;
        int lm = l*l+l+m;
        
        // x (#1)  Qnlm_re (#2)  dQnlm_dx_re (#4)
        ofs << boost::format("%1$+1.4f %2$+1.7e %3$+1.7e %4$+1.7e %5$+1.7e %6$+1.7e %7$+1.7e %8$+1.7e %9$+1.7e") 
            % R.getX() % coeff(n,lm).real() % coeff(n,lm).imag()
            % coeff_grad_x(n,lm).real() % coeff_grad_x(n,lm).imag()
            % coeff_grad_y(n,lm).real() % coeff_grad_y(n,lm).imag()
            % coeff_grad_z(n,lm).real() % coeff_grad_z(n,lm).imag()           
             << std::endl;        
    }    
    
    ofs.close();
    */   
}








