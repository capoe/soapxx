#include <iostream>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <soap/radialbasis.hpp>
#include <soap/options.hpp>

static const double EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  ASSERT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

class TestRadialBasisGaussian : public ::testing::Test
{
public:
    soap::Options _options;
    soap::RadialBasis *_radbasis;
    std::string _constructor_stdout;
    std::string _constructor_stdout_ref;
   
    virtual void SetUp() {

        _options.set("radialcutoff.Rc", 4.);
        _options.set("radialbasis.type", "gaussian");
        _options.set("radialbasis.mode", "adaptive");
        _options.set("radialbasis.N", 9);
        _options.set("angularbasis.L", 6);
        _options.set("radialbasis.sigma", 0.5);
        _options.set("radialbasis.integration_steps", 15);
        
        soap::RadialBasisFactory::registerAll();
        _radbasis = soap::RadialBasisOutlet().create(_options.get<std::string>("radialbasis.type"));
        ::testing::internal::CaptureStdout();
	    _radbasis->configure(_options);
        _constructor_stdout = ::testing::internal::GetCapturedStdout();

        _constructor_stdout_ref = 
            "Adjusted radial cutoff to 3.41181 based on sigma_0 = 0.5, L = 6, stride = 0.5\n"
            "Created 9 radial Gaussians at:\n"
            " r = +0.0000000e+00 (sigma = +5.0000000e-01) \n"
            " r = +2.5000000e-01 (sigma = +5.1887452e-01) \n"
            " r = +5.0943726e-01 (sigma = +5.7432939e-01) \n"
            " r = +7.9660196e-01 (sigma = +6.6727337e-01) \n"
            " r = +1.1302386e+00 (sigma = +8.0190914e-01) \n"
            " r = +1.5311932e+00 (sigma = +9.8559668e-01) \n"
            " r = +2.0239916e+00 (sigma = +1.2290136e+00) \n"
            " r = +2.6384983e+00 (sigma = +1.5466265e+00) \n"
            " r = +3.4118116e+00 (sigma = +1.9574676e+00) \n"
            "Radial basis overlap matrix\n"
            "+1.0000e+00 +9.5931e-01 +8.1266e-01 +5.9142e-01 +3.7675e-01 +2.2009e-01 +1.2395e-01 +6.9855e-02 +4.0214e-02 \n"
            "+9.5931e-01 +1.0000e+00 +9.3655e-01 +7.5104e-01 +5.1642e-01 +3.1649e-01 +1.8212e-01 +1.0292e-01 +5.8776e-02 \n"
            "+8.1266e-01 +9.3655e-01 +1.0000e+00 +9.2139e-01 +7.1532e-01 +4.7799e-01 +2.8856e-01 +1.6580e-01 +9.4361e-02 \n"
            "+5.9142e-01 +7.5104e-01 +9.2139e-01 +1.0000e+00 +9.1258e-01 +6.9606e-01 +4.5873e-01 +2.7546e-01 +1.5859e-01 \n"
            "+3.7675e-01 +5.1642e-01 +7.1532e-01 +9.1258e-01 +1.0000e+00 +9.0760e-01 +6.8560e-01 +4.4877e-01 +2.6901e-01 \n"
            "+2.2009e-01 +3.1649e-01 +4.7799e-01 +6.9606e-01 +9.0760e-01 +1.0000e+00 +9.0474e-01 +6.7975e-01 +4.4337e-01 \n"
            "+1.2395e-01 +1.8212e-01 +2.8856e-01 +4.5873e-01 +6.8560e-01 +9.0474e-01 +1.0000e+00 +9.0307e-01 +6.7638e-01 \n"
            "+6.9855e-02 +1.0292e-01 +1.6580e-01 +2.7546e-01 +4.4877e-01 +6.7975e-01 +9.0307e-01 +1.0000e+00 +9.0207e-01 \n"
            "+4.0214e-02 +5.8776e-02 +9.4361e-02 +1.5859e-01 +2.6901e-01 +4.4337e-01 +6.7638e-01 +9.0207e-01 +1.0000e+00 \n"
            "Radial basis Cholesky decomposition\n"
            "+1.0000e+00 +9.5931e-01 +8.1266e-01 +5.9142e-01 +3.7675e-01 +2.2009e-01 +1.2395e-01 +6.9855e-02 +4.0214e-02 \n"
            "+9.5931e-01 +2.8236e-01 +5.5590e-01 +6.5053e-01 +5.4894e-01 +3.7312e-01 +2.2389e-01 +1.2717e-01 +7.1531e-02 \n"
            "+8.1266e-01 +5.5590e-01 +1.7483e-01 +4.5265e-01 +5.9484e-01 +5.2457e-01 +3.6246e-01 +2.1931e-01 +1.2535e-01 \n"
            "+5.9142e-01 +6.5053e-01 +4.5265e-01 +1.4881e-01 +4.2606e-01 +5.7602e-01 +5.0881e-01 +3.5048e-01 +2.1189e-01 \n"
            "+3.7675e-01 +5.4894e-01 +5.9484e-01 +4.2606e-01 +1.4613e-01 +4.2698e-01 +5.7217e-01 +4.9862e-01 +3.4046e-01 \n"
            "+2.2009e-01 +3.7312e-01 +5.2457e-01 +5.7602e-01 +4.2698e-01 +1.5183e-01 +4.3733e-01 +5.7369e-01 +4.9171e-01 \n"
            "+1.2395e-01 +2.2389e-01 +3.6246e-01 +5.0881e-01 +5.7217e-01 +4.3733e-01 +1.6003e-01 +4.4949e-01 +5.7673e-01 \n"
            "+6.9855e-02 +1.2717e-01 +2.1931e-01 +3.5048e-01 +4.9862e-01 +5.7369e-01 +4.4949e-01 +1.6801e-01 +4.6027e-01 \n"
            "+4.0214e-02 +7.1531e-02 +1.2535e-01 +2.1189e-01 +3.4046e-01 +4.9171e-01 +5.7673e-01 +4.6027e-01 +1.7464e-01 \n"
            "Radial basis transformation matrix\n"
            "+1.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 +0.0000e+00 \n"
            "-3.3974e+00 +3.5415e+00 -7.5092e-15 -3.4133e-16 -5.3098e-16 +3.3945e-16 +9.0948e-31 -1.7053e-31 +1.3263e-31 \n"
            "+6.1541e+00 -1.1260e+01 +5.7197e+00 +6.7046e-15 -1.6762e-15 +0.0000e+00 -1.0755e-15 +1.2945e-16 -1.8179e-16 \n"
            "-7.8418e+00 +1.8770e+01 -1.7398e+01 +6.7199e+00 +7.4196e-15 -1.4839e-15 +3.2713e-15 -3.9375e-16 +5.5296e-16 \n"
            "+7.9966e+00 -2.2192e+01 +2.7442e+01 -1.9592e+01 +6.8431e+00 +4.3519e-15 -6.5278e-15 +1.0880e-15 -8.7222e-16 \n"
            "-7.1003e+00 +2.1400e+01 -3.0931e+01 +2.9604e+01 -1.9245e+01 +6.5864e+00 +9.6627e-15 -2.0131e-15 +9.8310e-16 \n"
            "+5.7853e+00 -1.8265e+01 +2.8772e+01 -3.2217e+01 +2.8125e+01 -1.7999e+01 +6.2488e+00 +3.3970e-15 -8.4926e-16 \n"
            "-4.4840e+00 +1.4517e+01 -2.3973e+01 +2.9234e+01 -2.9840e+01 +2.5665e+01 -1.6718e+01 +5.9520e+00 +5.8750e-16 \n"
            "+3.3731e+00 -1.1074e+01 +1.8762e+01 -2.3966e+01 +2.6611e+01 -2.6745e+01 +2.3425e+01 -1.5687e+01 +5.7261e+00 \n";
        
    }

    virtual void TearDown() {
        delete _radbasis;
        _radbasis = NULL;
    }
};

TEST_F(TestRadialBasisGaussian, Constructor) {
    EXPECT_EQ(_radbasis->identify(), "gaussian");
    EXPECT_EQ(_radbasis->N(), 9);

    // GENERATE REFERENCE
    /*
    std::vector<std::string> fields_ref;
    boost::split(fields_ref, _constructor_stdout, boost::is_any_of("\n"));
    for (auto f = fields_ref.begin(); f != fields_ref.end(); ++f) {
        std::cout << "        \"" << (*f) << "\\n\"" << std::endl;
    }
    */

    // COMPARE CONSTRUCTOR STDOUT
    std::vector<std::string> fields;
    std::vector<std::string> fields_ref;

    boost::replace_all(_constructor_stdout, "\n", " ");
    boost::replace_all(_constructor_stdout_ref, "\n", " ");
    boost::replace_all(_constructor_stdout, "  ", " ");
    boost::replace_all(_constructor_stdout_ref, "  ", " ");
    boost::split(fields, _constructor_stdout, boost::is_any_of(" "));
    boost::split(fields_ref, _constructor_stdout_ref, boost::is_any_of(" "));

    auto f = fields.begin();
    auto fref = fields_ref.begin();
    for ( ; f != fields.end() && fref != fields_ref.end(); ++f, ++fref) {
        EXPECT_EQ((*f), (*fref));
    }
}

TEST_F(TestRadialBasisGaussian, computeCoefficientsGradients) {
    int N = _radbasis->N();
    int L = _options.get<int>("angularbasis.L");

    soap::RadialBasis::radcoeff_t Gnl = soap::RadialBasis::radcoeff_zero_t(N,L+1);
    soap::RadialBasis::radcoeff_t dGnl_dx = soap::RadialBasis::radcoeff_zero_t(N,L+1);
    soap::RadialBasis::radcoeff_t dGnl_dy = soap::RadialBasis::radcoeff_zero_t(N,L+1);
    soap::RadialBasis::radcoeff_t dGnl_dz = soap::RadialBasis::radcoeff_zero_t(N,L+1);

    /*
    double sigma = 0.5;
    soap::vec d(0.,1.,0.);
    double r = 0.;
    _radbasis->computeCoefficients(d, r, sigma, Gnl, dGnl_dx, dGnl_dy, dGnl_dz);
    */

}

TEST_F(TestRadialBasisGaussian, NumericalIntegrationBessel) {

    // CHECK NUMERICAL INTEGRATION
    double ai = 2.;
    double ak = 2.;
    double beta_ik = ai+ak;
    double ri = 1.;
    double rk = 1.;
    double rho_ik = ak*rk/beta_ik;
    int L_plus_1 = 7;
    int integration_steps = 15;

    /*
    for (int n = 0; n < 100; ++n) {
        ri = n*0.05;
        std::vector<double> integrals;
        integrals.resize(L_plus_1, 0.);
        std::vector<double> integrals_derivative;
        integrals_derivative.resize(L_plus_1, 0.);
        soap::compute_integrals_il_expik_r2_dr(
            ai, ri, beta_ik, rho_ik, L_plus_1, integration_steps,
            &integrals, &integrals_derivative);
        
        std::cout << boost::format("%1$+1.7e ") % ri;
        for (int l = 0; l < L_plus_1; ++l) {
            std::cout << boost::format("l=%1$d %2$+1.7e %3$+1.7e ") % l % integrals[l] % integrals_derivative[l];
        }
        std::cout << std::endl;
    }
    */

    std::vector<double> r_list{0., 0.5, 3.};

    std::vector<double> samples;
    std::vector<double> samples_ref{
        +3.2927839e-01, +0.0000000e+00, +0.0000000e+00, 
        +5.5035011e-01, +2.7537077e-02, +2.6313175e-04,
        +2.3985116e+05, +1.8595085e+05, +9.9749558e+04};
        
    std::vector<double> samples_d;
    std::vector<double> samples_d_ref{
        +3.7086702e-01, +0.0000000e+00, +0.0000000e+00,
        +9.6300224e-01, +5.1400603e-01, +6.0036046e-02,
        +1.8193830e+06, +1.6815557e+06, +1.2788038e+06};

    for (auto &ri : r_list) {
        std::vector<double> integrals;
        integrals.resize(L_plus_1, 0.);
        std::vector<double> integrals_derivative;
        integrals_derivative.resize(L_plus_1, 0.);

        soap::compute_integrals_il_expik_r2_dr(
            ai, ri, beta_ik, rho_ik, L_plus_1, integration_steps,
            &integrals, &integrals_derivative);

        samples.push_back(integrals[0]);
        samples.push_back(integrals[3]);
        samples.push_back(integrals[6]);

        samples_d.push_back(integrals_derivative[1]);
        samples_d.push_back(integrals_derivative[2]);
        samples_d.push_back(integrals_derivative[4]);
    }

    for (int i = 0; i < samples.size(); ++i) {
        EXPECT_NEAR_RELATIVE(samples[i], samples_ref[i], 1e-7);
    }
    for (int i = 0; i < samples_d.size(); ++i) {
        EXPECT_NEAR_RELATIVE(samples_d[i], samples_d_ref[i], 1e-7);
    }
    
}




