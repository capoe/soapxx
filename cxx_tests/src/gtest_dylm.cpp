#include <iostream>
#include <vector>
#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <soap/functions.hpp>

TEST(TestFunctions, PowerExponentZero) {
    // CAREFUL, THIS CAN LEAD TO NAN'S DEPENDING ON IMPLEMENTATION!
    double p = pow(0.,0);
    std::complex<double> q = soap::pow_nnan(std::complex<double>(0.,0.),0);
    
    EXPECT_EQ(p, 1);
    EXPECT_EQ(q, std::complex<double>(1.,0.));
}

TEST(TestFunctions, GradientYlm) {

    // SETUP
    std::vector< soap::vec > r_list;    
    r_list.push_back(soap::vec(2.,0.,0.));
    r_list.push_back(soap::vec(0.,1.,0.));
    r_list.push_back(soap::vec(0.,0.,1.));
    r_list.push_back(soap::vec(std::sqrt(0.5),0.,std::sqrt(0.5)));
    r_list.push_back(soap::vec(-0.2,0.3,0.7));
    r_list.push_back(soap::vec(-0.2,-0.4,-0.1));

    std::vector< std::pair<int,int> > lm_list;
    lm_list.push_back(std::pair<int,int>(1,0));
    lm_list.push_back(std::pair<int,int>(1,1));
    lm_list.push_back(std::pair<int,int>(2,0));
    lm_list.push_back(std::pair<int,int>(2,1));
    lm_list.push_back(std::pair<int,int>(2,-1));
    lm_list.push_back(std::pair<int,int>(2,2));
    lm_list.push_back(std::pair<int,int>(2,-2));

    std::vector<std::complex<double> > results;
    results.push_back(std::complex<double>(-1.4959138e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+2.4430126e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-1.8319660e-33, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.9918275e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+4.8860251e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.4430126e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+2.4430126e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+1.4011873e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.1017810e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+1.3011025e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-1.0154458e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.0308916e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+1.0154458e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(-6.4769779e-34, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, -1.7274707e-01));
    results.push_back(std::complex<double>(+1.0577708e-17, -0.0000000e+00));
    results.push_back(std::complex<double>(-3.4549415e-01, +2.1154717e-17));
    results.push_back(std::complex<double>(+2.1155415e-17, -2.5907484e-33));
    results.push_back(std::complex<double>(+1.2953528e-33, +2.1155415e-17));
    results.push_back(std::complex<double>(-3.4549415e-01, +6.9868116e-22));
    results.push_back(std::complex<double>(-6.9868116e-22, -3.4549415e-01));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(-1.7274707e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, -3.4549415e-01));
    results.push_back(std::complex<double>(+1.7274707e-01, -0.0000000e+00));
    results.push_back(std::complex<double>(-4.1046975e-01, -4.2462388e-02));
    results.push_back(std::complex<double>(-4.2462388e-02, -3.7508443e-01));
    results.push_back(std::complex<double>(-9.9078905e-02, +1.4861836e-01));
    results.push_back(std::complex<double>(-6.1032432e-01, +2.8721145e-01));
    results.push_back(std::complex<double>(+2.8721145e-01, -1.7950715e-01));
    results.push_back(std::complex<double>(+7.1802861e-02, +1.4360572e-01));
    results.push_back(std::complex<double>(-3.5475869e-33, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+5.7936491e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(-4.3445409e-49, +0.0000000e+00));
    results.push_back(std::complex<double>(-7.0951738e-33, +0.0000000e+00));
    results.push_back(std::complex<double>(+1.1587298e-16, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(-6.6904654e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+6.6904654e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+4.8244079e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-7.2366119e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+4.4798074e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+8.5820834e-02, +0.0000000e+00));
    results.push_back(std::complex<double>(+1.7164167e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-8.5820834e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+2.3652473e-17, -0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, -2.3652473e-17));
    results.push_back(std::complex<double>(-3.8627420e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-4.7304947e-17, +5.7930895e-33));
    results.push_back(std::complex<double>(+5.7930895e-33, +4.7304947e-17));
    results.push_back(std::complex<double>(-4.7303384e-17, -7.7254840e-01));
    results.push_back(std::complex<double>(-7.7254840e-01, +1.5622986e-21));
    results.push_back(std::complex<double>(-1.5622986e-21, -7.7254840e-01));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(-3.3449648e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(-0.0000000e+00, -5.4627422e-01));
    results.push_back(std::complex<double>(+3.3449648e-17, -0.0000000e+00));
    results.push_back(std::complex<double>(-7.5968600e-01, -1.6881911e-01));
    results.push_back(std::complex<double>(-1.6881911e-01, -6.1900340e-01));
    results.push_back(std::complex<double>(-1.4470209e-01, +2.1705314e-01));
    results.push_back(std::complex<double>(+2.2773536e-01, -2.8028967e-01));
    results.push_back(std::complex<double>(-2.8028967e-01, -1.9269915e-01));
    results.push_back(std::complex<double>(+6.6568797e-01, +1.3313759e+00));
    results.push_back(std::complex<double>(-2.3652473e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -2.3652473e-17));
    results.push_back(std::complex<double>(+3.8627420e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+4.7304947e-17, +5.7930895e-33));
    results.push_back(std::complex<double>(-5.7930895e-33, +4.7304947e-17));
    results.push_back(std::complex<double>(+4.7303384e-17, -7.7254840e-01));
    results.push_back(std::complex<double>(+7.7254840e-01, +1.5622986e-21));
    results.push_back(std::complex<double>(+1.5622986e-21, -7.7254840e-01));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+3.3449648e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -5.4627422e-01));
    results.push_back(std::complex<double>(-3.3449648e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+7.5968600e-01, -1.6881911e-01));
    results.push_back(std::complex<double>(+1.6881911e-01, -6.1900340e-01));
    results.push_back(std::complex<double>(+1.4470209e-01, +2.1705314e-01));
    results.push_back(std::complex<double>(-2.2773536e-01, -2.8028967e-01));
    results.push_back(std::complex<double>(+2.8028967e-01, -1.9269915e-01));
    results.push_back(std::complex<double>(-6.6568797e-01, +1.3313759e+00));
    results.push_back(std::complex<double>(+1.4482963e-33, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +3.8627420e-01));
    results.push_back(std::complex<double>(-2.3652473e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+9.4606768e-17, +7.7254840e-01));
    results.push_back(std::complex<double>(-8.6895864e-33, -4.7304947e-17));
    results.push_back(std::complex<double>(+4.7304947e-17, -5.7929938e-33));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +0.0000000e+00));
    results.push_back(std::complex<double>(+2.7313711e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, +5.4627422e-01));
    results.push_back(std::complex<double>(-2.7313711e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.6930668e-01, +3.2557971e-01));
    results.push_back(std::complex<double>(-3.4366747e-01, -1.7685812e-01));
    results.push_back(std::complex<double>(+7.0341296e-02, +1.6881911e-01));
    results.push_back(std::complex<double>(-1.1561949e+00, -9.1094143e-01));
    results.push_back(std::complex<double>(+6.3065176e-01, +3.8539830e-01));
    results.push_back(std::complex<double>(-2.1021725e-01, +2.8028967e-01));
    results.push_back(std::complex<double>(+1.4482963e-33, -0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -3.8627420e-01));
    results.push_back(std::complex<double>(-2.3652473e-17, +0.0000000e+00));
    results.push_back(std::complex<double>(+9.4606768e-17, -7.7254840e-01));
    results.push_back(std::complex<double>(-8.6895864e-33, +4.7304947e-17));
    results.push_back(std::complex<double>(+4.7304947e-17, +5.7929938e-33));
    results.push_back(std::complex<double>(+0.0000000e+00, -0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -0.0000000e+00));
    results.push_back(std::complex<double>(+2.7313711e-01, -0.0000000e+00));
    results.push_back(std::complex<double>(+0.0000000e+00, -5.4627422e-01));
    results.push_back(std::complex<double>(-2.7313711e-01, +0.0000000e+00));
    results.push_back(std::complex<double>(-2.6930668e-01, -3.2557971e-01));
    results.push_back(std::complex<double>(-3.4366747e-01, +1.7685812e-01));
    results.push_back(std::complex<double>(+7.0341296e-02, -1.6881911e-01));
    results.push_back(std::complex<double>(-1.1561949e+00, +9.1094143e-01));
    results.push_back(std::complex<double>(+6.3065176e-01, -3.8539830e-01));
    results.push_back(std::complex<double>(-2.1021725e-01, -2.8028967e-01));   
    
    // COMPUTE
    int res_idx = -1;
    for (int lm = 0; lm < lm_list.size(); ++lm) {
        int l = lm_list[lm].first;
        int m = lm_list[lm].second;

        for (int n = 0; n < r_list.size(); ++n) {
            soap::vec r = r_list[n];    
            std::vector<std::complex<double> > dylm = soap::GradSphericalYlm::eval(l, m, r);

            // VERIFY
            res_idx += 1;
            EXPECT_NEAR(dylm[0].real(), results[res_idx].real(), 1e-7);
            EXPECT_NEAR(dylm[0].imag(), results[res_idx].imag(), 1e-7);
            res_idx += 1;
            EXPECT_NEAR(dylm[1].real(), results[res_idx].real(), 1e-7);
            EXPECT_NEAR(dylm[1].imag(), results[res_idx].imag(), 1e-7);            
            res_idx += 1;
            EXPECT_NEAR(dylm[2].real(), results[res_idx].real(), 1e-7);
            EXPECT_NEAR(dylm[2].imag(), results[res_idx].imag(), 1e-7);
        }        
    }
}




