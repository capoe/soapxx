#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <soap/functions.hpp>
#include <boost/format.hpp>

TEST(TestFunctions, ModifiedSphericalBessel1stKind) {

    int L = 10;
    double dr = 0.05;
    int N = 100;
    
    soap::ModifiedSphericalBessel1stKind mosbest(L);    
    
    mosbest.evaluate(0.0, true);
    EXPECT_DOUBLE_EQ(mosbest._in[0], 1.);
    EXPECT_DOUBLE_EQ(mosbest._in[1], 0.);
    EXPECT_DOUBLE_EQ(mosbest._in[10], 0.);    
    EXPECT_DOUBLE_EQ(mosbest._din[0], 0.);
    EXPECT_DOUBLE_EQ(mosbest._din[1], 1./3.);
    
    mosbest.evaluate(1., true);
    EXPECT_NEAR(mosbest._in[0], +1.1752012e+00, 1e-4);
    EXPECT_NEAR(mosbest._in[1], +3.6787944e-01, 1e-4);
    EXPECT_NEAR(mosbest._in[9], +1.5641127e-09, 1e-4);
    EXPECT_NEAR(mosbest._din[0], +3.6787944e-01, 1e-4);
    EXPECT_NEAR(mosbest._din[7], +3.5860451e-06, 1e-4);
    
    /*
    std::vector<double> r_list;
    r_list.push_back(0.);
    r_list.push_back(0.1);
    r_list.push_back(1.);

    for (int i = 0; i < r_list.size(); ++i) {
        mosbest.evaluate(r_list[i], true); 
        std::cout << "r = " << r_list[i] << std::endl;    
        for (int n = 0; n <= 10; ++n) {
            std::cout << boost::format("%1$2d %2$+1.7e %3$+1.7e") % n % mosbest._in[n] % mosbest._din[n] << std::endl;
        }
    }
    */

    /*
    for (int i = 0; i < N; ++i) {
        double ri = i*dr;
        // Compute
        mosbest.evaluate(ri, true);
        
        // Output
        std::cout << boost::format("%1$+1.7e ") % ri;
        for (int l = 0; l <= L; ++l) {
            std::cout << boost::format("l=%1$d %2$+1.7e %3$+1.7e ") % l % mosbest._in[l] % mosbest._din[l];
        }
        std::cout << std::endl;
    }
    */
    
        
}

