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

class TestOptions : public ::testing::Test
{
public:

    soap::Options _options;
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
        
    }

    virtual void TearDown() {
        ;
    }
};

TEST_F(TestOptions, InterpretBoolean) {
    _options.set("spectrum.gradients", true);
    bool gradients = _options.get<bool>("spectrum.gradients");
    //std::cout << "true == " << gradients << std::endl;
    EXPECT_EQ(gradients, true);

    _options.set("spectrum.gradients", false);
    gradients = _options.get<bool>("spectrum.gradients");
    //std::cout << "false == " << gradients << std::endl;
    EXPECT_EQ(gradients, false);

    _options.set("spectrum.gradients", "false");
    try {
        gradients = _options.get<bool>("spectrum.gradients");
        std::cout << "false ==" << gradients << std::endl;
        assert(false);
    } catch (const std::exception &e) {
        EXPECT_STREQ(e.what(), "invalid type: wrong or missing type in spectrum.gradients");
    } catch (...) {
        assert(false);
    }
}

