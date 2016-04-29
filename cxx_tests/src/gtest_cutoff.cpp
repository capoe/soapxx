#include <iostream>
#include <fstream>
#include <vector>
#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <soap/cutoff.hpp>
#include <soap/options.hpp>

class TestCutoffShiftedCosine : public ::testing::Test
{
public:
    soap::Options _options;
    soap::CutoffFunction *_cutoff;
    std::string _constructor_stdout;
    std::string _constructor_stdout_ref;
   
    virtual void SetUp() {
        _options.set("radialcutoff.type", "shifted-cosine");
	    _options.set("radialcutoff.Rc", 4.);
	    _options.set("radialcutoff.Rc_width", 0.5);
	    _options.set("radialcutoff.center_weight", 1.);
        
        soap::CutoffFunctionFactory::registerAll();
        _cutoff = soap::CutoffFunctionOutlet().create(_options.get<std::string>("radialcutoff.type"));
        ::testing::internal::CaptureStdout();
	    _cutoff->configure(_options);
        _constructor_stdout = ::testing::internal::GetCapturedStdout();
        _constructor_stdout_ref = "Weighting function with Rc = 4, _Rc_width = 0.5, central weight = 1\n";
    }

    virtual void TearDown() {
        delete _cutoff;
        _cutoff = NULL;
    }
};

TEST_F(TestCutoffShiftedCosine, Constructor) {
    EXPECT_EQ(_cutoff->identify(), "shifted-cosine");
    EXPECT_DOUBLE_EQ(_cutoff->getCenterWeight(), 1.);
    EXPECT_DOUBLE_EQ(_cutoff->getCutoffWidth(), 0.5);
    EXPECT_DOUBLE_EQ(_cutoff->getCutoff(), 4.);

    EXPECT_EQ(_constructor_stdout, _constructor_stdout_ref);
}

TEST_F(TestCutoffShiftedCosine, Weight) {

    double r0 = 3.5;
    double r1 = 4.;
    double dr = r1-r0;

    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(r1+1.), -1e-10);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(r1+0.00001), -1e-10);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(r1), 0.);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(r1-2.*dr/3.), 0.75);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(r0), 1.);

    /*
    std::vector<double> r_list;
    r_list.push_back(0.);
    for (int ridx = 0; ridx < 100; ++ridx) {
        double r = 3.45+ridx*0.01;
        r_list.push_back(r);
    }

    for (int rd = 0; rd < r_list.size(); ++rd) {
        double r = r_list[rd];
        double w = _cutoff->calculateWeight(r);

        std::cout << r << " " << w << std::endl;
    }
    */
}

TEST_F(TestCutoffShiftedCosine, GradientWeight) {
    soap::vec d1(std::sqrt(0.5),0.,std::sqrt(0.5));

    soap::vec gw;
    double abs_gw;
    soap::vec norm_gw;
    double diff;

    double r0 = 3.5;
    double r1 = 4.;
    double dr = r1-r0;

    gw = _cutoff->calculateGradientWeight(r0, d1);
    abs_gw = soap::linalg::abs(gw);
    EXPECT_DOUBLE_EQ(abs_gw, 0.);

    gw = _cutoff->calculateGradientWeight(r1, d1);
    abs_gw = soap::linalg::abs(gw);
    EXPECT_NEAR(abs_gw, 0., 1e-10);

    gw = _cutoff->calculateGradientWeight(r1-5./6.*dr, d1);
    abs_gw = soap::linalg::abs(gw);
    norm_gw = gw/abs_gw;
    diff = soap::linalg::abs(norm_gw-d1);
    EXPECT_NEAR(abs_gw, 0.5*0.5*M_PI/dr, 1e-10);
    EXPECT_NEAR(diff, 2., 1e-10);

    /*
    std::vector< std::pair<double, soap::vec> > rd_list;
    rd_list.push_back(std::pair<double, soap::vec>(0., d1));
    for (int rd = 0; rd < 100; ++rd) {
        double r = 3.45+rd*0.01;
        rd_list.push_back(std::pair<double, soap::vec>(r, d1));
    }

    for (int rd = 0; rd < rd_list.size(); ++rd) {
        double r = rd_list[rd].first;
        soap::vec d = rd_list[rd].second;
        soap::vec grad = _cutoff->calculateGradientWeight(r, d);

        double abs_grad = soap::linalg::abs(grad);
        std::cout << r << " " << abs_grad << std::endl;
    }
    */
    
    /*
    // MANUAL TESTING
    int N = 100;
    double dx = 0.05;
    
    std::string logfile = "tmp_cutoff";
    std::ofstream ofs;
    ofs.open(logfile.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Bad file handle: " + logfile);
    }
    
    for (int i = 0; i < N; ++i) {
        soap::vec d(1.,0.,0.);
        double r = i*dx;
        
        double w = _cutoff->calculateWeight(r);
        soap::vec w_grad = _cutoff->calculateGradientWeight(r, d);        
        
        ofs << boost::format("%1$+1.4f %2$+1.7e %3$+1.7e %4$+1.7e %5$+1.7e") 
            % r % w
            % w_grad.getX()
            % w_grad.getY()
            % w_grad.getZ()
             << std::endl;
        
    }
    */
}

class TestCutoffHeaviside : public ::testing::Test
{
public:
    soap::Options _options;
    soap::CutoffFunction *_cutoff;
    std::string _constructor_stdout;
    std::string _constructor_stdout_ref;
   
    virtual void SetUp() {
        _options.set("radialcutoff.type", "heaviside");
	    _options.set("radialcutoff.Rc", 4.);
	    _options.set("radialcutoff.Rc_width", 0.5);
	    _options.set("radialcutoff.center_weight", 1.);
        
        soap::CutoffFunctionFactory::registerAll();
        _cutoff = soap::CutoffFunctionOutlet().create(_options.get<std::string>("radialcutoff.type"));
        ::testing::internal::CaptureStdout();
	    _cutoff->configure(_options);
        _constructor_stdout = ::testing::internal::GetCapturedStdout();
        _constructor_stdout_ref = "Weighting function with Rc = 4, central weight = 1\n";
    }

    virtual void TearDown() {
        delete _cutoff;
        _cutoff = NULL;
    }
};

TEST_F(TestCutoffHeaviside, Constructor) {
    EXPECT_EQ(_cutoff->identify(), "heaviside");
    EXPECT_DOUBLE_EQ(_cutoff->getCenterWeight(), 1.);
    EXPECT_DOUBLE_EQ(_cutoff->getCutoffWidth(), 0.5);
    EXPECT_DOUBLE_EQ(_cutoff->getCutoff(), 4.);
    EXPECT_EQ(_constructor_stdout, _constructor_stdout_ref);
}

TEST_F(TestCutoffHeaviside, Weight) {
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(1.), 1.);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(3.999), 1.);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(4.), 1.);
    EXPECT_DOUBLE_EQ(_cutoff->calculateWeight(4.0001), -1e-10);
}






