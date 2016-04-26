#include <iostream>
#include <vector>
#include <boost/format.hpp>
#include <soap/functions.hpp>

int main() {

    std::cout << "soapxx/test/functions" << std::endl;

    std::vector< soap::vec > r_list;    
    r_list.push_back(soap::vec(2.,0.,0.));
    r_list.push_back(soap::vec(0.,1.,0.));
    r_list.push_back(soap::vec(0.,0.,1.));
    r_list.push_back(soap::vec(std::sqrt(0.5),0.,std::sqrt(0.5)));
    r_list.push_back(soap::vec(-0.2,0.3,0.7));

    std::vector< std::pair<int,int> > lm_list;
    lm_list.push_back(std::pair<int,int>(1,0));
    lm_list.push_back(std::pair<int,int>(1,-1));
    lm_list.push_back(std::pair<int,int>(1,1));
    lm_list.push_back(std::pair<int,int>(2,0));
    lm_list.push_back(std::pair<int,int>(2,1));
    lm_list.push_back(std::pair<int,int>(2,2));
    
    // CAREFUL, THIS CAN LEAD TO NAN'S DEPENDING ON IMPLEMENTATION!
    double p = pow(0.,0);
    std::cout << p << std::endl;
    
    std::complex<double> q = soap::pow_nnan(std::complex<double>(0.,0.),0);
    std::cout << q << std::endl;

    for (int lm = 0; lm < lm_list.size(); ++lm) {
        int l = lm_list[lm].first;
        int m = lm_list[lm].second;

        std::cout << "====" << l << m << "====" << std::endl;

        for (int n = 0; n < r_list.size(); ++n) {
            soap::vec r = r_list[n];    
            
            std::vector<std::complex<double> > dylm = soap::GradSphericalYlm::eval(l, m, r);
            
            std::cout << "r = " << r << std::flush;
            std::cout << "=>" << std::flush;
            std::cout << dylm[0] << dylm[1] << dylm[2] << std::endl;        
        }        
    }
}




