#include <iostream>
#include <vector>
#include <soap/functions.hpp>
#include <boost/format.hpp>

int main() {

    std::cout << "soapxx/test/functions" << std::endl;

    int N = 10;
    std::vector<double> r_list;
    r_list.push_back(0.);
    r_list.push_back(0.1);
    r_list.push_back(1.);

    soap::ModifiedSphericalBessel1stKind sph_in(N);

    for (int i = 0; i < r_list.size(); ++i) {
        sph_in.evaluate(r_list[i], true); 
        std::cout << "r = " << r_list[i] << std::endl;    
        for (int n = 0; n <= 10; ++n) {
            std::cout << boost::format("%1$2d %2$+1.7e %3$+1.7e") % n % sph_in._in[n] % sph_in._din[n] << std::endl;
        }
    }

}
