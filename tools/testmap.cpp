#include <iostream>
#include <vector>
#include "soap/dmap.hpp"

using namespace soap;

int main(void) {

    // >>> for (int j=0; j<4; ++j) {
    // >>>     std::cout << "Running ..." << std::endl;
    // >>>     std::vector<tmpobj_t*> data;
    // >>>     for (int i=0; i<300000; ++i) {
    // >>>         if (i % 10000 == 0)
    // >>>             std::cout << "Allocating " << i << std::endl;
    // >>>         tmpobj_t *d = new tmpobj_t();
    // >>>         d->reserve(i % 10000, 1000.);
    // >>>         data.push_back(d);
    // >>>     }
    // >>>     int a;
    // >>>     std::cin >> a;
    // >>>     for (auto it=data.begin(); it!=data.end(); ++it) {
    // >>>         delete *it;
    // >>>     }
    // >>>     std::cout << "All gone" << std::endl;
    // >>>     std::cin >> a;
    // >>> } 

    // >>> std::vector<DMap*> data;
    // >>> for (int i=0; i<300000; ++i) {
    // >>>     if (i % 10000 == 0)
    // >>>         std::cout << "Allocating " << i << std::endl;
    // >>>     DMap *d = new DMap();
    // >>>     d->reserve(i % 10000, 1000.);
    // >>>     data.push_back(d);
    // >>> }
    // >>> int a;
    // >>> std::cin >> a;
    // >>> for (auto it=data.begin(); it!=data.end(); ++it) {
    // >>>     delete *it;
    // >>> }
    // >>> std::cout << "All gone" << std::endl;
    // >>> std::cin >> a;

    for (int j=0; j<4; ++j) {
        std::vector<DMapMatrixSet*> data;
        for (int i=0; i<10; ++i) {
            std::cout << "Loading " << i << std::endl;
            std::string path = "dset_rec_00.arch";
            auto *new_dset = new DMapMatrixSet(path);
            data.push_back(new_dset);
        }

        int a;
        std::cin >> a;

        for (auto it=data.begin(); it!=data.end(); ++it) {
            std::cout << "Deleting" << std::endl;
            delete *it;
        }

        std::cout << "Done";
        std::cin >> a;
    }
}
