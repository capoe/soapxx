#ifndef _SOAP_DMAP_HPP
#define _SOAP_DMAP_HPP

#include <assert.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include "soap/types.hpp"
#include "soap/globals.hpp"
#include "soap/spectrum.hpp"
//#include "soap/linalg/Eigen/Dense"

namespace soap {

namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

struct DMap
{
    typedef double dtype_t;
    typedef ub::vector<dtype_t> vec_t;
    //typedef Eigen::VectorXf vec_t;
    typedef std::map<std::pair<std::string, std::string>, vec_t*> dmap_t;
    DMap();
    ~DMap();
    dmap_t::iterator begin() { return dmap.begin(); }
    dmap_t::iterator end() { return dmap.end(); }
    int size() { return dmap.size(); }
    void multiply(double c);
    double dot(DMap *other);
    void adapt(AtomicSpectrum *spectrum);
    dmap_t dmap;
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dmap;
    }
};

struct DMapMatrix
{
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> ub_matrix_t;
    //typedef Eigen::MatrixXf matrix_t;
    typedef ub::matrix<dtype_t> matrix_t;
    typedef std::vector<DMap*> dmm_t;
    DMapMatrix();
    ~DMapMatrix();
    void dot(DMapMatrix *other, ub_matrix_t &output);
    bpy::object dotNumpy(DMapMatrix *other, std::string np_dtype);
    void append(Spectrum *spectrum);
    void save(std::string archfile);
    void load(std::string archfile);
    dmm_t::iterator begin() { return dmm.begin(); }
    dmm_t::iterator end() { return dmm.end(); }
    int size() { return dmm.size(); }
    dmm_t dmm;
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dmm;
    }
};

struct BlockLaplacian
{
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> block_t;
    typedef std::vector<block_t*> blocks_t;
    BlockLaplacian();
    ~BlockLaplacian();
    void appendNumpy(boost::python::object &np_array, std::string np_dtype);
    void save(std::string archfile);
    void load(std::string archfile);
    int rows() { return n_rows; }
    int cols() { return n_cols; }
    blocks_t::iterator begin() { return blocks.begin(); }
    blocks_t::iterator end() { return blocks.end(); }
    void dotRight();
    void dotLeft();
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & blocks;
        arch & n_rows;
        arch & n_cols;
    }
    blocks_t blocks;
    int n_rows;
    int n_cols;
};

}

#endif /* _SOAP_DMAP_HPP_ */
