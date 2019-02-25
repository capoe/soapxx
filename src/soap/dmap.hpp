#ifndef _SOAP_DMAP_HPP
#define _SOAP_DMAP_HPP

#include <assert.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/list.hpp>

#include "soap/types.hpp"
#include "soap/globals.hpp"
#include "soap/spectrum.hpp"
//#include "soap/linalg/Eigen/Dense"

#include "soap/cutoff.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

struct DMap
{
    typedef float dtype_t;
    //typedef double dtype_t;
    typedef ub::vector<dtype_t> vec_t;
    //typedef Eigen::VectorXf vec_t;
    //typedef std::map<std::pair<std::string, std::string>, vec_t*> dmap_t;
    typedef std::pair<unsigned short int, vec_t*> channel_t;
    typedef std::vector<channel_t> dmap_t;
    DMap();
    ~DMap();
    dmap_t::iterator begin() { return dmap.begin(); }
    dmap_t::iterator end() { return dmap.end(); }
    int size() { return dmap.size(); }
    void sort();
    void multiply(double c);
    double dot(DMap *other);
    void add(DMap *other);
    double dotFilter(DMap *other);
    void adapt(AtomicSpectrum *spectrum);
    std::string getFilter() { return filter; }
    dmap_t dmap;
    std::string filter;
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dmap;
        arch & filter;
    }
};

class DMapMatrix
{
  public:
    typedef double dtype_t;
    //typedef Eigen::MatrixXf matrix_t;
    typedef ub::matrix<dtype_t> matrix_t;
    typedef ub::vector<dtype_t> vec_t;
    typedef std::vector<DMap*> dmm_t;
    typedef std::map<std::string, DMapMatrix*> views_t;
    DMapMatrix();
    DMapMatrix(std::string archfile);
    ~DMapMatrix();
    void clear();
    void sum();
    void dot(DMapMatrix *other, matrix_t &output);
    void dotFilter(DMapMatrix *other, matrix_t &output);
    bpy::object dotNumpy(DMapMatrix *other, std::string np_dtype);
    bpy::object dotFilterNumpy(DMapMatrix *other, std::string np_dtype);
    void append(Spectrum *spectrum);
    void save(std::string archfile);
    void load(std::string archfile);
    DMap *getRow(int idx) { return dmm[idx]; }
    void addView(std::string filter);
    DMapMatrix *getView(std::string filter);
    dmm_t::iterator begin() { return dmm.begin(); }
    dmm_t::iterator end() { return dmm.end(); }
    int rows() { return dmm.size(); }
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dmm;
        arch & views;
        arch & is_view;
    }
  private:
    DMapMatrix(bool set_as_view);
    dmm_t dmm;
    views_t views;
    bool is_view;
};

void dmm_inner_product(
    DMapMatrix &AX, 
    DMapMatrix &BX, 
    double power, 
    bool filter,
    DMapMatrix::matrix_t &output);

class DMapMatrixSet
{
  public:
    typedef std::vector<DMapMatrix*> dset_t;
    DMapMatrixSet();
    DMapMatrixSet(std::string archfile);
    ~DMapMatrixSet();
    dset_t::iterator begin() { return dset.begin(); }
    dset_t::iterator end() { return dset.end(); }
    int size() { return dset.size(); }
    DMapMatrix *get(int idx) { return dset[idx]; }
    void append(DMapMatrix *dmm);
    void save(std::string archfile);
    void load(std::string archfile);
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dset;
    }
  private:
    dset_t dset;
};

struct BlockLaplacian
{
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> block_t;
    typedef std::vector<block_t*> blocks_t;
    BlockLaplacian();
    BlockLaplacian(std::string archfile);
    ~BlockLaplacian();
    block_t *addBlock(int n_rows_block, int n_cols_block);
    void appendNumpy(boost::python::object &np_array, std::string np_dtype);
    void save(std::string archfile);
    void load(std::string archfile);
    int rows() { return n_rows; }
    int cols() { return n_cols; }
    blocks_t::iterator begin() { return blocks.begin(); }
    blocks_t::iterator end() { return blocks.end(); }
    bpy::object dotNumpy(bpy::object &np_other, std::string np_dtype);
    void dot(block_t &other, block_t &output);
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

class Proto
{
  public:
    typedef double dtype_t;
    typedef std::vector<BlockLaplacian*> Gnab_t;
    typedef ub::matrix<dtype_t> matrix_t;
    Proto();
    ~Proto();
    void parametrize(DMapMatrix &AX, DMapMatrix &BX, BlockLaplacian &DAB);
    bpy::object projectPython(DMapMatrix *AX, DMapMatrix *BX, 
        double xi, std::string np_dtype);
    void project(DMapMatrix *AX, DMapMatrix *BX, double xi, matrix_t &output);
    static void registerPython(); 
  private:
    Gnab_t Gnab;
    CutoffFunction *cutoff;
    DMapMatrix *AX;
    DMapMatrix *BX;
};

static std::map<std::string, unsigned short int> ELEMENT_ENCODING {
  { "H" ,   0 },
  { "C" ,   1 },
  { "N" ,   2 },
  { "O" ,   3 },
  { "F" ,   4 },
  { "P" ,   5 },
  { "S" ,   6 },
  { "Cl",   7 },
  { "Br",   8 },
  { "I" ,   9 }
};


}

#endif /* _SOAP_DMAP_HPP_ */
