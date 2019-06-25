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

class TypeEncoder
{
  public:
    typedef unsigned short int code_t;
    typedef std::map<std::string, code_t> encoder_t;
    typedef std::vector<std::string> order_t;
    TypeEncoder();
    ~TypeEncoder();
    void clear();
    int size() { return order.size(); }
    encoder_t::iterator begin() { return encoder.begin(); }
    encoder_t::iterator end() { return encoder.end(); }
    void list();
    boost::python::list getTypes();
    std::string decode(code_t code);
    code_t encode(std::string type);
    code_t encode(std::string type1, std::string type2);
    code_t encode(code_t code1, code_t code2);
    void add(std::string type);
    static void registerPython();
  private:
    encoder_t encoder {
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
   order_t order { "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I" };
};

struct TypeEncoderUI
{
    static void clear();
    static void add(std::string);
    static void list();
    static boost::python::list types();
    static TypeEncoder::code_t encode(std::string type1, std::string type2);
};

extern TypeEncoder ENCODER;

struct GradMap;
struct DMap
{
    typedef double dtype_t;
    //typedef float dtype_t;
    typedef ub::vector<dtype_t> vec_t;
    typedef ub::matrix<dtype_t> matrix_t;
    //typedef Eigen::VectorXf vec_t;
    typedef std::pair<TypeEncoder::code_t, vec_t*> channel_t;
    typedef std::vector<channel_t> dmap_t;
    typedef std::vector<GradMap*> pid_gradmap_t;
    DMap();
    DMap(std::string filter_type);
    ~DMap();
    dmap_t::iterator begin() { return dmap.begin(); }
    dmap_t::iterator end() { return dmap.end(); }
    pid_gradmap_t::iterator beginGradients() { return pid_gradmap.begin(); }
    pid_gradmap_t::iterator endGradients() { return pid_gradmap.end(); }
    int size() { return dmap.size(); }
    bpy::object val(int chidx, std::string np_dtype);
    dtype_t val() { return (*(dmap[0].second))(0); }
    void sort();
    void sortGradients();
    void multiply(double c);
    boost::python::list listChannels();
    void add(DMap *other);
    void add(DMap *other, double c);
    void slice(std::vector<int> &idcs);
    void addIgnoreGradients(DMap *other, double c);
    void addGradients(DMap *other, double c);
    bpy::object dotOuterNumpy(DMap *other, std::string np_dtype);
    void dotOuter(DMap *other, matrix_t &output);
    double dot(DMap *other);
    double dotFilter(DMap *other);
    DMap *dotGradLeft(DMap *other, double coeff, double power, DMap *res);
    void normalize();
    void convolve(int N, int L);
    void adapt(AtomicSpectrum *spectrum);
    void adapt(AtomicSpectrum::map_xnkl_t &map_xnkl);
    void adaptPidGradients(
        int center_pid, 
        AtomicSpectrum::map_pid_xnkl_t &map_pid_xnkl, 
        bool comoving_center=true);
    void adaptCoherent(AtomicSpectrum *spectrum);
    std::string getFilter() { return filter; }
    dmap_t dmap;
    pid_gradmap_t pid_gradmap;
    std::string filter;
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dmap;
        arch & pid_gradmap;
        arch & filter;
    }
};

struct GradMap
{
    typedef std::vector<DMap*> gradmap_t;
    GradMap();
    GradMap(int particle_id, std::string filter);
    GradMap(int particle_id, std::string filter, double gx, double gy, double gz);
    ~GradMap();
    void clear();
    void zero();
    void multiply(double c);
    void add(GradMap *other, double scale);
    int getPid() { return pid; }
    void adapt(AtomicSpectrum::map_xnkl_t &map_xnkl);
    DMap *get(int idx) { assert(idx <= 2); return gradmap[idx]; }
    DMap *x() { return gradmap[0]; }
    DMap *y() { return gradmap[1]; }
    DMap *z() { return gradmap[2]; }
    DMap::dtype_t xval() { return gradmap[0]->val(); }
    DMap::dtype_t yval() { return gradmap[1]->val(); }
    DMap::dtype_t zval() { return gradmap[2]->val(); }
    gradmap_t gradmap;
    int pid;
    std::string filter;
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & gradmap;
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
    void normalize();
    void convolve(int N, int L);
    void dot(DMapMatrix *other, matrix_t &output);
    void dotFilter(DMapMatrix *other, matrix_t &output);
    bpy::object dotNumpy(DMapMatrix *other, std::string np_dtype);
    bpy::object dotFilterNumpy(DMapMatrix *other, std::string np_dtype);
    void slicePython(bpy::list &py_idcs);
    void slice(std::vector<int> &idcs);
    void append(DMap *dmap);
    void append(Spectrum *spectrum);
    void appendCoherent(Spectrum *spectrum);
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
    typedef std::vector<dset_t*> views_t;
    DMapMatrixSet();
    DMapMatrixSet(bool set_as_view);
    DMapMatrixSet(std::string archfile);
    ~DMapMatrixSet();
    dset_t::iterator begin() { return dset.begin(); }
    dset_t::iterator end() { return dset.end(); }
    int size() { return dset.size(); }
    void clear();
    void slicePython(bpy::list &py_idcs);
    void slice(std::vector<int> &idcs);
    DMapMatrix *get(int idx) { return dset[idx]; }
    DMapMatrixSet *getView(boost::python::list idcs);
    void append(DMapMatrix *dmm);
    void extend(DMapMatrixSet *other);
    void save(std::string archfile);
    void load(std::string archfile);
    static void registerPython();
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & dset;
        arch & views;
        arch & is_view;
    }
  private:
    dset_t dset;
    views_t views;
    bool is_view;
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
    boost::python::object getItemNumpy(int idx) { return getBlockNumpy(idx, "float64"); }
    boost::python::object getBlockNumpy(int idx, std::string np_dtype="float64");
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


}

#endif /* _SOAP_DMAP_HPP_ */
