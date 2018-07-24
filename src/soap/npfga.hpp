#ifndef SOAP_NPFGA_HPP
#define SOAP_NPFGA_HPP

#include "soap/types.hpp"
#include <map>
#include <cmath>

namespace soap { namespace npfga {

namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

typedef double dtype_t;
typedef ub::matrix<dtype_t> matrix_t;
typedef ub::zero_matrix<dtype_t> zero_matrix_t;

class Operator;
class FNode;

struct Instruction
{
    typedef std::vector<Instruction*> args_t;
    Instruction(Operator *oper, std::string tagstr, double pow, double prefactor);
    Instruction(Operator *op, std::vector<Instruction*> &args_in);
    Instruction() : op(NULL), tag(""), power(1.0), prefactor(1.0), is_root(false) {;}
    ~Instruction();
    Instruction *deepCopy(Instruction *);
    std::string getBasename();
    std::string stringify(std::string format="");
    void simplifyInstruction();
    void raiseToPower(double p);
    void multiplyBy(double c) { prefactor *= c; }
    bool containsConstant();
    Operator *op;
    args_t args;
    bool is_root;
    std::string tag;
    std::string expr;
    double power;
    double prefactor;
};

struct FNodeStats
{
    FNodeStats() : cov(-1.), p(-1.) {;}
    ~FNodeStats() {;}
    double cov;
    double p;
};

struct FNodeDimension
{
    typedef std::map<std::string, double> dim_map_t;
    FNodeDimension() {;}
    FNodeDimension(const FNodeDimension &other) { dim_map = other.dim_map; }
    FNodeDimension(std::string dimstr);
    FNodeDimension(dim_map_t &dim_map_in);
    ~FNodeDimension() {;}
    std::string calculateString();
    void eraseZeros();
    void raiseToPower(double p);
    void add(FNodeDimension &other);
    void subtract(FNodeDimension &other);
    void addFactor(const std::string &unit, const double &power);
    void subtractFactor(const std::string &unit, const double &power);
    bool matches(FNodeDimension &other, bool check_reverse=true);
    bool isDimensionless() { return (dim_map.size() == 0); }
    dim_map_t dim_map;
};

struct FNodeCheck
{
    FNodeCheck(double min_power, double max_power)
        : min_pow(min_power), max_pow(max_power) {;}
    bool check(FNode* fnode);
    double min_pow;
    double max_pow;
};

class FNode 
{
  public:
    FNode(Operator *op, std::string varname, std::string varplus, 
        std::string varzero, std::string dimstr, bool is_root);
    FNode(Operator *op, FNode *par1, bool maybe_negative, bool maybe_zero);
    FNode(Operator *op, FNode *par1, FNode *par2, bool maybe_negative, bool maybe_zero);
    ~FNode();
    FNodeDimension &getDimension() { return dimension; }
    std::string calculateTag();
    Instruction *getOrCalculateInstruction();
    int getGenerationIdx() { return generation_idx; }
    bool isDimensionless() { return dimension.isDimensionless(); }
    bool notNegative() { return !maybe_negative; }
    bool notZero() { return !maybe_zero; }
    double &getValue() { return value; }
    double &evaluate();
    void seed(double v) { value = prefactor*v; }
    std::vector<FNode*> &getParents() { return parents; }
    std::string calculateDimString() { return dimension.calculateString(); }
    Operator *getOperator() { return op; }
    bool containsOperator(std::string optag);
  private:
    int generation_idx;
    bool is_root;
    bool maybe_negative;
    bool maybe_zero;
    double prefactor;
    double value;
    std::string tag;
    std::string expr;
    Operator *op;
    std::vector<FNode*> parents;
    FNodeDimension dimension;
    FNodeStats stats;
    Instruction *instruction;
};

static std::map<std::string, int> OP_PRIORITY {
    { "I", 2 },
    { "*", 1 },
    { ":", 1 },
    { "+", 0 },
    { "-", 0 },
    { "e", 2 },
    { "l", 2 },
    { "|", 2 },
    { "s", 1 },
    { "r", 1 },
    { "^", 1 },
    { "2", 1 }
};

static std::map<std::string, bool> OP_COMMUTES {
    { "I", false },
    { "*", true },
    { ":", false },
    { "+", true },
    { "-", false },
    { "e", false },
    { "l", false },
    { "|", false },
    { "s", false },
    { "r", false },
    { "2", false }
};

class Operator
{
  public:
    Operator() : tag("?") {;}
    ~Operator() {;}
    std::string getTag() { return tag; }
    FNode *generateAndCheck(FNode *f1, FNodeCheck &chk);
    FNode *generateAndCheck(FNode *f1, FNode *f2, FNodeCheck &chk);
    virtual std::string format(std::vector<std::string> &args) { assert(false); }
    virtual double evaluate(std::vector<FNode*> &fnodes) { return -1; }
  protected:
    // Unary
    virtual bool checkInput(FNode *f1) { assert(false); }
    virtual FNode* generate(FNode *f1) { assert(false); }
    // Binary
    virtual bool checkInput(FNode *f1, FNode *f2) { assert(false); }
    virtual FNode* generate(FNode *f1, FNode *f2) { assert(false); }
    std::string tag;
};

class OIdent : public Operator
{
  public:
    OIdent() { tag = "I"; }
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue(); }
};

class OExp : public Operator 
{
  public:
    OExp() { tag = "e"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::exp(fnodes[0]->getValue()); }
};

class OLog : public Operator 
{
  public:
    OLog() { tag = "l"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::log(fnodes[0]->getValue()); }
};

class OMod : public Operator
{
  public:
    OMod() { tag = "|"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::abs(fnodes[0]->getValue()); }
};

class OSqrt : public Operator
{
  public:   
    OSqrt() { tag = "s"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return std::sqrt(fnodes[0]->getValue()); }
};

class OInv : public Operator
{
  public:   
  OInv() { tag = "r"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return 1./fnodes[0]->getValue(); }
};

class O2 : public Operator
{
  public:   
    O2() { tag = "2"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return std::pow(fnodes[0]->getValue(), 2.0); }
};

class OPlus : public Operator
{
  public:
    OPlus() { tag = "+"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()+fnodes[1]->getValue(); }
};

class OMinus : public Operator
{
  public:
    OMinus() { tag = "-"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()-fnodes[1]->getValue(); }
};
    
class OMult : public Operator
{
  public:
    OMult() { tag = "*"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()*fnodes[1]->getValue(); }
};

class ODiv : public Operator
{
  public:
    ODiv() { tag = ":"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()/fnodes[1]->getValue(); }
};

class OP_MAP
{
  public:
    typedef std::map<std::string, Operator*> op_map_t;
    static Operator *get(std::string);
    ~OP_MAP();
  private:
    OP_MAP();
    op_map_t op_map;
    static OP_MAP *getInstance();
    static OP_MAP *instance;
};


class FGraph
{
  public:
    typedef std::vector<Operator*> op_vec_t;
    FGraph();
    ~FGraph();
    void addRootNode(std::string varname, std::string varplus, 
        std::string varzero, std::string unit);
    void registerNewNode(FNode *new_node);
    void generate();
    void addLayer(std::string uops, std::string bops);
    void generateLayer(op_vec_t &uops, op_vec_t &bops);
    void apply(matrix_t &input, matrix_t &output);
    bpy::object applyNumpy(bpy::object &np_input, std::string np_dtype);
    bpy::object applyAndCorrelateNumpy(bpy::object &np_X, bpy::object &np_y, std::string np_dtype);
    void applyAndCorrelate(matrix_t &X_in, matrix_t &X_out, matrix_t &Y_in, matrix_t &cov_out);
    static void registerPython();
  private:
    std::vector<FNode*> root_fnodes;
    std::vector<FNode*> fnodes;
    std::vector<op_vec_t> uop_layers;
    std::vector<op_vec_t> bop_layers;
    std::map<std::string, Operator*> uop_map;
    std::map<std::string, Operator*> bop_map;
    std::map<std::string, FNode*> fnode_map;
};

void zscoreMatrixByColumn(matrix_t &X);

void correlateMatrixColumnsPearson(matrix_t &X_in, matrix_t &Y_in, matrix_t &cov_out);

}}

#endif
