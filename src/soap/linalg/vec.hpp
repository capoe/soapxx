#ifndef _SOAP_LINALG_VEC_H
#define	_SOAP_LINALG_VEC_H

#include <boost/version.hpp>

#if BOOST_VERSION >= 106400
#define BOOST_PYTHON_STATIC_LIB  
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
#endif
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/python.hpp>
#if BOOST_VERSION >= 106400
#include <boost/python/numpy.hpp>
#else
#include <boost/python/numeric.hpp>
#endif

namespace soap { namespace linalg {
using namespace std;
/**
    \brief Vector class for a 3 component vector

    This class represents a 3 component vector to store e.g. positions, velocities, forces, ...
    Operators for basic vector-vector and vector-scalar operations are defined.
    you can access the elements with the functions x(), y(), z(), both reading and writing is possible;
    x + v.x();
    v.x() = 5.;
*/

class vec {
public:
    
    vec();
    vec(const vec &v);
    vec(const double r[3]);
    vec(const double &x, const double &y, const double &z);
    vec(const boost::numeric::ublas::vector<double> &v);
#if BOOST_VERSION >= 106400
    vec(const boost::python::numpy::ndarray &v);
#else
    vec(const boost::python::numeric::array &v);
#endif
    
    vec &operator=(const vec &v);
    vec &operator+=(const vec &v);
    vec &operator-=(const vec &v);
    vec &operator*=(const double &d);
    vec &operator/=(const double &d);
    
    /**
     * \brief get full access to x element
     * @return reference to x
     */
    double &x() { return _x; }
    /**
     * \brief get full access to y element
     * @return reference to y
     */
    double &y() { return _y; }
    /**
     * \brief get full access to z element
     * @return reference to z
     */
    double &z() { return _z; }
    
    void setX(const double &x) { _x = x; }
    void setY(const double &y) { _y = y; }
    void setZ(const double &z) { _z = z; }
    
    /**
     * \brief read only access to x element
     * @return x const reference to x
     *
     * This function can be usefule when const is used to allow for better
     * optimization. Always use getX() instead of x() if possible.
     */
    const double &getX() const { return _x; }
    /**
     * \brief read only access to y element
     * @return x const reference to y
     *
     * This function can be usefule when const is used to allow for better
     * optimization. Always use getY() instead of y() if possible.
     */
    const double &getY() const { return _y; }
    /**
     * \brief read only access to z element
     * @return x const reference to z
     *
     * This function can be usefule when const is used to allow for better
     * optimization. Always use getZ() instead of Z() if possible.
     */
    const double &getZ() const { return _z; }

    void nill() { return; }
    
    /**
     * \brief normalize the vector
     * @return normalized vector
     * This function normalizes the vector and returns itself after normalization.
     * After this call, the vector stores the normalized value.
     */
    vec &normalize();
    
    boost::numeric::ublas::vector<double> converttoub();
    
    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) { arch & _x; arch & _y; arch & _z; }

    static void registerPython() {
        using namespace boost::python;
        class_<vec>("vec", init<double, double, double>())
#if BOOST_VERSION >= 106400
            .def(init<boost::python::numpy::ndarray &>())
#else
	    .def(init<boost::python::numeric::array &>())
#endif
            .add_property("x", make_function(&vec::x, return_value_policy<copy_non_const_reference>()), &vec::setX)
            .add_property("y", make_function(&vec::y, return_value_policy<copy_non_const_reference>()), &vec::setY)
            .add_property("z", make_function(&vec::z, return_value_policy<copy_non_const_reference>()), &vec::setZ)
            .def(self_ns::str(self_ns::self));
    }
    
    private:
        double _x, _y, _z;
};

inline vec::vec() {}

inline vec::vec(const vec &v)
    : _x(v._x), _y(v._y), _z(v._z) {}
        
inline vec::vec(const double r[3])
    : _x(r[0]), _y(r[1]), _z(r[2]) {}

inline vec::vec(const boost::numeric::ublas::vector<double> &v)
    {try
    {_x=v(0);
     _y=v(1);
     _z=v(2);
    }
    catch(std::exception &err){throw std::length_error("Conversion from ub::vector to votca-vec failed");} 
}

#if BOOST_VERSION >= 106400
inline vec::vec(const boost::python::numpy::ndarray &v) {
#else
inline vec::vec(const boost::python::numeric::array &v) {
#endif
   _x = boost::python::extract<double>(v[0]);
   _y = boost::python::extract<double>(v[1]);
   _z = boost::python::extract<double>(v[2]);
}

inline vec::vec(const double &x, const double &y, const double &z)
        : _x(x), _y(y), _z(z) {}
    
inline bool operator==(const vec &v1, const vec &v2)
{
    return ((v1.getX()==v2.getX()) && (v1.getY()==v2.getY()) && (v1.getZ()==v2.getZ()));
}

inline bool operator!=(const vec &v1, const vec &v2)
{
    return ((v1.getX()!=v2.getX()) || (v1.getY()!=v2.getY()) || (v1.getZ()==v2.getZ()));
}

inline vec &vec::operator=(const vec &v)
{ 
        _x=v._x; _y=v._y; _z=v._z;
        return *this;
}    

inline vec &vec::operator+=(const vec &v)
{ 
        _x+=v._x; _y+=v._y; _z+=v._z;
        return *this;
}    
        
inline vec &vec::operator-=(const vec &v)
{ 
        _x-=v._x; _y-=v._y; _z-=v._z;
        return *this;
}    

inline vec &vec::operator*=(const double &d)
{ 
        _x*=d; _y*=d; _z*=d;
        return *this;
}    

inline vec &vec::operator/=(const double &d)
{ 
        _x/=d; _y/=d; _z/=d;
        return *this;
}    

inline vec operator+(const vec &v1, const vec &v2)
{
    return (vec(v1)+=v2);
}

inline vec operator-(const vec &v1, const vec &v2)
{
    return (vec(v1)-=v2);
}

inline vec operator-(const vec &v1){
    return vec (-v1.getX(), -v1.getY(), -v1.getZ());
}

inline vec operator*(const vec &v1, const double &d)
{
    return (vec(v1)*=d);
}

inline vec operator*(const double &d, const vec &v1)
{
    return (vec(v1)*=d);
}

inline vec operator/(const vec &v1, const double &d)
{
    return (vec(v1)/=d);
}

inline std::ostream &operator<<(std::ostream &out, const vec& v)
{
      out << '[' << v.getX() << " " << v.getY() << " " << v.getZ() << ']';
      return out;
}

// dot product
inline double operator*(const vec &v1, const vec &v2)
{
    return v1.getX()*v2.getX() + v1.getY()*v2.getY() + v1.getZ()*v2.getZ();
}

// cross product
inline vec operator^(const vec &v1, const vec &v2)
{
    return vec(
        v1.getY()*v2.getZ() - v1.getZ()*v2.getY(),
        v1.getZ()*v2.getX() - v1.getX()*v2.getZ(),
        v1.getX()*v2.getY() - v1.getY()*v2.getX()
    );
}

inline double abs(const vec &v)
{
    return sqrt(v*v);
}

inline double maxnorm(const vec &v) {
    return ( std::abs(v.getX()) > std::abs(v.getY()) ) ?
         ( ( std::abs(v.getX()) > std::abs(v.getZ()) ) ? 
             std::abs(v.getX()) : std::abs(v.getZ()) )
      :  ( ( std::abs(v.getY()) > std::abs(v.getZ()) ) ? 
             std::abs(v.getY()) : std::abs(v.getZ()) );
}

inline vec &vec::normalize()
{ 
    return ((*this)*=1./abs(*this));
}



inline boost::numeric::ublas::vector<double> vec::converttoub() {
    boost::numeric::ublas::vector<double> temp=boost::numeric::ublas::zero_vector<double>(3);
    temp(0)=_x;
    temp(1)=_y;
    temp(2)=_z;
    return temp;


}
}}
#endif	/* _vec_H */

