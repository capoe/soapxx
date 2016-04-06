#ifndef _SOAP_LINALG_NUMPY
#define _SOAP_LINALG_NUMPY

// See https://mail.python.org/pipermail/cplusplus-sig/2006-December/011409.html

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/python.hpp>

namespace soap { namespace linalg {

/** Converts between numpy arrays and boost ublas matrices. */
struct numpy_converter
{
    //using namespace boost::python;

	//typedef boost::python::object object;
	//typedef boost::python::tuple tuple;

	boost::python::object numpy;
	boost::python::object array_type;
	boost::python::object array_function;
	boost::python::object dtype;

	/** Constructor. dtype_name determines type of matrices created. */
	numpy_converter( const char * dtype_name = "float64" )
	{
		PyObject* module = ::PyImport_Import( boost::python::object( "numpy" ).ptr() );
		if( ! module  )
		{
			throw std::logic_error( "Could not import numpy" );
		}
		numpy = boost::python::object( boost::python::handle<>( module ) );
		array_type = numpy.attr( "ndarray" );
		array_function = numpy.attr( "array" );
		set_dtype( dtype_name );
	}

	/** Set which dtype the created numpy matrices have. */
	void set_dtype( const char * dtype_name = "float64" )
	{
		dtype = numpy.attr( dtype_name );
	}

	/** Convert a numpy matrix to a ublas one. */
	template< typename T >
	boost::numeric::ublas::matrix< T > &
	numpy_to_ublas(
		boost::python::object a,
		boost::numeric::ublas::matrix< T > & m )
	{
		boost::python::tuple shape( a.attr("shape") );
		if( boost::python::len( shape ) != 2 )
		{
			throw std::logic_error( "numeric::array must have 2 dimensions" );
		}
		m.resize(
			boost::python::extract< unsigned >( shape[0] ),
			boost::python::extract< unsigned >( shape[1] ) );
		for( unsigned i = 0; i < m.size1(); ++i )
		{
			for( unsigned j = 0; j < m.size2(); ++j )
			{
				m( i, j ) = boost::python::extract< T >( a[ 
boost::python::make_tuple( i, j ) ] );
			}
		}
		return m;
	}

	/** Convert a ublas matrix to a numpy matrix. */
	template< typename T >
	boost::python::object
	ublas_to_numpy(
		const boost::numeric::ublas::matrix< T > & m )
	{
		//create a numpy array to put it in
		boost::python::object result(
			array_type(
				boost::python::make_tuple( m.size1(), m.size2() ),
				dtype ) );

		//copy the elements
		for( unsigned i = 0; m.size1() != i; ++i )
		{
			for( unsigned j = 0; m.size2() != j; ++j )
			{
				result[ boost::python::make_tuple( i, j ) ] = m( i, j );
			}
		}

		return result;
	}
};

}} /* CLOSE NAMESPACE */

#endif
