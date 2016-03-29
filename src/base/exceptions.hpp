#ifndef _SOAP_EXCEPTIONS_HPP
#define	_SOAP_EXCEPTIONS_HPP

#include <stdexcept>
#include <exception>

namespace soap { namespace base {

class OutOfRange : public std::runtime_error
{
public:
    explicit OutOfRange(std::string mssg) : std::runtime_error("OutOfRange["+mssg+"]") { ; }
};

class NotImplemented : public std::runtime_error
{
public:
    explicit NotImplemented(std::string mssg) : std::runtime_error("NotImplemented["+mssg+"]") { ; }
};

}}

#endif
