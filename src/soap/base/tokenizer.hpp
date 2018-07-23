#ifndef SOAP_TOKENIZER_HPP
#define	SOAP_TOKENIZER_HPP

#include <string>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

namespace soap { namespace base {

class Tokenizer 
{                        
public:
    typedef boost::tokenizer<boost::char_separator<char> >::iterator
        iterator;
    Tokenizer(const std::string &str, const char *separators) {
        _str = str;
        boost::char_separator<char> sep(separators);
        tok = new boost::tokenizer<boost::char_separator<char> >(_str, sep);
    }
    ~Tokenizer() {
        delete tok;
    }
    iterator begin() { return tok->begin(); }
    iterator end() { return tok->end(); }
    void ToVector(std::vector<std::string> &v) {
        for(iterator iter=begin(); iter!=end(); ++iter)
            v.push_back(*iter);
    }
    std::vector<std::string> ToVector() {
        std::vector<std::string> v;
        this->ToVector(v);
        return v;
    }
    template < typename T >
    void ConvertToVector(std::vector<T> &v){
        std::vector<std::string> tmp;
        ToVector(tmp);
        v.resize(tmp.size());
        typename std::vector<T>::iterator viter = v.begin();
        typename std::vector<std::string>::iterator iter;
        for(iter = tmp.begin(); iter!=tmp.end(); ++iter, ++viter)
            *viter = boost::lexical_cast<T, std::string>(*iter);
    }
private:
    boost::tokenizer< boost::char_separator<char> > *tok;
    std::string _str;
};

}}

#endif
