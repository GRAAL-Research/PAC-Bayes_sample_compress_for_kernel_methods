// ------------------------------------------------------------------------------------------------
// PAC-BAYES SAMPLE COMPRESS LEARNING ALGORITHM (aka PBSC) 
// Version 0.92 (June 26, 2011), Released under the BSD-license 
// ------------------------------------------------------------------------------------------------
// Author: 
//    Pascal Germain 
//    Groupe de Recherche en Apprentissage Automatique de l'Universite Laval (GRAAL) 
//    http://graal.ift.ulaval.ca/ 
//
// Reference: 
//    Pascal Germain, Alexandre Lacoste, François Laviolette, Mario Marchand, and Sara Shanian. 
//    A PAC-Bayes Sample Compression Approach to Kernel Methods. In Proceedings of the 28th 
//    International Conference on Machine Learning, Bellevue, WA, USA, June 2011. 
// ------------------------------------------------------------------------------------------------


#ifndef STR_VALUE_H
#define STR_VALUE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "StrValue.h"

//////////////////////////////////////////////////////
//                   CStrValue                      //
//////////////////////////////////////////////////////
    
class CStrValue
{
public:
    CStrValue();
    CStrValue(const CStrValue&      _copy );
    CStrValue(const std::string &   _value);
    CStrValue(const char*           _value);  
    CStrValue(const int &           _value);
    CStrValue(const double &        _value);
    CStrValue(const float &         _value);
    CStrValue(const bool &          _value);

    template<typename T>
    CStrValue(const std::vector<T>& _value);
    
    operator    std::string() const;
    operator    int() const;
    operator    double() const;
    operator    float() const;
    operator    bool() const;

    template<typename T>
    operator    std::vector<T>() const;
    
    CStrValue&              operator=( const CStrValue& _copy);
    bool                    operator==(const CStrValue& _value);
    friend std::ostream&    operator<< (std::ostream& _os, const CStrValue& _param);

    const char* c_str() const;

private:
    std::string m_strValue;

};

typedef std::map<std::string, CStrValue> StrValueMap;


// Build a string from a vector
template<typename T>
CStrValue::CStrValue(const std::vector<T>& _array)
{
    std::string separator = ";";
    int nb = _array.size();
    m_strValue = nb > 0 ? CStrValue(_array[0]).c_str() : "";

    for (int i = 1; i < nb; ++i)
        m_strValue += separator + CStrValue(_array[i]).c_str();
}


// Convert the string to a vector
template<typename T>
CStrValue::operator  std::vector<T>() const
{
    std::string     separators  = ";";
    int             len         = m_strValue.length();
    int             posFirst    = 0;
    int             posLast;
    std::vector<T>  array;

    do
    {
        posLast = m_strValue.find_first_of(separators, posFirst);

        if (posLast == -1)
            posLast = len;

        array.push_back( (T)CStrValue(m_strValue.substr(posFirst, posLast - posFirst)) );
        posFirst = posLast + 1;

    } while ( posLast < len );

    return array;
}



#endif // STR_VALUE_H
