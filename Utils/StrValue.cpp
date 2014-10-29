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


#include "StrValue.h"

#include <sstream>
#include <cstdlib>

using namespace std;

//////////////////////////////////////////////////////
//                   CStrValue                      //
//////////////////////////////////////////////////////
CStrValue::CStrValue()
{
    m_strValue.clear();
}

CStrValue::CStrValue(const CStrValue& _copy)
{
    m_strValue = _copy.m_strValue;
}

CStrValue& CStrValue::operator=(const CStrValue & _copy)
{
    m_strValue = _copy.m_strValue;
    return (*this);
}


const char* CStrValue::c_str() const
{
    return m_strValue.c_str();
}


bool CStrValue::operator==(const CStrValue& _value)
{
    return m_strValue == _value.m_strValue;
}


std::ostream & operator<<(std::ostream& _os, const CStrValue& _param)
{
    _os << _param.m_strValue.c_str();
    return _os;
}


// std::string
CStrValue::operator std::string() const
{
    return m_strValue;
}


CStrValue::CStrValue(const std::string & _value)
{
    m_strValue = _value;
}

CStrValue::CStrValue(const char* _value)
{
    if (_value == NULL)
        m_strValue.clear();
    else
        m_strValue = _value;
}


// int
CStrValue::operator int() const
{
    return atoi(m_strValue.c_str());
}

CStrValue::CStrValue(const int & _value)
{
    stringstream s;
    s << _value;
    m_strValue = s.str();
}


// double
CStrValue::operator double() const
{
    return atof(m_strValue.c_str());
}

CStrValue::CStrValue(const double & _value)
{
    stringstream s;
    s.precision(12);
    s << _value;
    m_strValue = s.str();
}


// float
CStrValue::operator float() const
{
    return (float)atof(m_strValue.c_str());
}

CStrValue::CStrValue(const float & _value)
{
    stringstream s;
    s << _value;
    m_strValue = s.str();
}

// bool
CStrValue::operator bool() const
{
    if ( m_strValue == "true" || m_strValue == "TRUE" || m_strValue == "True" ||
         m_strValue == "yes"  || m_strValue == "YES"  || m_strValue == "Yes"  ||
         m_strValue == "1"                              )
        return true;
    else
        return false;
}

CStrValue::CStrValue(const bool & _value)
{
    m_strValue = _value ? "true" : "false";
}
