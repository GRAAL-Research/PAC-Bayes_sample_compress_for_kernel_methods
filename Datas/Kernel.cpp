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


#include "Kernel.h"
#include "Utils/MathUtils.h"
#include <iostream>


template <class T>
void setParam(const StrValueMap& _map, const char* _key, T& _var, const T& _default)
{
    StrValueMap::const_iterator it = _map.find(_key);
    _var = (it == _map.end()) ? _default : (T)(it->second);
}

//  Constructor: Default
CKernel::CKernel()
{
    m_kernelFct = RBF;
    m_params[0] = 0.1;
    m_params[1] = 0.0;
    m_params[2] = 0.0;
}

// Constructor: Takes a costum function
CKernel::CKernel(KernelFct _fct, double _param1/*=1.0*/, double _param2/*=1.0*/, double _param3/*=1.0*/)
{
    m_kernelFct = _fct;
    m_params[0] = _param1;
    m_params[1] = _param2;
    m_params[2] = _param3;
}


// Constructor: Initializes with text parameters (usually specified by a user)
CKernel::CKernel(const StrValueMap& _map)
{
    m_kernelFct = LINEAR;
    m_params[0] = 1.0;
    m_params[1] = 1.0;
    m_params[2] = 1.0;

    unserialize(_map);
}


StrValueMap CKernel::serialize()
{
    StrValueMap map;

    if (m_kernelFct == LINEAR)
    {
        map["kernel"]       = "LINEAR";
    }
    else if (m_kernelFct == RBF)
    {
        map["kernel"]       = "RBF";
        map["kernel.gamma"] = m_params[0];
    }
    else if (m_kernelFct == TANH)
    {
        map["kernel"]       = "TANH";
        map["kernel.s"]     = m_params[0];
        map["kernel.c"]     = m_params[1];
    }
    else if (m_kernelFct == POLYNOMIAL)
    {
        map["kernel"]       = "POLY";
        map["kernel.d"]     = m_params[0];
        map["kernel.s"]     = m_params[1];
        map["kernel.c"]     = m_params[2];
    }
    else
    {
        map["kernel"]       = "CUST0M";
        map["kernel.p1"]    = m_params[0];
        map["kernel.p2"]    = m_params[1];
        map["kernel.p3"]    = m_params[2];
    }

    return map;
}


void CKernel::unserialize(const StrValueMap& _map)
{
    std::string name = "RBF";
    setParam(_map, "kernel", name, name);

    if (name == "RBF")
    {
        m_kernelFct = RBF;
        setParam(_map, "kernel.gamma", m_params[0], 0.1);
    }
    else if (name == "TANH")
    {
        m_kernelFct = TANH;
        setParam(_map, "kernel.s", m_params[0], 1.0);
        setParam(_map, "kernel.c", m_params[1], 0.0);
    }
    else if (name == "POLY")
    {
        m_kernelFct = POLYNOMIAL;
        setParam(_map, "kernel.d", m_params[0], 1.0);
        setParam(_map, "kernel.s", m_params[1], 1.0);
        setParam(_map, "kernel.c", m_params[2], 0.0);
    }
}

// Fill a (already allocated) matrix with kernel values
//  - Rows correspond to examples from dataset X1
//  - Columns correspond to examples from dataset X2
//  => ie: K[i,j] = kernel( X1[i], X2[j] )
void CKernel::fillKernelMatrix(const CDataMatrix &_X1, const CDataMatrix &_X2, gsl_matrix* _K)
{
    if (_X1.nbFt != _X2.nbFt)
        throw std::logic_error("[CKernel::fillKernelMatrix] Different number of features.");

    if (_K == NULL || (int)_K->size1 != _X1.nbEx || (int)_K->size2 != _X2.nbEx)
        throw std::logic_error("[CKernel::fillKernelMatrix] Kernel matrix incorrectly initialized.");

    gsl_vector x1, x2;
    for (int i = 0; i < _X1.nbEx; ++i)
    {
        x1 = _X1.getRow(i);

        for (int j = 0; j < _X2.nbEx; ++j)
        {
            x2 = _X2.getRow(j);

            gsl_matrix_set(_K, i, j, kernel(&x1,&x2));
        }
    }
}


// Allocate memory for a new matrix and compute kernel values with 'fillKernelMatrix' function defined above.
CDataMatrix CKernel::createKernelMatrix(const CDataMatrix &_X1, const CDataMatrix &_X2)
{
    if (_X1.nbFt != _X2.nbFt)
        throw std::logic_error("[CKernel::createKernelMatrix] Different number of features.");

    CDataMatrix K;
    K.init(_X1.nbEx, _X2.nbEx, (_X1.Y != NULL) );

    fillKernelMatrix(_X1, _X2, K.X);

    if (_X1.Y != NULL)
        MathUtils::assign(K.Y, _X1.Y);

    return K;
}

// LINEAR KERNEL
// Function:    k(x,y) = x*y
// Parameters:  none
double CKernel::LINEAR(gsl_vector* _x1, gsl_vector* _x2, double *_params)
{
    return MathUtils::dot(_x1, _x2);
}

// POLYNOMIAL KERNEL
// Function:    k(x,y) = (s x*y+c)^d
// Parameters:  d = _params[0]
//              s = _params[1]
//              c = _params[2]
double CKernel::POLYNOMIAL(gsl_vector* _x1, gsl_vector* _x2, double *_params)
{
    return pow( ( _params[1] * MathUtils::dot(_x1, _x2) + _params[2] ) , _params[0] );
}

// GAUSSIAN KERNEL
// Function:    k(x,y) = exp(-gamma ||x-y||^2)
// Parameters:  gamma = _params[0]
double CKernel::RBF(gsl_vector* _x1, gsl_vector* _x2, double *_params)
{
    return exp( -1 * _params[0] * MathUtils::sqrDistance(_x1,_x2) );
}

// SIGMOID KERNEL
// Parameters:  s = _params[0]
//              c = _params[1]
double CKernel::TANH(gsl_vector* _x1, gsl_vector* _x2, double *_params)
{
    return tanh( _params[0] * MathUtils::dot(_x1, _x2) + _params[1] );
}

