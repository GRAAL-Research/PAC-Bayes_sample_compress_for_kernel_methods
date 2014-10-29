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


#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <fstream>

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_rng.h"


namespace MathUtils
{
// FUNCTION PROTOTYPES //
// All function are inlined below to accelerate the mathematical operations

// Copy a vector into an other one
void    assign(gsl_vector*  _ptrVector, gsl_vector*  _vector1);
void    assign(gsl_vector*  _ptrVector, const std::vector<double>& _vector1);
void    assign(gsl_vector*  _ptrVector, const std::vector<int> & _vector1);
void    assign(std::vector<double> *_ptrVector, gsl_vector*  _vector1);

// Scalar product (aka Dot product)
double  dot( gsl_vector*  _vector1,  gsl_vector*  _vector2);

// Matrix by vector product
void    mvProduct(gsl_vector*  _ptrVector, gsl_matrix* _matrix1, gsl_vector*  _vector2,  bool bTransposeMatrix = false);

// Matrix by matrix product
void    matrixProduct(gsl_matrix* _ptrMatrix, gsl_matrix* _matrix1, gsl_matrix* _matrix2,
                      bool bTransposeMatrix1 = false , bool bTransposeMatrix2 = false );

// Sum (and absolute sum) of vector elements
double  sum(gsl_vector*  _vector);
double  sumAbs(gsl_vector*  _vector);

// Pairwise multiplication of vectors
void    multiply(gsl_vector*  _ptrVector, double _value);
void    multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _value);
void    multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1);
void    multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2);

// Pairwise divisions of vectors
void    divide(gsl_vector*  _ptrVector, double _value);
void    divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _value);
void    divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1);
void    divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2);

// Pairwise addition of vectors
void    add(gsl_vector*  _ptrVector, double _value);
void    add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor = 1.0);
void    add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2, double _factor = 1.0);
void    add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor1, gsl_vector*  _vector2, double _factor2);

// Pairwise substraction of vectors
void    substract(gsl_vector*  _ptrVector, double _value);
void    substract(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2, double _factor = 1.0);
void    substract(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor = 1.0);

// Compute the distance between two vectors
double  distance( gsl_vector*  _vector1, gsl_vector*  _vector2 );
double  sqrDistance( gsl_vector*  _vector1, gsl_vector*  _vector2 );


// TEMPLATE FONCTIONS //

// Randomly shuffle any std::vector
// rng is a initilized random number generator
 template <class T>
 void shuffleVector( std::vector<T> & _vector, gsl_rng* _rng )
{
    size_t nb = _vector.size();
    for (size_t i = 0; i < nb; ++i)
        std::swap( _vector[ i ], _vector[ i + gsl_rng_uniform_int(_rng, nb-i) ] );
};


// FUNCTION DEFINITIONS //

inline double dot( gsl_vector*  _vector1,  gsl_vector*  _vector2)
{
    double result;

    gsl_blas_ddot(_vector1, _vector2, &result);
    return result;
}


inline double sum(gsl_vector*  _vector)
{
    double sum = 0.0;

    for (unsigned int i = 0; i < _vector->size; ++i)
        sum += gsl_vector_get(_vector, i);

    return sum;
}

inline double sumAbs(gsl_vector*  _vector)
{
    return gsl_blas_dasum(_vector);
}


inline void matrixProduct(gsl_matrix* _ptrMatrix, gsl_matrix* _matrix1, gsl_matrix* _matrix2,
                           bool bTransposeMatrix1 /*= false*/ , bool bTransposeMatrix2 /*= false*/ )
{
    gsl_blas_dgemm( bTransposeMatrix1 ? CblasTrans : CblasNoTrans,
                    bTransposeMatrix2 ? CblasTrans : CblasNoTrans,
                    1.0, _matrix1, _matrix2, 0.0, _ptrMatrix );
}


inline void mvProduct(gsl_vector*  _ptrVector, gsl_matrix* _matrix1, gsl_vector*  _vector2,
                           bool bTransposeMatrix /*= false*/)
{
    gsl_blas_dgemv( bTransposeMatrix ? CblasTrans : CblasNoTrans, 1, _matrix1, _vector2, 0, _ptrVector);
}


inline void divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1)
{
    gsl_vector_div(_ptrVector, _vector1);
}

inline void divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2)
{
    gsl_blas_dcopy(_vector1, _ptrVector);
    divide(_ptrVector, _vector2);
}


inline void add(gsl_vector*  _ptrVector, double _value)
{
    gsl_vector_add_constant(_ptrVector, _value);
}

inline void add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor /*= 1.0*/)
{
    gsl_blas_daxpy(_factor, _vector1, _ptrVector);
}

inline void add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2,
                     double _factor /*= 1.0*/)
{
    gsl_blas_dcopy(_vector1, _ptrVector);
    add(_ptrVector, _vector2, _factor);
}

inline void add(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor1, gsl_vector*  _vector2, double _factor2)
{
    multiply(_ptrVector, _vector1, _factor1);
    add(_ptrVector, _vector2, _factor2);
}

inline void substract(gsl_vector*  _ptrVector, double _value)
{
    add(_ptrVector, -1*_value);
}

inline void substract(gsl_vector*  _ptrVector, gsl_vector*  _vector1,
                           gsl_vector*  _vector2, double _factor /*= 1.0*/)
{
    add(_ptrVector, _vector1, _vector2, -1*_factor);
}

inline void substract(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _factor /*= 1.0*/)
{
    add(_ptrVector, _vector1, -1*_factor);
}

inline void multiply(gsl_vector*  _ptrVector, double _value)
{
    gsl_blas_dscal(_value, _ptrVector);
}

inline void multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _value)
{
    gsl_blas_dcopy(_vector1, _ptrVector);
    multiply(_ptrVector, _value);
}

inline void multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1)
{
    gsl_vector_mul(_ptrVector, _vector1);
}

inline void multiply(gsl_vector*  _ptrVector, gsl_vector*  _vector1, gsl_vector*  _vector2)
{
    gsl_blas_dcopy(_vector1, _ptrVector);
    multiply(_ptrVector, _vector2);
}


inline void divide(gsl_vector*  _ptrVector, double _value)
{
    if (_value == 0.0)
        throw std::logic_error("[MathUtils::divide] Division by zero");
    else
        multiply(_ptrVector, 1.0/_value);
}

inline void divide(gsl_vector*  _ptrVector, gsl_vector*  _vector1, double _value)
{
    if (_value == 0.0)
        throw std::logic_error("[MathUtils::divide] Division by zero");
    else
        multiply(_ptrVector, _vector1, 1.0/_value);
}


inline void assign(gsl_vector*  _ptrVector, gsl_vector*  _vector1)
{
    gsl_vector_memcpy(_ptrVector, _vector1);
}

inline void assign(gsl_vector*  _ptrVector, const std::vector<double>& _vector1)
{
    size_t nb = _vector1.size();
    for (size_t i = 0; i < nb; ++i)
        gsl_vector_set( _ptrVector, i, _vector1[i] );
}

inline void assign(std::vector<double> *_ptrVector, gsl_vector*  _vector1)
{
    for (size_t i = 0; i < _vector1->size; ++i)
        (*_ptrVector)[i] = gsl_vector_get(_vector1, i) ;
}

inline void assign(gsl_vector*  _ptrVector, const std::vector<int> & _vector1)
{
    for (size_t i = 0; i < _ptrVector->size; ++i)
        gsl_vector_set( _ptrVector, i, _vector1[i] );
}


inline double distance( gsl_vector*  _vector1, gsl_vector*  _vector2 )
{
    return sqrt( sqrDistance(_vector1, _vector2) );
}

inline double sqrDistance( gsl_vector*  _vector1, gsl_vector*  _vector2 )
{
    return dot(_vector1, _vector1) + dot(_vector2, _vector2) - 2*dot(_vector1, _vector2);
}


} // namespace MathUtils

#endif // MATH_UTILS_H
