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


#ifndef KERNEL_H
#define KERNEL_H

#include "DataMatrix.h"
#include "Utils/StrValue.h"

#include <gsl/gsl_vector.h>

class CKernel
{
public:
    // Prototype of a kernel function
    typedef double (*KernelFct)(gsl_vector*, gsl_vector*, double*);

    // Constructors / Destructors (the cycle of life!)
    CKernel();
    CKernel(KernelFct _fct, double _param1=1.0, double _param2=1.0, double _param3=1.0);
    CKernel(const StrValueMap& _map);
    ~CKernel()  { }

    // Allow to save and reconstruct the kernel
    StrValueMap serialize();
    void        unserialize(const StrValueMap& _map);

    // Compute kernel function between two vector-examples
    double kernel(gsl_vector* _x1, gsl_vector* _x2);

    // Compute the Kernel Matrix between two matrix-datasets
    CDataMatrix createKernelMatrix(const CDataMatrix& _X1, const CDataMatrix& _X2);
    void        fillKernelMatrix(const CDataMatrix& _X1, const CDataMatrix& _X2, gsl_matrix* _K);

    // Already implemented Kernel Funnctions
    static double LINEAR        (gsl_vector* _x1, gsl_vector* _x2, double* _params);
    static double POLYNOMIAL    (gsl_vector* _x1, gsl_vector* _x2, double* _params);
    static double RBF           (gsl_vector* _x1, gsl_vector* _x2, double* _params);
    static double TANH          (gsl_vector* _x1, gsl_vector* _x2, double* _params);

private:
    // Pointer to kernel function
    KernelFct   m_kernelFct;

    // Kernel parameter values
    double      m_params[3];
};


// Call the kernel function with appropriate parameters
inline double CKernel::kernel(gsl_vector* _x1, gsl_vector* _x2)
{
    return m_kernelFct(_x1, _x2, m_params);
}

# endif // KERNEL_H
