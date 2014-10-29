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


#ifndef COMMON_H
#define COMMON_H

#include "Datas/Kernel.h"
#include "Utils/FileUtils.h"
#include "Utils/MathUtils.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#define ERROR(x) { cout << x << endl; return EXIT_FAILURE; }

#define STR_VERSION "Version 0.92 (June 26, 2012)"

#define STR_LINE \
    "----------------------------------------------------------------------------------------------------\n"

#ifndef STR_APPNAME
#define STR_APPNAME "\n ***!!! Define macro 'STR_APPNANE' before including 'common.h' !!!***"
#endif

const char* STR_HEADER =
    STR_LINE
    "PAC-BAYES SAMPLE COMPRESS LEARNING ALGORITHM - " STR_APPNAME " \n"
    STR_VERSION ", Released under the BSD-license \n"
    STR_LINE
    "Author: \n"
    "    Pascal Germain \n"
    "    Groupe de Recherche en Apprentissage Automatique de l'Universite Laval (GRAAL) \n"
    "    http://graal.ift.ulaval.ca/ \n"
    "\n"
    "Reference: \n"
    "    Pascal Germain, Alexandre Lacoste, François Laviolette, Mario Marchand, and Sara Shanian. \n"
    "    A PAC-Bayes Sample Compression Approach to Kernel Methods. In Proceedings of the 28th \n"
    "    International Conference on Machine Learning, Bellevue, WA, USA, June 2011. \n"
    STR_LINE
    ;


CDataMatrix createKernelMatrix(CDataMatrix _data1, CDataMatrix _data2, CKernel _kernel)
{
    CDataMatrix K;

    K.init(_data1.nbEx, _data2.nbEx+1);

    gsl_matrix_view view = gsl_matrix_submatrix(K.X, 0, 0, _data1.nbEx, _data2.nbEx);
    _kernel.fillKernelMatrix(_data1, _data2, &view.matrix);

    K.setCol(_data2.nbEx, 1.0); // bias

    MathUtils::assign(K.Y, _data1.Y);

    return K;
}


#endif // COMMON_H
