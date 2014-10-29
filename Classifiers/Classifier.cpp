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


#include "Classifier.h"
#include "Utils/MathUtils.h"

using namespace std;


// Compute the proportion of missclassification on a datatset
double CClassifier::calcRisk(const CDataMatrix& _data, gsl_vector* _vPredictions /*=NULL*/)
{
    gsl_vector* vPredTmp;

    if (_data.nbEx == 0)
        return 0.0;

    if (_vPredictions == NULL)
        vPredTmp = gsl_vector_alloc(_data.nbEx);
    else
        vPredTmp = _vPredictions;

    // Function 'classify' must be implemented in the derived class
    classify(_data, vPredTmp);

    // Compare each prediction with the true label
    int nbErrors = 0;
    for (int i = 0; i < _data.nbEx; ++i)
    {
        if ( (_data.getY(i)>0.0) != (gsl_vector_get(vPredTmp, i)>0.0) )
            ++nbErrors;
    }

    if (_vPredictions == NULL)
        gsl_vector_free(vPredTmp);

    // Return the proportion of misscllassification
    return (double)nbErrors/_data.nbEx;
}
