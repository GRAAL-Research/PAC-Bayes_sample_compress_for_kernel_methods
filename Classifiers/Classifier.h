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


#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "Datas/DataMatrix.h"
#include "Utils/StrValue.h"

#include <iostream>
#include <vector>

#include "gsl/gsl_rng.h"

class CClassifier
{
public:
    // Constructor / Destructor (the cycle of life)
    CClassifier()                   { }
    virtual ~CClassifier()          { }

    // Allocate / Desallocate memory
    virtual void        init()      { }
    virtual void        free()      { }

    // Classify a dataset (predictions provided in a label vector)
    virtual void        classify(const CDataMatrix& _data, gsl_vector* _vPredictions) = 0;

    // Compute the proportion of missclassification on a datatset
    double              calcRisk(const CDataMatrix& _data, gsl_vector* _vPredictions = NULL);

    // Allow to save and reconstruct the classifier
    virtual StrValueMap serialize()                         { return StrValueMap(); }
    virtual void        unserialize(StrValueMap& /*_map*/)  { }
};

#endif // CLASSIFIER_H
