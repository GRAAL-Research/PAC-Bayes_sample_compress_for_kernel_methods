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


#ifndef LINEAR_CLASSIFIER_H
#define	LINEAR_CLASSIFIER_H

#include "Classifier.h"

class CLinearClassifier : public CClassifier
{
public:
    // Constructor / Destructor (the cycle of life)
    CLinearClassifier(int _cardinality);
    virtual ~CLinearClassifier()    { }

    // Allocate / Desallocate memory
    virtual void    init();
    virtual void    free();

    // Classification function
    virtual void        classify(const CDataMatrix& _data, gsl_vector* _vPredictions);

    virtual StrValueMap serialize();
    virtual void        unserialize(StrValueMap& _map);

    // Set / Get weight vector
    void            setWeights(gsl_vector* _vWeights);
    gsl_vector*     getWeights()                        { return m_vWeights;        }
    int             getCardinality()                    { return m_vWeights->size;  }


private:

    // Weight vector
    gsl_vector*     m_vWeights;

    // Cardinality of the weight vector
    int             m_card;

};

#endif	// LINEAR_CLASSIFIER_H
