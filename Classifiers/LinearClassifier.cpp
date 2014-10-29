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


#include "LinearClassifier.h"
#include "Datas/DataMatrix.h"
#include "Utils/MathUtils.h"
#include "LinearClassifier.h"
#include "Classifiers/LinearClassifier.h"

using namespace std;


// Constructor
CLinearClassifier::CLinearClassifier(int _card)
:CClassifier()
{
    m_vWeights  = NULL;
    m_card      = _card;
}


// Allocate memory
void CLinearClassifier::init()
{
    m_vWeights = gsl_vector_calloc(m_card);
}


// Desallocate memory
void CLinearClassifier::free()
{
    if (m_vWeights != NULL)
        gsl_vector_free(m_vWeights);

    m_vWeights  = NULL;
    m_card      = 0;
}


// Classification function
void CLinearClassifier::classify(const CDataMatrix& _data, gsl_vector* _vPredictions)
{
    if ((int)_vPredictions->size != _data.nbEx)
        throw logic_error("[CLinearClassifier::classify] Prediction vector incorrectly initialized");

    if (m_card != _data.nbFt)
        throw logic_error("[CLinearClassifier::classify] Incompatible amount of features");

    MathUtils::mvProduct(_vPredictions, _data.X, m_vWeights);
}


// Set weight vector
void CLinearClassifier::setWeights(gsl_vector* _vWeights)
{
    if ((int)_vWeights->size == m_card)
        gsl_vector_memcpy(m_vWeights, _vWeights);
    else
        throw logic_error("[CLinearClassifier::setWeights] Incompatible vector lengths");
}


// Save classifier
StrValueMap CLinearClassifier::serialize()
{
    StrValueMap map;

    vector<double> weights(m_card);
    MathUtils::assign(&weights, m_vWeights);

    map["type"]     = "LinearClassifier";
    map["weights"]  = weights;

    return map;
}


// Reconstruct (or load) classifier
void CLinearClassifier::unserialize(StrValueMap &_map)
{
    vector<double> weights = _map["weights"];

    if (m_vWeights == NULL || (int)weights.size() != m_card)
    {
        free();
        m_card = weights.size();
        init();
    }

    MathUtils::assign(m_vWeights, weights);
}
