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


#ifndef PBSC_ALIGN_LEARNER_H
#define PBSC_ALIGN_LEARNER_H

#include "Learner.h"
#include "Utils/TabLogFile.h"

class CPbscAlignLearner : public CLearner
{
public:
    // Constructor / Destructor (the cycle of life!)
    CPbscAlignLearner( );
    virtual ~CPbscAlignLearner()    {}

    // Allocate / Desallocate memory
    virtual void            init();
    virtual void            free();

    // Set algorithm parameters
    virtual void            setParameters(const StrValueMap& _params);

    // Execute learning algorithm
    virtual CClassifier*    learn();

    // Get algorithm statistics (after learning)
    virtual StrValueMap     getStats();

    
protected:
    // Compute the optimal weight transfer for a component of the weight vector
    double      findDelta(int _wIndex);

    // Compute objective function cost value
    double      calcCost();

    // Log file helpers
    void        initLog();
    void        writeLog();
    void        finalizeLog();

    // Algorithm parameters
    double      param_q;                // minimum of the quadratic risk
    double      param_stopCriteria;     // convergence criteria
    int         param_maxIter;          // maximum number of iteration
    int         param_seed;             // random number generator seed
    int         param_writeStep;

    // Weight vector
    gsl_vector* m_vWeights;

    // Distribution of weights over examples
    gsl_vector* m_vDist;

    // Sum of the squared elements on each kernel matrix columns
    gsl_vector* m_vColSquared;

    // Log file
    CTabLogFile m_log;

    // Number of iterations performed
    int         m_iter;

    // Cost value
    double      m_cost;

    // Ratio of weight vector component values reaching a boundarie
    double      m_saturation;

    // Maximum weight exchange during an iteration
    double      m_maxDelta;

    // Random number generator
    gsl_rng*    m_randomNumberGen;
    
};

#endif // PBSC_ALIGN_LEARNER_H
