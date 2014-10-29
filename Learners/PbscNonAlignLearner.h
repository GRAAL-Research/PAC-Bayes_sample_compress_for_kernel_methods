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


#ifndef PBSC_NON_ALIGN_LEARNER_H
#define PBSC_NON_ALIGN_LEARNER_H

#include "Learner.h"
#include "Utils/TabLogFile.h"

class CPbscNonAlignLearner : public CLearner
{
public:
    // Constructor / Destructor (the cycle of life!)
    CPbscNonAlignLearner( );
    virtual ~CPbscNonAlignLearner()     {}

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
    double          findDelta(int _index1, int _index2);

    static double   fctDelta(double x, void *_params);

    struct          ParamsDelta { double  mult, dot, sqr, w1, w2; };

    // Compute objective function cost value
    double          calcCost(double* _ptrKL = NULL);

    // Regroup complementary weights (vector of size 2*m) in a regular weight vector (of size m)
    void            groupWeights();

    // Log file helpers
    void        initLog();
    void        writeLog();
    void        finalizeLog();

    // Algorithm parameters
    double      param_q;
    double      param_C;
    double      param_stopCriteria;
    int         param_maxIter;
    int         param_seed;
    int         param_writeStep;

    // Weight vector
    gsl_vector* m_vWeights;
    gsl_vector* m_vGroupWeights;

    // Distribution of weights over examples
    gsl_vector* m_vDist;

    // Log file
    CTabLogFile m_log;

    // Number of iterations performed
    int         m_iter;

    // Cost value
    double      m_cost;

    // Kullback-Leibler divergence value
    double      m_KL;

    // Maximum weight exchange during an iteration
    double      m_maxDelta;

    // Maximum number of iteration performed by Brent root-finding method
    int         m_maxBrent;

    // Random number generator
    gsl_rng*    m_randomNumberGen;
    
};

#endif // PBSC_NON_ALIGN_LEARNER_H
