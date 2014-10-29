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


#include "PbscAlignLearner.h"

#include "Classifiers/LinearClassifier.h"
#include "Utils/MathUtils.h"
#include <iostream>

using namespace std;


// Constructor (default)
CPbscAlignLearner::
CPbscAlignLearner()
: CLearner()
{

}


// Set algorithm parameters
void CPbscAlignLearner::setParameters(const StrValueMap& _params)
{
    CLearner::setParameters(_params);

    setParam(_params, "q",                  param_q,                0.02            );
    setParam(_params, "stopCriteria",       param_stopCriteria,     1e-16           );
    setParam(_params, "nIter",              param_maxIter,          200000          );
    setParam(_params, "seed",               param_seed,             (int)time(NULL) );
    setParam(_params, "writeStep",          param_writeStep,        100             );
}


// Get algorithm statistics (after learning)
StrValueMap CPbscAlignLearner::getStats()
{
    StrValueMap stats = CLearner::getStats();

    stats["nIter"] = m_iter;
    stats["cost"]  = m_cost;
    stats["saturation"] = m_saturation;
    stats["q"]     = param_q;
    stats["seed"]  = param_seed;

    return stats;
}


// Allocate memory
void CPbscAlignLearner::init()
{
    // Random number generator
    m_randomNumberGen = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(m_randomNumberGen, param_seed);

    CLearner::init();
}


// Desallocate memory
void CPbscAlignLearner::free()
{
    gsl_rng_free(m_randomNumberGen);
    m_randomNumberGen = NULL;

    CLearner::free();
}


// Execute learning algorithm
CClassifier* CPbscAlignLearner::learn()
{
    m_pClassifier = new CLinearClassifier(data_train.nbFt);
    m_pClassifier->init();

    // Weight vector
    m_vWeights = gsl_vector_calloc(data_train.nbFt);
    double saturationValue = 1.0/(data_train.nbFt);
    gsl_vector_set_all(m_vWeights, 0.0);

    // Distribution on examples
    m_vDist = gsl_vector_alloc(data_train.nbEx);
    MathUtils::mvProduct(m_vDist, data_train.X, m_vWeights, false);
    MathUtils::add(m_vDist, data_train.Y, -param_q);
   
    // For each column of the kernel matrix, we compute the sum of its squarred elements
    // (This constant will by used during the minimization procedure)
    m_vColSquared = gsl_vector_alloc(data_train.nbFt);
    for (int i = 0; i < data_train.nbFt; ++i)
    {
        gsl_vector v = data_train.getCol(i);
        gsl_vector_set(m_vColSquared, i, MathUtils::dot(&v, &v));
    }

    // Visit order (shuffled before each iteration)
    vector<int> visitOrder(data_train.nbFt);
    for (int i = 0; i < data_train.nbFt; ++i)
        visitOrder[i] = i;

    m_saturation    = 0.0;
    m_maxDelta      = 0.0;
    if (param_bWriteLog)
        initLog();

    // Minimization procedure
    double          weight, delta;
    int             wIndex;
    bool            bContinue;
    int             barStep;

    barStep     = (param_maxIter <= 100) ? 1 : param_maxIter/100;
    bContinue   = true;
    m_iter      = 0;
    while (bContinue && m_iter < param_maxIter)
    {
        // Writing log file, if necessary
        if (param_bWriteLog && m_iter % param_writeStep == 0)
            writeLog();

        // Progress bar
        if ( m_iter % barStep == 0 )
            cout << "-" << flush;

        // Shuffling the visit order vector
        MathUtils::shuffleVector(visitOrder, m_randomNumberGen);
        
        // Visit each component of the weight vector
        m_maxDelta   = 0.0;
        m_saturation = 0.0;
        for (int i = 0; i < data_train.nbFt; ++i)
        {
            // Select the component to minimize
            wIndex = visitOrder[i];
            weight = gsl_vector_get(m_vWeights, wIndex);
             
            // Compute weight transfer
            delta = findDelta(wIndex);
            
            if (weight+delta >= saturationValue)
            {
                delta         = saturationValue-weight;
                m_saturation += saturationValue;
            }
            else if (weight+delta <= -saturationValue)
            {
                delta         = -saturationValue-weight;
                m_saturation += saturationValue;
            }

            // Updating weight vector
            gsl_vector_set(m_vWeights, wIndex, weight+delta);
            m_maxDelta = max(m_maxDelta, fabs(delta));
            
            // Updating distribution on examples
            gsl_vector v = data_train.getCol(wIndex);
            MathUtils::add(m_vDist, &v, delta);
        }

        // Stoping criteria
        if (m_maxDelta < param_stopCriteria)
            bContinue = false;        
        
        ++m_iter;
    }

    // Writting log file, if necessary
    if (param_bWriteLog)
    {
        writeLog();
        finalizeLog();
    }
    else
        m_cost = calcCost();

    cout << endl;

    ((CLinearClassifier*)m_pClassifier)->setWeights(m_vWeights);

    // Freeing memory
    gsl_vector_free(m_vWeights);
    gsl_vector_free(m_vDist);

    return m_pClassifier;
}


// Compute the optimal weight transfer for a component of the weight vector
double CPbscAlignLearner::findDelta(int _wIndex)
{
    gsl_vector v = data_train.getCol(_wIndex);
    double dot = MathUtils::dot(m_vDist, &v);
    double sqr = gsl_vector_get(m_vColSquared, _wIndex);

    return -dot/sqr;
}


// Compute objective function cost value
double CPbscAlignLearner::calcCost()
{
    gsl_vector* vMargins = gsl_vector_alloc(data_train.nbEx);

    MathUtils::mvProduct(vMargins, data_train.X, m_vWeights, false);
    MathUtils::multiply(vMargins, data_train.Y);

    double loss = 0.0;
    double tmp;

    for (int i = 0; i < data_train.nbEx; ++i)
    {
        tmp  =  param_q - gsl_vector_get(vMargins, i);
        loss += tmp*tmp;
    }
    
    gsl_vector_free(vMargins);

    return loss;
}


// Initialize log file (open file and write headers)
void CPbscAlignLearner::initLog()
{
    m_log.init(param_sLogFile.c_str());
    m_log.begin();

    vector<string> header;
    header.push_back("Iter");
    header.push_back("Cost");
    header.push_back("maxDelta");
    header.push_back("Saturation");
    header.push_back("TrainRisk");
    header.push_back("TestRisk");

    m_log.createHeader(header);

}


// Finalize log file (close file)
void CPbscAlignLearner::finalizeLog()
{
    m_log.end();
}


// Write a line in the log file
void CPbscAlignLearner::writeLog()
{
    StrValueMap map;

    m_cost = calcCost();

   ((CLinearClassifier*)m_pClassifier)->setWeights(m_vWeights);

    map["Iter"]         = m_iter;
    map["Cost"]         = m_cost;
    map["maxDelta"]     = m_maxDelta;
    map["Saturation"]   = m_saturation;

    map["TrainRisk"]    = m_pClassifier->calcRisk(data_train);

    if (m_hasTestData)
    {
        map["TestRisk"] = m_pClassifier->calcRisk(data_test);
    }

    m_log.write(map);
}
