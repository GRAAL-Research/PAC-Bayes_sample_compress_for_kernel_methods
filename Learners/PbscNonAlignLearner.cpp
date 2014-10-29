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


#include "PbscNonAlignLearner.h"

#include "Classifiers/LinearClassifier.h"
#include "Utils/MathUtils.h"
#include <gsl/gsl_roots.h>
#include <iostream>

using namespace std;

// Epsilon value for Brent root-finding method
#define EPS_BRENT 1e-16


// Constructor (default)
CPbscNonAlignLearner::
CPbscNonAlignLearner()
: CLearner()
{

}


// Set algorithm parameters
void CPbscNonAlignLearner::setParameters(const StrValueMap& _params)
{
    CLearner::setParameters(_params);

    setParam(_params, "C",                  param_C,                1.0             );
    setParam(_params, "q",                  param_q,                0.02            );
    setParam(_params, "stopCriteria",       param_stopCriteria,     1e-8            );
    setParam(_params, "nIter",              param_maxIter,          20000           );
    setParam(_params, "seed",               param_seed,             (int)time(NULL) );
    setParam(_params, "writeStep",          param_writeStep,        100             );
}


// Get algorithm statistics (after learning)
StrValueMap CPbscNonAlignLearner::getStats()
{
    StrValueMap stats = CLearner::getStats();

    stats["nIter"] = m_iter;
    stats["cost"]  = m_cost;
    stats["C"]     = param_C;
    stats["q"]     = param_q;
    stats["seed"]  = param_seed;

    return stats;
}


// Allocate memory
void CPbscNonAlignLearner::init()
{
    // Random number generator
    m_randomNumberGen = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(m_randomNumberGen, param_seed);

    CLearner::init();
}


// Desallocate memory
void CPbscNonAlignLearner::free()
{
    gsl_rng_free(m_randomNumberGen);
    m_randomNumberGen = NULL;

    CLearner::free();
}


// Execute learning algorithm
CClassifier* CPbscNonAlignLearner::learn()
{
    m_pClassifier = new CLinearClassifier(data_train.nbFt);
    m_pClassifier->init();

    // Weight vector
    m_vWeights = gsl_vector_alloc(2*data_train.nbFt);
    gsl_vector_set_all(m_vWeights, 1.0/(2*data_train.nbFt) );

    m_vGroupWeights = gsl_vector_alloc(data_train.nbFt);
    groupWeights();

    // Distribution on examples
    m_vDist = gsl_vector_alloc(data_train.nbEx);
    MathUtils::mvProduct(m_vDist, data_train.X, m_vGroupWeights, false);
    MathUtils::add(m_vDist, data_train.Y, -param_q);
   
    // Visit order (shuffled before each iteration)
    vector<int> visitOrder1(2*data_train.nbFt);
    for (int i = 0; i < 2*data_train.nbFt; ++i)
        visitOrder1[i] = i;

    m_maxDelta      = 0.0;
    m_maxBrent      = 0;
    if (param_bWriteLog)
        initLog();

    // Minimization procedure
    double          delta;
    double          weight1, weight2;
    int             index1, index2;
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
        MathUtils::shuffleVector(visitOrder1, m_randomNumberGen);

        // Visit each component of the weight vector
        for (int i = 0; i < 2*data_train.nbFt; ++i)
        {
            // Select the component to minimize
            index1  = visitOrder1[i];
            weight1 = gsl_vector_get(m_vWeights, index1);

            do{
                index2  = gsl_rng_uniform_int(m_randomNumberGen, 2*data_train.nbFt);
                weight2 = gsl_vector_get(m_vWeights, index2);
            } while( index1 == index2 || 2*EPS_BRENT >= weight1+weight2 );

            // Compute weight transfer
            delta = findDelta(index1, index2);

            double c1=0, c2=0;
            if (param_bVerbose)
            {
                cout << " i1=" << index1;
                cout << " i2=" << index2;
                cout << " w1=" << weight1;
                cout << " w2=" << weight2;
                cout << " dt=" << delta;
                cout << " nB=" << m_maxBrent; m_maxBrent=0;
                c1 = calcCost();

            }

            // Updating weight vector (transfer between w1 and w2)
            gsl_vector_set(m_vWeights, index1, weight1+delta);
            gsl_vector_set(m_vWeights, index2, weight2-delta);
            m_maxDelta = max(m_maxDelta, fabs(delta));
            
            // Updating distribution on examples
            gsl_vector g1 = data_train.getCol(index1 - ( index1 < data_train.nbFt ? 0 : data_train.nbFt ) );
            gsl_vector g2 = data_train.getCol(index2 - ( index2 < data_train.nbFt ? 0 : data_train.nbFt ) ) ;
            MathUtils::add( m_vDist, &g1, +delta * ( index1 < data_train.nbFt ? +1 : -1 ) );
            MathUtils::add( m_vDist, &g2, -delta * ( index2 < data_train.nbFt ? +1 : -1 ) );

            if (param_bVerbose)
            {
                c2= calcCost();
                cout << " c1=" << c1;
                cout << " c2=" << c2;
                if (c2-c1 > 1e-12*param_C)
                    cout << " <= PROBLEM! ( " << (c2-c1) << " )" << endl;
                cout << endl;
            }

        }

        // Stoping criteria
        if (m_iter % 100 == 0)
        {
            if (m_maxDelta < param_stopCriteria)
                bContinue = false;
            else
                m_maxDelta = 0.0;
        }
        
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

    groupWeights();
    ((CLinearClassifier*)m_pClassifier)->setWeights(m_vGroupWeights);

    // Freeing memory
    gsl_vector_free(m_vWeights);
    gsl_vector_free(m_vGroupWeights);
    gsl_vector_free(m_vDist);

    return m_pClassifier;
}


// Compute the optimal weight transfer between two components of the weight vector.
// To do so, we find the root of a function (called F(delta) in commentaries below)
// See Supplementaty materials of the related paper for details
double CPbscNonAlignLearner::findDelta(int _index1, int _index2)
{
    double w1  = gsl_vector_get(m_vWeights, _index1);
    double w2  = gsl_vector_get(m_vWeights, _index2);

    if ( _index1 == _index2 || 2*EPS_BRENT >= w1+w2 )
    {
        return 0.0;
    }

    // Compute constant values appearing in the function F(delta)
    gsl_vector *v = gsl_vector_calloc(data_train.nbEx);

    gsl_vector g1 = data_train.getCol( _index1 - ( _index1 < data_train.nbFt ? 0 : data_train.nbFt ) );
    gsl_vector g2 = data_train.getCol( _index2 - ( _index2 < data_train.nbFt ? 0 : data_train.nbFt ));
    MathUtils::add( v, &g1, (_index1 < data_train.nbFt ? +1 : -1), &g2, -1*(_index2 < data_train.nbFt ? +1 : -1) );

    double dot = MathUtils::dot(m_vDist, v);
    double sqr = MathUtils::dot(v,v);

    gsl_vector_free(v);

    double mult = 0.5 * data_train.nbEx * param_q*param_q/param_C;

    ParamsDelta fctparams = { mult, dot, sqr, w1, w2 };

    // Compute the values of F(delta) at the limit of the possible interval
    double valInf = fctDelta(-w1+EPS_BRENT, &fctparams);
    double valSup = fctDelta( w2-EPS_BRENT, &fctparams);

    // As F(delta) is always decreasing or always increasing, the zero of F(delta)
    // is not in the interval iif valInf and vInf have the same sign.  Then, the
    // optimal value of delta is a limit of the interval
    if (valInf*valSup > 0.0)
    {
        return valInf > 0.0 ? w2-EPS_BRENT : -w1+EPS_BRENT;
    }

    // Brent's method initialization
    int iter = 0, max_iter = 100;

    double x = 0.0;
    const gsl_root_fsolver_type *T;

    gsl_root_fsolver *s;
    gsl_function FDF;
    FDF.function = &fctDelta;
    FDF.params = &fctparams;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set (s, &FDF, -w1+EPS_BRENT, w2-EPS_BRENT );

    // Brent's method
    double x0;
    int status1;
    int status2;
    do
    {
        ++iter;
        gsl_root_fsolver_iterate (s);
        x0 = x;
        x = gsl_root_fsolver_root (s);
        status1 = gsl_root_test_delta (x, x0, 0, 1e-10);
        status2 = gsl_root_test_residual ( fctDelta( x, &fctparams ), 1e-10 );
    }
    while ((status1 == GSL_CONTINUE || status2 == GSL_CONTINUE)&& iter < max_iter);

    gsl_root_fsolver_free(s);

    m_maxBrent = max(iter, m_maxBrent);

    return x;
}


// Function evaluated by Brent root-finding method
double CPbscNonAlignLearner::fctDelta(double x, void *_params)
{
    ParamsDelta* P = (ParamsDelta*)_params;

    return P->mult * log( (P->w2 - x) / (P->w1 + x))
             - x*P->sqr - P->dot;
}


// Compute objective function cost value
double CPbscNonAlignLearner::calcCost(double* _ptrKL /*= NULL*/)
{
    groupWeights();

    gsl_vector* vMargins = gsl_vector_alloc(data_train.nbEx);
    MathUtils::mvProduct(vMargins, data_train.X, m_vGroupWeights, false);
    MathUtils::multiply(vMargins, data_train.Y);

    double loss = 0.0;
    double tmp;

    for (int i = 0; i < data_train.nbEx; ++i)
    {
        tmp  =  1.0 - gsl_vector_get(vMargins, i) / param_q;
        loss += tmp*tmp;
    }
    
    gsl_vector_free(vMargins);

    double KL = log(2*data_train.nbFt);
    double w_i;

    for (int i = 0; i < 2*data_train.nbFt; ++i)
    {
        w_i = gsl_vector_get(m_vWeights, i);
        if (w_i > 0)
            KL += w_i * log(w_i);
    }

    return param_C * loss + data_train.nbEx * KL;
}


// Regroup complementary weights (vector of size 2*m) in a regular weight vector (of size m)
void CPbscNonAlignLearner::groupWeights()
{
    gsl_vector_view vFirstPart  = gsl_vector_subvector(m_vWeights, 0, data_train.nbFt);
    gsl_vector_view vSecondPart = gsl_vector_subvector(m_vWeights, data_train.nbFt, data_train.nbFt);

    MathUtils::substract(m_vGroupWeights, &vFirstPart.vector, &vSecondPart.vector);
}


// Initialize log file (open file and write headers)
void CPbscNonAlignLearner::initLog()
{
    m_log.init(param_sLogFile.c_str());
    m_log.begin();

    vector<string> header;
    header.push_back("Iter");
    header.push_back("Cost");
    header.push_back("maxDelta");
    header.push_back("maxBrent");
    header.push_back("TrainRisk");
    header.push_back("TestRisk");

    m_log.createHeader(header);

}


// Finalize log file (close file)
void CPbscNonAlignLearner::finalizeLog()
{
    m_log.end();
}


// Write a line in the log file
void CPbscNonAlignLearner::writeLog()
{
    StrValueMap map;

    m_cost = calcCost();

   groupWeights();
   ((CLinearClassifier*)m_pClassifier)->setWeights(m_vGroupWeights);

    map["Iter"]         = m_iter;
    map["Cost"]         = m_cost;
    map["maxDelta"]     = m_maxDelta;
    map["maxBrent"]     = m_maxBrent;

    map["TrainRisk"]    = m_pClassifier->calcRisk(data_train);

    if (m_hasTestData)
    {
        map["TestRisk"] = m_pClassifier->calcRisk(data_test);
    }

    m_log.write(map);

    m_maxBrent = 0;
}
