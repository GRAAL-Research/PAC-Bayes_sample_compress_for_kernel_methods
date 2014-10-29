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


#include "Learner.h"

using namespace std;

// Constructor (default)
CLearner::CLearner()
{
    m_hasTestData       = false;
    m_pClassifier       = NULL;
}


// Set algorithm parameters
void CLearner::setParameters(const StrValueMap& _params)
{
    setParam(_params, "verbose",    param_bVerbose,     false                   );
    setParam(_params, "writeLog",   param_bWriteLog,    true                    );
    setParam(_params, "log",        param_sLogFile,     string("learner.log")   );

    if ( param_sLogFile == "0" )
        param_bWriteLog = false;
}


// Get algorithm statistics (after learning)
StrValueMap CLearner::getStats()
{
    StrValueMap stats;

    stats["time"] = (int)(clock() / CLOCKS_PER_SEC);

    return stats;
}


// Set training dataset
void  CLearner::setTrainData(const CDataMatrix& _trainData)
{
    data_train = _trainData;
}


// Set testing dataset
void  CLearner::setTestData(const CDataMatrix& _testData)
{
    data_test     = _testData;
    m_hasTestData = true;
}

