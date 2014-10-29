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


#ifndef LEARNER_H
#define LEARNER_H

#include "Datas/DataMatrix.h"
#include "Classifiers/Classifier.h"
#include "Utils/StrValue.h"

class CLearner
{
public:
    // Constructor / Destructor (the cycle of life!)
    CLearner();
    virtual ~CLearner() { }

    // Allocate / Desallocate memory
    virtual void            init()  { }
    virtual void            free()  { }

    // Set algorithm parameters
    virtual void            setParameters(const StrValueMap& _params);

    // Set training (required) / testing (optional) datasets
    void                    setTrainData(const CDataMatrix& _trainData);
    void                    setTestData(const CDataMatrix& _testData);

    // Execute learning algorithm
    virtual CClassifier*    learn()     = 0;

    // Get resulting classifier (after learning)
    virtual CClassifier*    getClassifier() { return m_pClassifier; }

    // Get algorithm statistics (after learning)
    virtual StrValueMap     getStats();

protected:
    // Helper for setParameters function (see below)
    template <class T>
    void setParam(const StrValueMap& _map, const char* _key, T& _var, const T& _default);

    // Algorithm parametes
    bool                param_bVerbose;  // display more output
    bool                param_bWriteLog; // write a log file?
    std::string         param_sLogFile;  // log file name

    // Training / Testing sets
    CDataMatrix         data_train;
    CDataMatrix         data_test;
    bool                m_hasTestData;

    // Classifier produced after learning
    CClassifier*        m_pClassifier;
};


// Helper for setParameters function.
// _map : contains (key,value) pairs
// _key : name of the parameter (corresponding to a key in the map)
// _var : parameter variable (typically 'param_something')
// _default : parameter default value (if specified key is not in the map)
template <class T>
void CLearner::setParam(const StrValueMap& _map, const char* _key, T& _var, const T& _default)
{
    StrValueMap::const_iterator it = _map.find(_key); 
    _var = (it == _map.end()) ? _default : (T)(it->second);
}


#endif // LEARNER_H
