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


#define STR_APPNAME "ALIGNED CASE  (aka PBSC-A)"
#include "common.h"

#include "Learners/PbscAlignLearner.h"

using namespace std;

const char* STR_USAGE =
    "Usage: pbsc_align [-first_parameter <value>] ... [-last_parameter <value>] train_file [test_file] \n"
    "\n"
    "Required parameters: \n"
    "    train_file      Training dataset file  (tab/space separated, one exemple per line, \n"
    "                                            first column contains -1/+1 labels) \n"
    "\n"
    "Optionnal parameters: \n"
    "    test_file       Testing dataset file   (same format than the training dataset file) \n"
    "\n"
    "    -q              Minimum of the quadratic risk (default=0.02) \n"
    "    -kernel         Kernel function, ie one of the following : (default='RBF')\n"
    "                        'LINEAR' : Linear kernel      k(x,y) = x*y \n"
    "                        'RBF'    : Gaussian kernel    k(x,y) = exp(-gamma ||x-y||^2) \n"
    "                        'POLY'   : Polynomial kernel  k(x,y) = (s x*y+c)^d \n"
    "                        'TANH'   : Sigmoid kernel     k(x,y) = tanh(s x*y + c) \n"
    "    -kernel.gamma   Kernel parameter gamma (default=0.1) \n"
    "    -kernel.d       Kernel parameter d (default=2.0) \n"
    "    -kernel.s       Kernel parameter s (default=1.0) \n"
    "    -kernel.c       Kernel parameter c (default=0.0) \n"
    "\n"
    "    -stopCriteria   Stopping criteria (default=1e-16) \n"
    "    -nIter          Maximum number of iterations (defaut=2e5) \n"
    "    -seed           Random generator seed (defaut=<System time>) \n"
    "\n"
    "    -writeStep      Write log file at each n iterations (default=100) \n"
    "    -log            Log file name (0=none, default='learner.log)\n"
    "    -stats          Statistics file name (0=none, default='results.ini')\n"
    "    -model          Classifier file name (0=none, default='classifier.ini') \n"
    "    -config         Parameters file name (0=none, default='config.ini') \n"
    "\n"
    "Examples: \n"
    "    (1) Use the sigmoid (tanh) kernel with s=2 and c=0.1 on USvotes dataset: \n"
    "\n"
    "       ./pbsc_align -kernel TANH -kernel.s 2 -kernel.c 0.1 USvotes_train.dat USvotes_test.dat\n"
    "\n"
    "    (2) Use the rbf kernel with gamma=0.5 on USvotes dataset, don't write any log file and \n"
    "        don't compute a test risk for a faster execution: \n"
    "\n"
    "       ./pbsc_align -kernel.gamma 0.5 -log 0 USvotes_train.dat \n"
    "\n"
    "    (3) Load all parameters from 'example3.ini' file, except the q value (fix q=0.2 instead): \n"
    "\n"
    "       ./pbsc_align -config exemple3.ini -q 0.2 USvotes_train.dat \n"
    "\n"
    "    (4) Carefully write the trace of the 100 first iterations into 'USvotes_100.log' file: \n"
    "\n"
    "       ./pbsc_align -nIter 100 -writeStep 1 -log USvotes_100.log USvotes_train.dat USvotes_test.dat \n"
    ;


int main(int argc, char **argv)
{
    // Print header
    cout << STR_HEADER << endl;

    // Parse parameters
    StrValueMap argMap, argDefault;
    argDefault["gamma"]     = 1.0;
    argDefault["config"]    = "config.ini";
    argDefault["stats"]     = "results.ini";
    argDefault["model"]     = "classifier.ini";

    bool bHelp;
    vector<CStrValue> new_argv = FileUtils::parseCmdLine(argMap, argc, argv, bHelp);
    int new_argc = new_argv.size();

    if (bHelp || new_argc < 2)
        ERROR( STR_USAGE );

    // Load config file
    string strParam = argMap["config"];
    if (strParam != "0")
    {
        cout << "* Loading config file..." << endl;
        StrValueMap config = FileUtils::readStrValueMap( strParam.c_str() );
        argMap.insert(config.begin(), config.end());
        cout << "  " << config.size() << " parameters read." << endl;

    }
    
    argMap.insert(argDefault.begin(), argDefault.end());

    // Load dataset files
    CDataMatrix train, test;

    cout << "* Loading train file..." << endl;
    if ( train.loadFromFile( new_argv[1].c_str() ) )
        cout << "  " << train.nbEx << " examples loaded." << endl;
    else
        ERROR("  Error with file '" << new_argv[1] << "'.");


    if (new_argc > 2)
    {
        cout << "* Loading test file..." << endl;
        if( test.loadFromFile( new_argv[2].c_str() ) )
            cout << "  " << test.nbEx << " examples loaded." << endl;
        else
            ERROR("  Error with file '" << new_argv[2] << "'.");
    }



    // Creating Kernel Matrices
    CDataMatrix Ktrain, Ktest;

    cout << "* Creating Kernel Matrices... " << endl;
    CKernel     kernel(argMap);
    StrValueMap kMap = kernel.serialize();

    Ktrain = createKernelMatrix(train, train, kernel);
    cout << "  Train matrix : " << Ktrain.nbEx << " x " << Ktrain.nbFt << " elements." << endl;

    if (test.nbEx > 0)
    {
        Ktest = createKernelMatrix(test, train, kernel);
        cout << "  Test matrix  : " << Ktest.nbEx << " x " << Ktest.nbFt << " elements." << endl;
    }

    // Learn
    CPbscAlignLearner algo;

    algo.setTrainData(Ktrain);
    if (Ktest.nbEx > 0)
        algo.setTestData(Ktest);

    algo.setParameters(argMap);
    algo.init();

    cout << "* Learning..." << endl;
    CClassifier* classifier = algo.learn();

    StrValueMap stats = algo.getStats();
    stats.insert(kMap.begin(), kMap.end());

    cout << "* Testing..." << endl;
    stats["Train Risk"] = classifier->calcRisk(Ktrain);

    if (Ktest.nbEx > 0)
        stats["Test Risk"]  = classifier->calcRisk(Ktest);

    cout << endl;

    FileUtils::writeStrValueMap(stats, cout);

    strParam = (string)argMap["stats"];
    if (strParam != "0")
        FileUtils::saveStrValueMap(stats, strParam.c_str() );

    strParam = (string)argMap["model"];
    if (strParam != "0")
    {
        StrValueMap srlz = classifier->serialize();
        srlz.insert(kMap.begin(), kMap.end());
        FileUtils::saveStrValueMap(srlz, strParam.c_str() );
    }

    // Freeing memory
    classifier->free();
    algo.free();
    train.free();
    test.free();
    Ktrain.free();
    Ktest.free();

    return EXIT_SUCCESS;
}


