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


#define  STR_APPNAME "CLASSIFICATION FUNCTION"
#include "common.h"

#include "Classifiers/LinearClassifier.h"

using namespace std;

const char* STR_USAGE =
    "Usage: pbsc_classify [-label <value>] train_file test_file [model_file] [prediction_file] \n"
    "\n"
    "Required parameters: \n"
    "    train_file      Training dataset file  (tab/space separated, one exemple per line, \n"
    "                                            first column contains -1/+1 labels) \n"
    "    test_file       Testing dataset file   (same format than the training dataset file) \n"
    "\n"
    "Optionnal parameters: \n"
    "    model_file      Classifier file name outputed by the learner (default='classifier.ini') \n"
    "    prediction_file Write predictions into that file \n"
    "\n"
    "    -label          Indicates if the test file contains label (0=no label, default=1) \n";


int main(int argc, char **argv)
{
    // Print header
    cout << STR_HEADER << endl;

    // Parse parameters
    StrValueMap argMap;
    argMap["label"] = true;

    bool bHelp;
    vector<CStrValue> new_argv = FileUtils::parseCmdLine(argMap, argc, argv, bHelp);
    int new_argc = new_argv.size();

    if (bHelp || new_argc < 3)
        ERROR( STR_USAGE );

    CLinearClassifier classifier(0);
    CKernel kernel;

    cout << "* Loading classifier file..." << endl;
    string sModel = (new_argc > 3) ? new_argv[3] : "classifier.ini";
    StrValueMap map = FileUtils::readStrValueMap(sModel.c_str());

    classifier.unserialize(map);
    cout << "  Weight vector cardinality: " << classifier.getCardinality() << endl;

    kernel.unserialize(map);
    StrValueMap kernelMap = kernel.serialize();
    cout << "  Kernel type: " << kernelMap["kernel"] << endl;

    // Load dataset files
    CDataMatrix train, test;

    cout << "* Loading train file..." << endl;
    if ( train.loadFromFile( new_argv[1].c_str() ) )
        cout << "  " << train.nbEx << " examples loaded." << endl;
    else
        ERROR("  Error with file '" << new_argv[1] << "'.");


    cout << "* Loading test file..." << endl;
    if( test.loadFromFile( new_argv[2].c_str(), (bool)argMap["label"] ) )
        cout << "  " << test.nbEx << " examples loaded." << endl;
    else
        ERROR("  Error with file '" << new_argv[2] << "'.");

    // Creating Kernel Matrices
    CDataMatrix Ktest;

    cout << "* Creating Kernel Matrix... " << endl;
    Ktest = createKernelMatrix(test, train, kernel);
    cout << "  Test matrix  : " << Ktest.nbEx << " x " << Ktest.nbFt << " elements." << endl;

    // Compute classification
    gsl_vector* vPred = gsl_vector_alloc(test.nbEx);

    if ( (bool)argMap["label"] )
    {
        cout << "* Testing..." << endl;
        cout << "Risk = " << classifier.calcRisk(Ktest, vPred) << endl;
    }
    else
    {
        cout << "* Predicting labels..." << endl;

        if (new_argc < 4)
            cout << "  Warning: It is useless to predict if you do not write the result in a file!" << endl;
        else
            classifier.classify(Ktest, vPred);
    }

    // Save predictions
    if (new_argc > 4)
    {
        cout << "Save predictions..." << endl;

        FILE* out = fopen(new_argv[4].c_str(), "wt");
        gsl_vector_fprintf(out, vPred,"%f");
        fclose(out);
    }

    // Desallocate memory
    gsl_vector_free(vPred);
    classifier.free();
    Ktest.free();
    test.free();
    train.free();
    return EXIT_SUCCESS;
}
