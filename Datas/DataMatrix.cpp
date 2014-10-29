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


#include "DataMatrix.h"
#include "Utils/FileUtils.h"
#include "Utils/MathUtils.h"
#include <vector>
#include <algorithm>
#include "gsl/gsl_matrix.h"

using namespace std;


// Constructor
CDataMatrix::CDataMatrix()
{
    X = NULL;
    Y = NULL;
    
    nbFt = 0;
    nbEx = 0;
}


// Allocate memory
void CDataMatrix::init(int _nbEx, int _nbFt, bool _bLabelVector /*= true*/)
{
    nbEx = _nbEx;
    nbFt = _nbFt;

    X = gsl_matrix_alloc(nbEx, nbFt);

    if (_bLabelVector)
        Y = gsl_vector_alloc(nbEx);
    else
        Y = NULL;
}


// Desallocate memory
void CDataMatrix::free()
{
    if (X != NULL)  gsl_matrix_free(X);
    if (Y != NULL)  gsl_vector_free(Y);

    X = NULL;
    Y = NULL;

    nbEx = 0;
    nbFt = 0;
}


// Load a dataset file
// one line by example; first column contains labels, if any.
// (specified _bFirstColumnAsLabels=false if the data is unlabled)
int CDataMatrix::loadFromFile(const char* _sFilename, bool _bFirstColumnAsLabels /*= true*/)
{
    vector< vector<double> > tab;

    FileUtils::STabInfo info = FileUtils::readTab(_sFilename, tab);

    if (info.maxNbCols < 1)
    {
        cerr << "[CDataMatrix::loadFromFile] Error while reading file." << endl;
        return 0;
    }


    if (info.minNbCols != info.maxNbCols)
    {
        cerr << "[CDataMatrix::loadFromFile] The file contains lines of various size." << endl;
        return 0;
    }

    int jFirst = _bFirstColumnAsLabels ? 1 : 0;

    init(   info.nbLines,
            info.minNbCols-jFirst,
            _bFirstColumnAsLabels    );

    for (int i = 0; i < nbEx; ++i)
    {
        if (_bFirstColumnAsLabels)
            gsl_vector_set(Y, i, tab[i][0]);

        for (int j = 0; j < nbFt; ++j)
            gsl_matrix_set(X, i, j, tab[i][j+jFirst]);
    }

    return info.nbLines;
}


// Save a dataset file
// one line by example; first column contains labels, if any.
bool CDataMatrix::saveToFile(const char* _sFilename)
{
    // Opening file
    std::ofstream file(_sFilename);
    if ( !file.is_open() )
        return false;

    // Write file one line at the time
    for (int i = 0; i < nbEx; ++i)
    {
        if (Y != NULL)
            file << gsl_vector_get(Y, i) << "\t";

        for (int j = 0; j < nbFt; ++j)
        {
            if (j > 0)
                file << "\t";

            file << gsl_matrix_get(X, i, j);
        }


        file << "\n";
    }

    file.close();

    return true;
}


// Make a new copy of this dataset
CDataMatrix CDataMatrix::duplicate()
{
    CDataMatrix newData;

    newData.nbEx = nbEx;
    newData.nbFt = nbFt;

    if (X != NULL)
    {
        newData.X = gsl_matrix_alloc(nbEx, nbFt);
        gsl_matrix_memcpy(newData.X, X);
    }

    if (Y != NULL)
    {
        newData.Y = gsl_vector_alloc(nbEx);
        gsl_vector_memcpy(newData.Y, Y);
    }

    return newData;
}


// Make a new copy of this dataset and select desired examples (matrix rows):
// - If _bInverse==true, keep only examples of indexes in _vIndexes.
// - Otherwise, keep only examples of indexes not in _vIndexes.
CDataMatrix CDataMatrix::copyExamples(vector<int> _vIndexes, bool _bInverse /*= false*/)
{
    sort(_vIndexes.begin(), _vIndexes.end());
    unique(_vIndexes.begin(), _vIndexes.end());

    CDataMatrix newData;

    newData.init( _bInverse ? nbEx - _vIndexes.size() : _vIndexes.size(),
                  nbFt );

    bool bCopy;
    int j=0;
    int k=0;
    for (int i = 0; i < nbEx && j < newData.nbEx; ++i)
    {
        bCopy = false;
        if (i == _vIndexes[j])
        {
            ++j;
            if (!_bInverse)
                bCopy = true;
        }
        else if (_bInverse)
        {
            bCopy = true;
        }

        if (bCopy)
        {
            gsl_vector_view row = gsl_matrix_row(X, i);
            gsl_matrix_set_row(newData.X, k, &(row.vector));
            gsl_vector_set(newData.Y, k, gsl_vector_get(Y, i));
            ++k;
        }
    }

    return newData;
}


// Make a new copy of this dataset and select desired attributes (matrix columns)
// - If _bInverse==true, keep only attributes of indexes in _vIndexes.
// - Otherwise, keep only examples of attributes not in _vIndexes.
CDataMatrix CDataMatrix::copyAttributes(vector<int> _vIndexes, bool _bInverse /*= false*/)
{
    sort(_vIndexes.begin(), _vIndexes.end());
    unique(_vIndexes.begin(), _vIndexes.end());

    CDataMatrix newData;

    newData.init( nbEx,
                  newData.nbFt = _bInverse ? nbFt - _vIndexes.size() : _vIndexes.size() );

    bool bCopy;
    int j=0;
    int k=0;
    for (int i = 0; i < nbFt && j < newData.nbFt; ++i)
    {
        bCopy = false;
        if (i == _vIndexes[j])
        {
            ++j;
            if (!_bInverse)
                bCopy = true;
        }
        else if (_bInverse)
        {
            bCopy = true;
        }

        if (bCopy)
        {
            gsl_vector_view col = gsl_matrix_column(X, i);
            gsl_matrix_set_col(newData.X, k, &(col.vector));
            ++k;
        }
    }

    MathUtils::assign(newData.Y, Y);

    return newData;
}
