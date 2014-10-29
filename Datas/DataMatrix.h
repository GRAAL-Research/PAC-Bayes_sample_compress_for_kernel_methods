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


#ifndef DATA_MATRIX_H
#define	DATA_MATRIX_H

#include <gsl/gsl_matrix.h>
#include <vector>

class CDataMatrix
{
public:
    // Dataset values (declared 'public' for more commodity)
    gsl_matrix*     X;  // Features matrix (one example per line)
    gsl_vector*     Y;  // Labels vector
    int             nbEx, nbFt; // Matrix size [nb examples]x[nb features]

public:
    // Constructor / Destructor (the cycle of life!)
    CDataMatrix();
    virtual ~CDataMatrix()  {}

    // Allocate / Desallocate memory
    void        init(int _nbEx, int _nbFt, bool _bLabelVector = true);
    void        free();

    // File management (one line by example; first column contains labels, if any)
    int         loadFromFile(const char* _sFilename, bool _bLastColumnAsLabels = true);
    bool        saveToFile(const char* _sFilename);

    // Set / Get an attribute value (example i, attribute j)
    void        setX(int _i, int _j, double _value);
    double      getX(int _i, int _j) const;

    // Set / Get a label value (example i)
    void        setY(int _i, double _value);
    double      getY(int _i) const;

    // Set / Get a whole matrix line (example i)
    gsl_vector  getRow(int _i) const;
    void        setRow(int _i, double _value);

    // Set / Get a whole matrix column (example j)
    gsl_vector  getCol(int _j) const;
    void        setCol(int _j, double _value);

    // Create a new matrix from this one
    // vIndexes allows to select a subset of examples / attributes
    // If bInverse==true, vIndexes indicates examples / attributes that we DO NOT want.
    CDataMatrix duplicate();
    CDataMatrix copyExamples(std::vector<int> _vIndexes, bool _bInverse = false);
    CDataMatrix copyAttributes(std::vector<int> _vIndexes, bool _bInverse = false);


private:

};


inline void CDataMatrix::setX(int _i, int _j, double _value)
{
    gsl_matrix_set(X, _i, _j, _value);
}

inline double CDataMatrix::getX(int _i, int _j) const
{
    return gsl_matrix_get(X, _i, _j);
}

inline void CDataMatrix::setY(int _i, double _value)
{
    gsl_vector_set(Y, _i, _value);
}

inline double CDataMatrix::getY(int _i) const
{
    return gsl_vector_get(Y, _i);
}

inline gsl_vector CDataMatrix::getRow(int _i) const
{
    return gsl_matrix_row(X, _i).vector ;
}

inline void CDataMatrix::setRow(int _i, double _value)
{
    gsl_vector v = getRow(_i);
    gsl_vector_set_all(&v, _value);
}

inline gsl_vector CDataMatrix::getCol(int _j) const
{
    return gsl_matrix_column(X, _j).vector;
}

inline void CDataMatrix::setCol(int _j, double _value)
{
    gsl_vector v = getCol(_j);
    gsl_vector_set_all(&v, _value);
}

#endif	// DATA_MATRIX_H
