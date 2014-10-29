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


#ifndef FILE_UTILS_H
#define	FILE_UTILS_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include "StrValue.h"

namespace FileUtils
{

int         readLines(const char* _sFilename, std::vector< std::string >& _refLines);

struct STabInfo
{
   int  nbLines;
   int  minNbCols;
   int  maxNbCols;
   bool bValid;
};

template<class TYPE>
STabInfo    readTab(const char* _sFilename, std::vector< std::vector<TYPE> >& _refTab);

StrValueMap readStrValueMap(const char* _sFilename);

bool        saveStrValueMap(const StrValueMap& _map, const char* _sFilename);

void        writeStrValueMap(const StrValueMap& _map, std::ostream& _out);

std::string trim(const std::string& _str, const std::string& _drop = " \t\r");

std::vector<std::string> splitToArray(const std::string& _str, const std::string& _separators);

std::vector<CStrValue> parseCmdLine(StrValueMap & _argMap, int _argc, char* _argv[]);
std::vector<CStrValue> parseCmdLine(StrValueMap & _argMap, int _argc, char* _argv[], bool & _bHelp);


template<class TYPE>
STabInfo readTab(const char* _sFilename, std::vector< std::vector<TYPE> >& _refTab)
{
    STabInfo info = {0,0,0,false};
    
    // Opening file
    std::ifstream file(_sFilename);
    if ( !file.is_open() )
        return info;

    std::string strLine;
    int         nbCols;
    TYPE        value;

    // Read file one line at the time
    while ( !file.eof() )
    {
        getline(file, strLine);
        strLine = trim(strLine);

        if( strLine.empty() )  // Skip empty lines
            continue;

        std::stringstream stream;
        _refTab.push_back( std::vector<TYPE>() );
        stream << strLine;

        // Read line one column at the time
        nbCols = 0;
        while ( !stream.eof() ) 
        {
            stream >> value;
            _refTab.back().push_back(value);
            nbCols++;
        }

        info.nbLines++;
        info.minNbCols = info.nbLines > 1 ? std::min(info.minNbCols, nbCols) : nbCols;
        info.maxNbCols = info.nbLines > 1 ? std::max(info.maxNbCols, nbCols) : nbCols;
    }

    file.close();

    info.bValid = true;
    return info;
}  


template<class TYPE>
bool writeTab(const char* _sFilename, const std::vector< std::vector<TYPE> >& _refTab, char _separator = '\t')
{
    // Opening file
    std::ofstream file(_sFilename);
    if ( !file.is_open() )
        return false;

    typename std::vector< std::vector<TYPE> >::iterator  iterLine;
    typename std::vector<TYPE>::iterator                 iterCol;

    // Write file one line at the time
    for (iterLine = _refTab.begin(); iterLine != _refTab.end(); ++iterLine)
    {
        for (iterCol = iterLine->begin(); iterCol != iterLine->end(); ++iterCol)
        {
            if (iterCol != iterLine->begin())
                file << _separator;

            file << (*iterCol);
        }
        file << "\n";
    }

    file.close();

    return true;
}

} // namespace FileUtils

#endif	// FILE_UTILS_H
