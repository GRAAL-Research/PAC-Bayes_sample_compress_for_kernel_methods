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


#include "FileUtils.h"

#include <iomanip>

using namespace std;


namespace FileUtils
{

int readLines(const char* _sFilename, std::vector< std::string >& _refLines)
{
    int nbLines = 0;
    _refLines.clear();

    // Opening file
    std::ifstream file(_sFilename);
    if ( !file.is_open() )
        return nbLines;

    std::string strLine;

    // Read file one line at the time
    while ( !file.eof() )
    {
        getline(file, strLine);

        _refLines.push_back(strLine);
        ++nbLines;
    }

    file.close();

    return nbLines;
}


StrValueMap readStrValueMap(const char* _sFilename)
{
    StrValueMap ourMap;
    std::vector< std::string > lines;
    std::vector< std::string > fields;

    int nbLines = readLines(_sFilename, lines);

    for (int i = 0; i < nbLines; ++i)
    {
        fields = splitToArray(lines[i],  "#");

        if (fields.size() > 0)
            fields = splitToArray(fields[0], "=");

        if (fields.size() == 2)
        {
            ourMap[ trim(fields[0]) ] = trim(fields[1]);
        }
    }

    return ourMap;
}


bool saveStrValueMap(const StrValueMap& _map, const char* _sFilename)
{
    // Opening file
    std::ofstream file(_sFilename);
    if ( !file.is_open() )
        return false;

    writeStrValueMap(_map, file);

    file.close();

    return true;
}


void writeStrValueMap(const StrValueMap& _map, ostream& _out)
{
    StrValueMap::const_iterator iter;
    for (iter = _map.begin(); iter != _map.end(); ++iter)
        _out << setw(20) << left <<  iter->first << " = " << iter->second << std::endl;
}



string trim(const string& _str, const string& _drop /*= " \t\r"*/)
{
    size_t first = _str.find_first_not_of(_drop);
    size_t last  = _str.find_last_not_of(_drop) + 1;

    return (first==string::npos) ? "" : _str.substr(first, last-first);
}

vector<string> splitToArray(const string& _str, const string& _separators)
{
    int             posFirst = 0;
    int             posLast;
    bool            bEnd     = false;

    vector<string>  strArray;

    while (!bEnd)
    {
        posLast = _str.find_first_of(_separators, posFirst);

        if (posLast == -1)
        {
            bEnd    = true;
            posLast = _str.length();
        }

        if (posLast > posFirst)
        {
            strArray.push_back( _str.substr(posFirst, posLast - posFirst) );
        }

        posFirst = posLast + 1;
    }

    return strArray;
}


vector<CStrValue> parseCmdLine(StrValueMap & _argMap, int _argc, char* _argv[])
{
    bool dummy;
    return parseCmdLine(_argMap, _argc, _argv, dummy);
}

vector<CStrValue> parseCmdLine(StrValueMap & _argMap, int _argc, char* _argv[], bool & _bHelp)
{
    vector<CStrValue> argVector;
    _bHelp = false;
    int pos = 0;

    while (pos < _argc)
    {
        if (_argv[pos][0] == '-')
        {
            if (_argc > pos+1)
            {
                _argMap[ string( _argv[pos]+1 ) ] = CStrValue( _argv[pos+1] );
                pos += 2;
            }
            else if (string( _argv[pos] ) == "-h" || string( _argv[pos] ) == "--help" )
            {
                _bHelp = true;
                pos += 1;
            }
            else
                pos += 1;
        }
        else
        {
            argVector.push_back( _argv[pos] );
            pos += 1;
        }
    }

    return argVector;
}


} // namespace FileUtils
