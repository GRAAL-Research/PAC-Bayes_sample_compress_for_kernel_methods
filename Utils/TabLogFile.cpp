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


#include "TabLogFile.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>

using namespace std;

CTabLogFile::CTabLogFile()
{
    m_bEmpty = true;
}


void CTabLogFile::init(const char* _strFilename)
{
    m_strFilename   = _strFilename;
}


bool CTabLogFile::begin()
{
    m_ptrFile = new ofstream(m_strFilename.c_str());

    if (!m_ptrFile->is_open())
    {
        cerr << "[CLogFile::begin] Unable to open file: " << m_strFilename << endl;
        delete m_ptrFile;
        m_ptrFile = NULL;
        return false;
    }

    return true;
}


bool CTabLogFile::end()
{
    if (m_ptrFile == NULL)
    {
        cerr << "[CLogFile::end] The file doesn't seem to be open." << endl;
        return false;
    }

    m_ptrFile->close();
    delete m_ptrFile;
    m_ptrFile = NULL;

    return true;
}


bool CTabLogFile::createHeader(const std::vector<std::string>& _header)
{
    if (m_ptrFile == NULL)
        return false;

    m_header = _header;

    int nb   = (int)m_header.size();

    for (int i = 0; i < nb; ++i)
    {
        (*m_ptrFile) << m_header[i].c_str() << "\t";
    }
    (*m_ptrFile) << "\n";
    m_ptrFile->flush();

    m_bEmpty = false;
    return true;
}


bool CTabLogFile::write(const StrValueMap &_values)
{
    if (m_ptrFile == NULL)
        return false;

    if (m_bEmpty)
    {
        m_header.clear();

        StrValueMap::const_iterator iter;
        for (iter = _values.begin(); iter != _values.end(); ++iter)
            m_header.push_back(iter->first);

        createHeader(m_header);
    }

    int nb   = (int)m_header.size();

    for (int i = 0; i < nb; ++i)
    {
        StrValueMap::const_iterator val = _values.find(m_header[i]);
        if (val != _values.end())
            (*m_ptrFile) << val->second.c_str() << "\t";
        else
            (*m_ptrFile) << "N/A\t";
    }

    (*m_ptrFile) << "\n";
    m_ptrFile->flush();

    return true;
}
