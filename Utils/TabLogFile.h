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


#ifndef TAB_LOG_FILE_H
#define TAB_LOG_FILE_H

#include <vector>
#include "Utils/StrValue.h"

class CTabLogFile
{
public:

    CTabLogFile();
    virtual ~CTabLogFile() {};

    void    init(const char* _strFilename);
    bool    begin();
    bool    end();

    bool    createHeader(const std::vector<std::string>& _header);
    bool    write(const StrValueMap& _values);
protected:

    std::string                 m_strFilename;
    std::ofstream*              m_ptrFile;

    std::vector<std::string>    m_header;
    bool                        m_bEmpty;
};

#endif // TAB_LOG_FILE_H
