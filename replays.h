#include <string>
#include <iostream>
#include <stdlib.h>
#include "x_values.h"

std::string pruneOpeningVal(std::string& input)
{
    // get
    std::string::size_type loc = input.find("Protoss_Opening");
    if (loc == std::string::npos)
        loc = input.find("Zerg_Opening");
    if (loc == std::string::npos)
        loc = input.find("Terran_Opening");
    if (loc == std::string::npos)
    {
        std::cout << "ERROR: no Opening label" << std::endl;
        return "";
    }
    std::string::size_type begin = input.find_first_of(' ', loc);
    std::string::size_type end = input.find_first_of(';', begin);
    if (begin == std::string::npos || end == std::string::npos)
    {
        std::cout << "ERROR: bad Opening label formatting" << std::endl;
        return "";
    }
    std::string r = input.substr(begin+1, end-begin-1);
    // remove
    input.erase(loc, end-loc+1);
    return r;
}

int buildingInd(const char* cstr)
{
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
    {
        if (!strcmp(cstr, protoss_buildings_name[i]))
            return i;
    }
    return -1;
}

void getBuildings(std::string str, std::map<unsigned int, Protoss_Buildings>& b)
{
    while (str.length())
    {
        std::string::size_type begin = str.find_first_of(' ', 0);
        std::string::size_type end = str.find_first_of(';', begin);
        b.insert(std::make_pair<unsigned int, Protoss_Buildings>(
                    atoi(str.substr(begin+1, end-begin-1).c_str()), 
                    (Protoss_Buildings)buildingInd(str.substr(0, begin).c_str())));
        str.erase(0, end+1);
    }
}

