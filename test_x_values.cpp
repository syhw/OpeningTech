#include "x_values.h"
#include "enums_name_tables.h"
#include <iostream>
#include <set>
#include <vector>

#define PRINTALL
#define FILE_OUTPUT
#ifdef FILE_OUTPUT
#include <fstream>
#endif


#ifdef FILE_OUTPUT
void print_set(const std::set<Terran_Buildings>& s, std::ofstream& fout)
#else
void print_set(const std::set<Terran_Buildings>& s)
#endif
{
    for (std::set<Terran_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
#ifdef FILE_OUTPUT
        fout << terran_buildings_name[*it] << " ";
#else
        std::cout << terran_buildings_name[*it] << " ";
#endif
#ifdef FILE_OUTPUT
    fout << std::endl;
#else
    std::cout << std::endl;
#endif
}

#ifdef FILE_OUTPUT
void print_set(const std::set<Protoss_Buildings>& s, std::ofstream& fout)
#else
void print_set(const std::set<Protoss_Buildings>& s)
#endif
{
    for (std::set<Protoss_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
#ifdef FILE_OUTPUT
        fout << protoss_buildings_name[*it] << " ";
#else
        std::cout << protoss_buildings_name[*it] << " ";
#endif
#ifdef FILE_OUTPUT
    fout << std::endl;
#else
    std::cout << std::endl;
#endif
}

#ifdef FILE_OUTPUT
void print_set(const std::set<Zerg_Buildings>& s, std::ofstream& fout)
#else
void print_set(const std::set<Zerg_Buildings>& s)
#endif
{
    for (std::set<Zerg_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
#ifdef FILE_OUTPUT
        fout << zerg_buildings_name[*it] << " ";
#else
        std::cout << zerg_buildings_name[*it] << " ";
#endif
#ifdef FILE_OUTPUT
    fout << std::endl;
#else
    std::cout << std::endl;
#endif
}

int main(int argc, char* argv[])
{
    std::vector<std::set<Terran_Buildings> > terran = generate_terran_X_values();
    std::vector<std::set<Protoss_Buildings> > protoss = generate_protoss_X_values();
    std::vector<std::set<Zerg_Buildings> > zerg = generate_zerg_X_values();
#ifdef FILE_OUTPUT
std::ofstream terran_fout("terran_possible_tech_trees.txt");
std::ofstream protoss_fout("protoss_possible_tech_trees.txt");
std::ofstream zerg_fout("zerg_possible_tech_trees.txt");
#endif

    std::set<std::set<Terran_Buildings> > terran_verif;
    for (std::vector<std::set<Terran_Buildings> >::const_iterator it
            = terran.begin(); it != terran.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set(*it, terran_fout);
#else
        print_set(*it);
#endif
#endif
        terran_verif.insert(*it);
    }

    if (terran.size() != terran_verif.size()) 
        std::cout << "TEST FAIL" << std::endl;
    std::cout << "Terran, printed: " << terran.size() << " sets == tech trees" << std::endl;

    std::set<std::set<Protoss_Buildings> > protoss_verif;
    for (std::vector<std::set<Protoss_Buildings> >::const_iterator it
            = protoss.begin(); it != protoss.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set(*it, protoss_fout);
#else
        print_set(*it);
#endif
#endif
        protoss_verif.insert(*it);
    }

    if (protoss.size() != protoss_verif.size()) 
        std::cout << "TEST FAIL" << std::endl;
    std::cout << "Protoss, printed: " << protoss.size() << " sets == tech trees" << std::endl;

    std::set<std::set<Zerg_Buildings> > zerg_verif;
    for (std::vector<std::set<Zerg_Buildings> >::const_iterator it
            = zerg.begin(); it != zerg.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set(*it, zerg_fout);
#else
        print_set(*it);
#endif
#endif
        zerg_verif.insert(*it);
    }

    if (zerg.size() != zerg_verif.size()) 
    {
        std::cout << "TEST FAIL" << std::endl;
        std::cout << zerg.size() << " " << zerg_verif.size() << std::endl;
    }
    std::cout << "Zerg, printed: " << zerg.size() << " sets == tech trees" << std::endl;

    return 0;
}

