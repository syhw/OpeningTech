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


template<typename T>
#ifdef FILE_OUTPUT
void print_set(const std::set<int>& s, std::ofstream& fout)
#else
void print_set(const std::set<int>& s)
#endif
{
    for (std::set<int>::const_iterator it = s.begin(); it != s.end(); ++it)
    {
        Building tmpBuilding(static_cast<T>(*it)); 
#ifdef FILE_OUTPUT
        fout << tmpBuilding << " ";
#else
        std::cout << tmpBuilding << " ";
#endif
    }
#ifdef FILE_OUTPUT
    fout << std::endl;
#else
    std::cout << std::endl;
#endif
}

int main(int argc, char* argv[])
{
    std::ifstream fin1("testT.txt"); /// all protoss matches
    std::vector<std::set<int> > terran = get_X_values(fin1);
    std::ifstream fin2("testP.txt"); /// all protoss matches
    std::vector<std::set<int> > protoss = get_X_values(fin2);
    std::ifstream fin3("testZ.txt"); /// all protoss matches
    std::vector<std::set<int> > zerg = get_X_values(fin3);
#ifdef FILE_OUTPUT
std::ofstream terran_fout("terran_possible_tech_trees.txt");
std::ofstream protoss_fout("protoss_possible_tech_trees.txt");
std::ofstream zerg_fout("zerg_possible_tech_trees.txt");
#endif

    std::set<std::set<int> > terran_verif;
    for (std::vector<std::set<int> >::const_iterator it
            = terran.begin(); it != terran.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set<Terran_Buildings>(*it, terran_fout);
#else
        print_set<Terran_Buildings>(*it);
#endif
#endif
        terran_verif.insert(*it);
    }

    if (terran.size() != terran_verif.size()) 
        std::cout << "TEST FAIL" << std::endl;
    std::cout << "Terran, printed: " << terran.size() << " sets == tech trees" << std::endl;

    std::set<std::set<int> > protoss_verif;
    for (std::vector<std::set<int> >::const_iterator it
            = protoss.begin(); it != protoss.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set<Protoss_Buildings>(*it, protoss_fout);
#else
        print_set<Protoss_Buildings>(*it);
#endif
#endif
        protoss_verif.insert(*it);
    }

    if (protoss.size() != protoss_verif.size()) 
        std::cout << "TEST FAIL" << std::endl;
    std::cout << "Protoss, printed: " << protoss.size() << " sets == tech trees" << std::endl;

    std::set<std::set<int> > zerg_verif;
    for (std::vector<std::set<int> >::const_iterator it
            = zerg.begin(); it != zerg.end(); ++it)
    {
#ifdef PRINTALL
#ifdef FILE_OUTPUT
        print_set<Zerg_Buildings>(*it, zerg_fout);
#else
        print_set<Zerg_Buildings>(*it);
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

