#include "x_values.h"
#include "enums_name_tables.h"
#include <iostream>
#include <set>
#include <vector>

void print_set(const std::set<Terran_Buildings>& s)
{
    for (std::set<Terran_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
        std::cout << terran_buildings_name[*it] << " ";
    std::cout << std::endl;
}

void print_set(const std::set<Protoss_Buildings>& s)
{
    for (std::set<Protoss_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
        std::cout << protoss_buildings_name[*it] << " ";
    std::cout << std::endl;
}

void print_set(const std::set<Zerg_Buildings>& s)
{
    for (std::set<Zerg_Buildings>::const_iterator it = s.begin(); it != s.end(); ++it)
        std::cout << zerg_buildings_name[*it] << " ";
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    std::vector<std::set<Terran_Buildings> > terran = generate_terran_X_values();
    std::vector<std::set<Protoss_Buildings> > protoss = generate_protoss_X_values();
    std::vector<std::set<Zerg_Buildings> > zerg = generate_zerg_X_values();

    std::set<std::set<Terran_Buildings> > terran_verif;
    for (std::vector<std::set<Terran_Buildings> >::const_iterator it
            = terran.begin(); it != terran.end(); ++it)
    {
#ifdef PRINTALL
        print_set(*it);
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
        print_set(*it);
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
        print_set(*it);
#endif
        if (zerg_verif.count(*it))
            print_set(*it);
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

