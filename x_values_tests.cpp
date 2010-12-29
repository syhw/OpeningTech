#include "x_values.h"

int main(int argc, char* argv[])
{
    std::vector<std::set<Terran_Buildings> > terran = generate_terran_X_values();
    std::vector<std::set<Protoss_Buildings> > protoss = generate_protoss_X_values();
    std::vector<std::set<Zerg_Buildings> > zerg = generate_zerg_X_values();
}

