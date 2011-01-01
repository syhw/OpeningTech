#include <pl.h>
#include <iostream>
#include <vector>
#include "enums_name_tables.h"
#include "x_values.h"
using namespace std;

void test_X_possible(plValues& lambda, const plValues& X_Obs_conj)
{
    // if X is possible w.r.t. observations
    lambda[0] = 1; // true
    // else
    lambda[0] = 0; // false
}

int main() 
{
    std::vector<std::string> terran_openings;
    terran_openings.push_back("fast_drop"); // facto-starport-control tower
    terran_openings.push_back("full_metal"); // facto x2 + machine shop
    terran_openings.push_back("MM"); // raxes-academy
    terran_openings.push_back("fast_expand"); // CC first
    terran_openings.push_back("1rax_FE"); // rax-CC
    terran_openings.push_back("mech_timing_push"); // facto-armory-facto
    terran_openings.push_back("fast_air"); // starport x2
    terran_openings.push_back("BBS"); // rax-rax-supply
    terran_openings.push_back("unkown");


    std::vector<std::string> protoss_openings;
    protoss_openings.push_back("fast_legs"); // core-citadel-gates-legs
    protoss_openings.push_back("fast_DT"); // citadel-archives-DT
    protoss_openings.push_back("fast_air"); // gate-core-stargates
    protoss_openings.push_back("fast_expand"); // nexus first
    protoss_openings.push_back("reaver"); // robot-support bay
    protoss_openings.push_back("standard"); // 2 gates-core-robo-observatory
    protoss_openings.push_back("goons"); // gate-core-gates-range (NonY)
    protoss_openings.push_back("proxy_gates"); // pylon-gates @enemy
    protoss_openings.push_back("photon_rush"); // forge-pylon @enemy
    protoss_openings.push_back("unknown");


    std::vector<std::string> zerg_openings;
    zerg_openings.push_back("fast_pool"); // 4-8 pools
    zerg_openings.push_back("lings"); // early pool + no peons @ extractor
    zerg_openings.push_back("fast_mutas"); // early extractor-lair-spire
    zerg_openings.push_back("mass_hydras"); // expand-hydra den
    zerg_openings.push_back("mutas_into_hydras"); // expand-lair-spire-hatch-hydra
    zerg_openings.push_back("fast_lurkers"); // early gaz-lair-hydra den
    zerg_openings.push_back("unknown");

    /**********************************************************************
      VARIABLES SPECIFICATION
     **********************************************************************/
    plSymbol X("X", plIntegerType(0, NB_PROTOSS_X_VALUES));
    std::vector<plSymbol> observed;
    plSymbol lambda("lambda", PL_BINARY_TYPE);
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
    {
        // what has been observed
        observed.push_back(plSymbol(protoss_buildings_name[i], PL_BINARY_TYPE));
    }
    //plSymbol OpeningTerran("OpeningTerran", VALUES);
    plSymbol OpeningProtoss("OpeningProtoss", plLabelType(protoss_openings));
    //plSymbol OpeningZerg("OpeningZerg", VALUES);

    plSymbol Time("Time", plIntegerType(1,600));

    /**********************************************************************
      PARAMETRIC FORM SPECIFICATION
     **********************************************************************/
    // Specification of P(OpeningProtoss)
    std::vector<plProbValue> tableOpeningProtoss;
    for (unsigned int i = 0; i < protoss_openings.size(); i++) 
        tableOpeningProtoss.push_back(1.0); // TOLEARN
    plProbTable P_OpeningProtoss(OpeningProtoss, tableOpeningProtoss, false);

    // Specification of P(X) (possible tech trees)
    std::vector<plProbValue> tableX;
    for (unsigned int i = 0; i < NB_PROTOSS_X_VALUES; i++)
        tableX.push_back(1.0); // TOLEARN
    plProbTable P_X(X, tableX, false);

    // Specification of P(O_1..NB_PROTOSS_BUILDINGS)
    std::vector<plProbTable> P_Observed;
    plProbValue tmp_table[] = {0.5, 0.5}; // TOLEARN (per observation)
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
        P_Observed.push_back(plProbTable(observed[i], tmp_table, true));

    // Specification of P(lambda | X, O_1..NB_PROTOSS_BUILDINGS)
    plVariablesConjunction conjObs;
    for (std::vector<plSymbol>::const_iterator it = observed.begin();
            it != observed.end(); ++it)
        conjObs ^= (*it);
    plExternalFunction coherence(lambda, X^conjObs, test_X_possible);
    plFunctionalDirac P_lambda(lambda, X^conjObs, coherence);
    // P_lambda = 1 if lambda=coherence(X, observed), 0 otherwise
    return 0;
}
