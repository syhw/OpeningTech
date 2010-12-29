#include <pl.h>
#include <iostream>
#include <vector>
#include "enums_name_tables.h"
using namespace std;

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
    plProbValue tableOpeningProtoss[] = {0.4, 0.6};
    plProbTable P_OpeningProtoss(OpeningProtoss, tableOpeningProtoss);

    //li Specification of P(B)
    plProbValue tableB[] = {0.18, 0.82};
    plProbTable P_B(B, tableB);

    // Specification of P(C)
    plProbValue tableC[] = {0.75, 0.25};
    plProbTable P_C(C, tableC);

    // Specification of P(D | A B)
    plProbValue tableD_knowingA_B[] = {0.6, 0.4,  // P(D | [A=f]^[B=f])
        0.3, 0.7,  // P(D | [A=f]^[B=t])
        0.1, 0.9,  // P(D | [A=t]^[B=f])
        0.5, 0.5}; // P(D | [A=t]^[B=t])
    plDistributionTable P_D(D,A^B,tableD_knowingA_B);

    // Specification of P(E | C D)
    plProbValue tableE_knowingC_D[] = {0.59, 0.41,  // P(E | [C=f]^[D=f])
        0.25, 0.75,  // P(E | [C=f]^[D=t])
        0.8,  0.2,   // P(E | [C=t]^[D=f])
        0.35, 0.65}; // P(E | [C=t]^[D=t])
    plDistributionTable P_E(E,C^D,tableE_knowingC_D);

    /**********************************************************************
      DECOMPOSITION
     **********************************************************************/
    plJointDistribution jd(A^B^C^D^E, P_A*P_B*P_C*P_D*P_E);
    jd.draw_graph("bayesian_network.fig");
    cout<<"OK\n";
    /**********************************************************************
      PROGRAM QUESTION
     **********************************************************************/
    // Get the inferred conditional Distribution representing P(C B | E D)
    plCndDistribution CndP_CB;
    jd.ask(CndP_CB, C^B, E^D);

    // Create the value representing the evidence = [E=true]^[D=false]
    plValues evidence(E^D);
    evidence[E] = true;
    evidence[D] = false;

    // Get the Distribution representing P(C B | [E=true]^[D=false] )
    plDistribution P_CB;
    CndP_CB.instantiate(P_CB,evidence);

    // Get the normalized Distribution representing P(C B | [E=true]^[D=false] )
    plDistribution  T_P_CB;
    P_CB.compile(T_P_CB);

    // Display the result
    cout << T_P_CB << endl;
    plSerializer my_serializer();
    my_serializer.add_object("P_CB", P_CB);
    my_serializer.save("test.xml");

    // On Windows (Visual C++, MinGW) only.
#if defined(WIN32) || defined(_WIN32)
    cout << "Press any key to terminate..." << endl;
    getchar();
#endif

    return 0;
}
