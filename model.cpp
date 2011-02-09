#include <pl.h>
#include <iostream>
#include <vector>
#include "enums_name_tables.h"
#include "x_values.h"
#include "replays.h"
using namespace std;

std::vector<std::vector<plProbValue> > mean_Time_Protoss_when_X_and_Opening;
std::vector<std::vector<plProbValue> > stddev_Time_Protoss_when_X_and_Opening;

void test_X_possible(plValues& lambda, const plValues& X_Obs_conj)
{
    // if X is possible w.r.t. observations
    lambda[0] = 1; // true
    // else
    lambda[0] = 0; // false
}

/*void mean_Time_Protoss(plValues& Time, const plValues& X_Opening)
{
    // mean_Time_Protoss[X][Opening]
    Time[0] = mean_Time_Protoss_when_X_and_Opening[X_Opening[0]][X_Opening[1]];
}

void stddev_Time_Protoss(plValues& Time, const plValues& X_Opening)
{
    // stddev_Time_Protoss[X][Opening]
    Time[0] = stddev_Time_Protoss_when_X_and_Opening[X_Opening[0]][X_Opening[1]];
}*/

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
    /*protoss_openings.push_back("fast_legs"); // core-citadel-gates-legs
    protoss_openings.push_back("fast_DT"); // citadel-archives-DT
    protoss_openings.push_back("fast_air"); // gate-core-stargates
    protoss_openings.push_back("fast_expand"); // nexus first
    protoss_openings.push_back("reaver"); // robot-support bay
    protoss_openings.push_back("standard"); // 2 gates-core-robo-observatory
    protoss_openings.push_back("goons"); // gate-core-gates-range (NonY)
    protoss_openings.push_back("proxy_gates"); // pylon-gates @enemy
    protoss_openings.push_back("photon_rush"); // forge-pylon @enemy
    protoss_openings.push_back("unknown");*/
    protoss_openings.push_back("FastLegs");
    protoss_openings.push_back("FastDT");
    protoss_openings.push_back("FastObs");
    protoss_openings.push_back("ReaverDrop");
    protoss_openings.push_back("Carrier");
    protoss_openings.push_back("FastExpand");
    protoss_openings.push_back("Unknown");


    std::vector<std::string> zerg_openings;
    zerg_openings.push_back("fast_pool"); // 4-8 pools
    zerg_openings.push_back("lings"); // early pool + no peons @ extractor
    zerg_openings.push_back("fast_mutas"); // early extractor-lair-spire
    zerg_openings.push_back("mass_hydras"); // expand-hydra den
    zerg_openings.push_back("mutas_into_hydras"); // expand-lair-spire-hatch-hydra
    zerg_openings.push_back("fast_lurkers"); // early gaz-lair-hydra den
    zerg_openings.push_back("unknown");

    ///std::vector<std::set<Terran_Buildings> > terran = get_terran_X_values();
    std::vector<std::set<Protoss_Buildings> > protoss = get_protoss_X_values();
    ///std::vector<std::set<Zerg_Buildings> > zerg = get_zerg_X_values();

    /**********************************************************************
      VARIABLES SPECIFICATION
     **********************************************************************/
    plSymbol X("X", plIntegerType(0, protoss.size()));
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

    plSymbol Time("Time", plIntegerType(1,900));

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
    for (unsigned int i = 0; i < protoss.size(); i++)
        tableX.push_back(1.0); // TOLEARN
    plProbTable P_X(X, tableX, false);

    // Specification of P(O_1..NB_PROTOSS_BUILDINGS)
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plProbValue tmp_table[] = {0.5, 0.5}; // TOLEARN (per observation)
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
    {
        P_Observed.push_back(plProbTable(observed[i], tmp_table, true));
        listObs *= plProbTable(observed[i], tmp_table, true);
    }

    // Specification of P(lambda | X, O_1..NB_PROTOSS_BUILDINGS)
    plVariablesConjunction conjObs;
    for (std::vector<plSymbol>::const_iterator it = observed.begin();
            it != observed.end(); ++it)
        conjObs ^= (*it);
    plExternalFunction coherence(lambda, X^conjObs, test_X_possible);
    plFunctionalDirac P_lambda(lambda, X^conjObs, coherence);
    // P_lambda = 1 if lambda=coherence(X, observed), 0 otherwise
    
    // Specification of P(T | X, OpeningProtoss)
    // LEARNT FROM THE REPLAYS
    ///plExternalFunction mean(Time, X^OpeningProtoss, mean_Time_Protoss);
    ///plExternalFunction stddev(Time, X^OpeningProtoss, stddev_Time_Protoss);
    ///plCndBellShape P_Time(Time, X^OpeningProtoss, mean, stddev);
    plCndLearnObject<plLearnBellShape> time_learner(Time, X^OpeningProtoss);
    plValues vals(time_learner.get_variables());
    string input;
    while (cin)
    {
        getline(cin, input);
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening != "")
        {
            multimap<unsigned int, Building> tmpBuildings;
            getBuildings(input, tmpBuildings);
            vals[OpeningProtoss] = tmpOpening;
            tmpBuildings.erase(0); // key == 0
            for (map<unsigned int, Building>::const_iterator it 
                    = tmpBuildings.begin(); 
                    it != tmpBuildings.end(); ++it)
            {

            }
            //vals[X] = getXVal(input);
            //if (!time_learner.add_point())
            //    cout << "point not added" << endl;
        }
    }


    /**********************************************************************
      DECOMPOSITION
     **********************************************************************/
    plJointDistribution jd(X^conjObs^lambda^OpeningProtoss^Time, 
            P_X*listObs*P_lambda*P_OpeningProtoss
            *time_learner.get_computable_object()); //P_Time);
    jd.draw_graph("jd.fig");
    cout<<"OK\n";
    /**********************************************************************
      PROGRAM QUESTION
     **********************************************************************/
    /*
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
    /* */

    return 0;
}
