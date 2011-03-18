#include <pl.h>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include "enums_name_tables.h"
#include "x_values.h"

#define PRINT

using namespace std;

//P(lambda|X Protoss_Nexus Protoss_Expansion Protoss_Robotics_Facility Protoss_Pylon Protoss_Pylon2 Protoss_Assimilator Protoss_Observatory Protoss_Gateway Protoss_Gateway2 Protoss_Photon_Cannon Protoss_Citadel_of_Adun Protoss_Cybernetics_Core Protoss_Templar_Archives Protoss_Forge Protoss_Stargate Protoss_Fleet_Beacon Protoss_Arbiter_Tribunal Protoss_Robotics_Support_Bay Protoss_Shield_Battery)


std::vector<std::set<int> > protoss_X;

void test_X_possible(plValues& lambda, const plValues& X_Obs_conj)
{
    set<int> setX = protoss_X[X_Obs_conj[0]];
#ifdef PRINT
    ///////////// print 
    cout << ">>> setX: ";
    for (set<int>::const_iterator ibn
            = setX.begin();
            ibn != setX.end(); ++ibn)
    {
        cout << protoss_buildings_name[*ibn] << ", ";
    }
    cout << endl;
    /////////////
#endif

    set<int> setObs;
    set<int> intersect;
    //for (plValues::const_iterator it = X_Obs_conj.begin(); ...)
    for (unsigned int i = 1; i <= NB_PROTOSS_BUILDINGS; ++i)
    {
        if (X_Obs_conj[i])
        {
            setObs.insert(i-1);
            if (setX.count(i-1))
                intersect.insert(i-1);
        }
    }
#ifdef PRINT
    ///////////// print 
    cout << ">>> setObs: ";
    for (set<int>::const_iterator ibn
            = setObs.begin();
            ibn != setObs.end(); ++ibn)
    {
        cout << protoss_buildings_name[*ibn] << ", ";
    }
    cout << endl;
    /////////////
#endif

    vector<int> difference(setObs.size());
    vector<int>::iterator it = 
        set_difference(setObs.begin(), setObs.end(), 
                intersect.begin(), intersect.end(), 
                difference.begin());
    if (difference.begin() == it)
    {
        lambda[0] = 1; // true
    }
    else
    {
        lambda[0] = 0; // false
    }
}

int main(int argc, const char *argv[])
{
    ifstream fin("PvP.txt");
    protoss_X = get_X_values(fin);

    plSymbol X("X", plIntegerType(0, protoss_X.size()));
    std::vector<plSymbol> observed;
    plSymbol lambda("lambda", PL_BINARY_TYPE);
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
    {
        // what has been observed
        observed.push_back(plSymbol(protoss_buildings_name[i], PL_BINARY_TYPE));
    }
    
    // P(X) (possible tech trees)
    std::vector<plProbValue> tableX;
    for (unsigned int i = 0; i < protoss_X.size(); i++)
        tableX.push_back(1.0); // TOLEARN
    plProbTable P_X(X, tableX, false);

    // P(O_1..NB_PROTOSS_BUILDINGS)
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plProbValue tmp_table[] = {0.5, 0.5}; // TOLEARN (per observation)
    for (unsigned int i = 0; i < NB_PROTOSS_BUILDINGS; i++)
    {
        P_Observed.push_back(plProbTable(observed[i], tmp_table, true));
        listObs *= plProbTable(observed[i], tmp_table, true);
    }

    // P_lambda
    plVariablesConjunction ObsConj;
    for (std::vector<plSymbol>::const_iterator it = observed.begin();
            it != observed.end(); ++it)
        ObsConj ^= (*it);
    plVariablesConjunction X_Obs_conj = X^ObsConj;
    plExternalFunction coherence(lambda, X_Obs_conj, test_X_possible);
    plFunctionalDirac P_lambda(lambda, X_Obs_conj , coherence);
    
    plJointDistribution jd(lambda^X_Obs_conj, P_lambda*P_X*listObs);
    plCndDistribution Cnd_P_lambda_knowing_ObsConj;
    jd.ask(Cnd_P_lambda_knowing_ObsConj, lambda, ObsConj);
    cout << jd.ask(lambda, ObsConj) << endl;

//Protoss_Nexus Protoss_Expansion Protoss_Robotics_Facility Protoss_Pylon Protoss_Pylon2 Protoss_Assimilator Protoss_Observatory Protoss_Gateway Protoss_Gateway2 Protoss_Photon_Cannon Protoss_Citadel_of_Adun Protoss_Cybernetics_Core Protoss_Templar_Archives Protoss_Forge Protoss_Stargate Protoss_Fleet_Beacon Protoss_Arbiter_Tribunal Protoss_Robotics_Support_Bay Protoss_Shield_Battery = 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    plValues evidence(ObsConj);
    unsigned int i = 0;
    evidence[i++] = 1;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 1;
    evidence[i++] = 0;
    evidence[i++] = 1;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    evidence[i++] = 0;
    plDistribution D_lambda;
    Cnd_P_lambda_knowing_ObsConj.instantiate(D_lambda, evidence);

    cout << ">>> Distribution of Lambda: " << endl;

    cout << D_lambda.compile() << endl;

    return 0;
}
