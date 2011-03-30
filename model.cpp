#include <pl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include "enums_name_tables.h"
#include "x_values.h"
#include "replays.h"

using namespace std;

typedef void iovoid;

#define DEBUG_OUTPUT 0

/// TODO: use an iterator in values[] so that we directly
/// put a std::set instead of std::vector for vector_X
/// TODO: replace set by unordered_set (TR1 or boost) in a lot of places

std::vector<std::set<int> > vector_X;

/**
 * Prints the command line format/usage of the program
 */
iovoid usage()
{
    cout << "usage: ./model learn_from test_from" << endl;
    cout << "with names such as 'xRvS': " << endl;
    cout << " - x = l (learn) or t (test)" << endl;
    cout << " - R = T or P or Z, race against/of the buildings" << endl;
    cout << " v stands for versus" << endl;
    cout << " - S = T or P or Z, other race of the matchup" << endl;
}

/** 
 * Tests if the given X value (X plSymbol in plValues X_Obs_conj)
 * is compatible with what obervations have been seen
 * (observed plSymbol(s) in plValues X_Obs_conj)
 * {X ^ observed} covers all observed if X is possible
 * so X is impossible if {observed \ {X ^ observed}} != {}
 */
void test_X_possible(plValues& lambda, const plValues& X_Obs_conj)
{
    set<int> setX = vector_X[X_Obs_conj[0]];
    set<int> setObs;
    set<int> intersect;
    //for (plValues::const_iterator it = X_Obs_conj.begin(); ...)
    for (unsigned int i = 1; i <= X_Obs_conj.size() - 1; ++i) // -1 for X
    {
        if (X_Obs_conj[i])
        {
            setObs.insert(i-1);
            if (setX.count(i-1))
                intersect.insert(i-1);
        }
    }

    vector<int> difference(setObs.size());
    vector<int>::iterator it = 
        set_difference(setObs.begin(), setObs.end(), 
                intersect.begin(), intersect.end(), 
                difference.begin());
    if (difference.begin() == it)
    {
#if DEBUG_OUTPUT > 3
        cout << "DEBUG: test_X_possible TRUE" << endl;
#endif
        lambda[0] = 1; // true
    }
    else
    {
#if DEBUG_OUTPUT > 3
        cout << "DEBUG: test_X_possible FALSE" << endl;
#endif
        lambda[0] = 0; // false
    }
}

/**
 * Does all the learning in a timeLearner given the data in the inputstream
 */
template<class T>
void learn_T_knowing_X_Opening(ifstream& inputstream,
            plCndLearnObject<plLearnBellShape>& timeLearner,
            plSymbol& Opening, plSymbol& X, plSymbol& Time)
{
    string input;
    plValues vals(timeLearner.get_variables());
    map<int, int> count_X_examples;
    map<int, plValues> one_value_per_X;
    map<pair<int, string>, int> count_X_Op_examples;
    int nbpts = 0;
    
    while (getline(inputstream, input))
    {
        if (input.empty())
            break;
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening != "")
        {
            multimap<int, Building> tmpBuildings;
            getBuildings(input, tmpBuildings);
            tmpBuildings.erase(0); // key == 0 i.e. buildings not constructed
            std::set<int> tmpSet;
            tmpSet.insert(0);
            for (map<int, Building>::const_iterator it 
                    = tmpBuildings.begin(); 
                    it != tmpBuildings.end(); ++it)
            {
                if (it->first > LEARN_TIME_LIMIT)
                    break;
                tmpSet.insert(it->second.getEnumValue());
                vals[Opening] = tmpOpening;
#if DEBUG_OUTPUT > 1
                std::cout << "Opening: " << tmpOpening << std::endl;
#endif
                int tmp_ind = get_X_indice(tmpSet, vector_X);
                vals[X] = tmp_ind;
#if DEBUG_OUTPUT > 1
                std::cout << "X ind: " << tmp_ind
                    << std::endl;
#endif
                vals[Time] = it->first;
#if DEBUG_OUTPUT > 1
                std::cout << "Time: " << it->first << std::endl;
#endif
                if (count_X_examples.count(tmp_ind))
                {
                    count_X_examples[tmp_ind] = count_X_examples[tmp_ind] + 1;
                }
                else
                {
                    count_X_examples.insert(make_pair<int, int>(
                                tmp_ind, 1));
                    one_value_per_X.insert(make_pair<int, plValues>(
                                tmp_ind, vals));
                }

                pair<int, string> tmpPair = make_pair<int, string>(tmp_ind, 
                        tmpOpening);
                if (count_X_Op_examples.count(tmpPair))
                {
                    count_X_Op_examples[tmpPair] = 
                        count_X_Op_examples[tmpPair] + 1;
                }
                else
                {
                    count_X_Op_examples.insert(make_pair<pair<int, string>, 
                            int>(tmpPair, 1));
                }

                /// Add data point
                if (!timeLearner.add_point(vals))
                    cout << "ERROR: point not added" << endl;
                ++nbpts;
                vals.reset();
            }
        }
    }
    
    /// Check for possible errors
    for (map<int, int>::const_iterator it = count_X_examples.begin();
            it != count_X_examples.end(); ++it)
    {
        if (it->second == 0)
        {
            cout << "PROBLEM: We never encountered X: ";
            for (typename set<int>::const_iterator ibn
                    = vector_X[it->first].begin();
                    ibn != vector_X[it->first].end(); ++ibn)
            {
                Building tmpBuilding(static_cast<T>(*ibn));
                cout << tmpBuilding << ", ";
            }
            cout << endl;
        }
        else if (it->second == 1)
        {
            cout << "(POSSIBLE)PROBLEM: We encountered X only one time: ";
            for (typename set<int>::const_iterator ibn
                    = vector_X[it->first].begin();
                    ibn != vector_X[it->first].end(); ++ibn)
            {
                Building tmpBuilding(static_cast<T>(*ibn));
                cout << tmpBuilding << ", ";
            }
            cout << endl;
            // put another (slightly different) value
            one_value_per_X[it->first][Time] = 
                one_value_per_X[it->first][Time] + 2; /// totally arbitrary 2
            timeLearner.add_point(one_value_per_X[it->first]);
        }
    }
    //////////
    cout << "Number of points (total): " << nbpts << endl;
    cout << "Number of different pairs (X, Opening): " 
        << count_X_Op_examples.size() << endl;
    //////////
    /*
    for (unsigned int i = 0; i < vector_X.size(); i++)
    {
        plValues rightValues(X^Opening); 
        rightValues[Opening] = "FastExpand";
        rightValues[X] = i;
        cout << "Right values: " << rightValues << endl;
        cout << "Learnt parameters, mu: " << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(rightValues)->get_distribution()).mean() << ", stddev: " << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(rightValues)->get_distribution()).standard_deviation() << endl;
    }/**/
}

int main(int argc, const char *argv[])
{
    /**********************************************************************
      INITIALIZATION
     **********************************************************************/
    std::vector<std::string> terran_openings;
    /*terran_openings.push_back("fast_drop"); // facto-starport-control tower
    terran_openings.push_back("full_metal"); // facto x2 + machine shop
    terran_openings.push_back("MM"); // raxes-academy
    terran_openings.push_back("fast_expand"); // CC first
    terran_openings.push_back("1rax_FE"); // rax-CC
    terran_openings.push_back("mech_timing_push"); // facto-armory-facto
    terran_openings.push_back("fast_air"); // starport x2
    terran_openings.push_back("BBS"); // rax-rax-supply
    terran_openings.push_back("unkown");*/
    terran_openings.push_back("Bio");
    terran_openings.push_back("TwoFactory");
    terran_openings.push_back("VultureHarass");
    terran_openings.push_back("SiegeExpand");
    terran_openings.push_back("Standard");
    terran_openings.push_back("FastDropship");
    terran_openings.push_back("Unknown");

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
    /*zerg_openings.push_back("fast_pool"); // 4-8 pools
    zerg_openings.push_back("lings"); // early pool + no peons @ extractor
    zerg_openings.push_back("fast_mutas"); // early extractor-lair-spire
    zerg_openings.push_back("mass_hydras"); // expand-hydra den
    zerg_openings.push_back("mutas_into_hydras"); // expand-lair-spire-hatch-hydra
    zerg_openings.push_back("fast_lurkers"); // early gaz-lair-hydra den
    zerg_openings.push_back("unknown");*/
    zerg_openings.push_back("TwoHatchMuta");
    zerg_openings.push_back("ThreeHatchMuta");
    zerg_openings.push_back("HydraRush");
    zerg_openings.push_back("Standard");
    zerg_openings.push_back("HydraMass");
    zerg_openings.push_back("Lurker");
    zerg_openings.push_back("Unknown");

    std::vector<std::string> openings;
    int nbBuildings;
    const char** buildings_name;

    if (argv[1] != NULL && argv[1][2] == 'v')
    {
        string argv1 = string(argv[1]);
        /// match up: Enemy vs Us
        /// For instance ZvT will get Zerg buildings
        if (argv[1][1] == 'P')
        {
            ifstream fin("PvP.txt"); // PvP.txt / testP.txt
            vector_X = get_X_values(fin); /// Enemy race
            openings = protoss_openings;
            nbBuildings = NB_PROTOSS_BUILDINGS;
            buildings_name = protoss_buildings_name;
            cout << "X size: " << vector_X.size() << endl;
        }
        else if (argv[1][1] == 'T')
        {
            ifstream fin("testT.txt");
            vector_X = get_X_values(fin); /// Enemy race
            openings = terran_openings;
            nbBuildings = NB_TERRAN_BUILDINGS;
            buildings_name = terran_buildings_name;
        }
        else if (argv[1][1] == 'Z')
        {
            ifstream fin("testZ.txt");
            vector_X = get_X_values(fin); /// Enemy race
            openings = zerg_openings;
            nbBuildings = NB_ZERG_BUILDINGS;
            buildings_name = zerg_buildings_name;
        }
        else
        {
            cout << "ERROR in the first argument" << endl;
            usage();
            return 1;
        }
    }
    else
    {
        cout << "ERROR in the first argument" << endl;
        usage();
        return 1;
    }

    /**********************************************************************
      VARIABLES SPECIFICATION
     **********************************************************************/
    plSymbol X("X", plIntegerType(0, vector_X.size()));
    std::vector<plSymbol> observed;
    plSymbol lambda("lambda", PL_BINARY_TYPE);
    // what has been observed
    for (unsigned int i = 0; i < nbBuildings; i++)
        observed.push_back(plSymbol(buildings_name[i], PL_BINARY_TYPE));
    plSymbol Opening("Opening", plLabelType(openings));

    plSymbol Time("Time", plIntegerType(1, LEARN_TIME_LIMIT)); 

    /**********************************************************************
      PARAMETRIC FORM SPECIFICATION
     **********************************************************************/
    // Specification of P(Opening)
    std::vector<plProbValue> tableOpening;
    for (unsigned int i = 0; i < openings.size(); i++) 
        tableOpening.push_back(1.0); // TOLEARN
    plProbTable P_Opening(Opening, tableOpening, false);

    // Specification of P(X) (possible tech trees)
    std::vector<plProbValue> tableX;
    for (unsigned int i = 0; i < vector_X.size(); i++)
        tableX.push_back(1.0); // TOLEARN
    plProbTable P_X(X, tableX, false);

    // Specification of P(O_1..NB_<RACE>_BUILDINGS)
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plProbValue tmp_table[] = {0.5, 0.5}; // TOLEARN (per observation)
    for (unsigned int i = 0; i < nbBuildings; i++)
    {
        P_Observed.push_back(plProbTable(observed[i], tmp_table, true));
        listObs *= plProbTable(observed[i], tmp_table, true);
    }

    // Specification of P(lambda | X, O_1..NB_<RACE>_BUILDINGS)
    plVariablesConjunction ObsConj;
    for (std::vector<plSymbol>::const_iterator it = observed.begin();
            it != observed.end(); ++it)
        ObsConj ^= (*it);
    plVariablesConjunction X_Obs_conj = X^ObsConj;
    plExternalFunction coherence;
    if (argv[1][1] == 'P')
        coherence = plExternalFunction(lambda, X_Obs_conj, 
                test_X_possible);
    else if (argv[1][1] == 'T')
        coherence = plExternalFunction(lambda, X_Obs_conj, 
                test_X_possible);
    else if (argv[1][1] == 'Z')
        coherence = plExternalFunction(lambda, X_Obs_conj, 
                test_X_possible);
    plFunctionalDirac P_lambda(lambda, X_Obs_conj , coherence);
    
    // Specification of P(T | X, Opening)
    plCndLearnObject<plLearnBellShape> timeLearner(Time, X^Opening);
    cout << ">>>> Learning from: " << argv[1] << endl;
    ifstream inputstream(argv[1]);
    if (argv[1][1] == 'P')
        learn_T_knowing_X_Opening<Protoss_Buildings>(inputstream, timeLearner,
                Opening, X, Time);
    else if (argv[1][1] == 'T')
        learn_T_knowing_X_Opening<Terran_Buildings>(inputstream, timeLearner,
                Opening, X, Time);
    else if (argv[1][1] == 'Z')
        learn_T_knowing_X_Opening<Zerg_Buildings>(inputstream, timeLearner,
                Opening, X, Time);
    //////////
    cout << "Number of possible pairs (X, Opening): "
        << vector_X.size()*openings.size();
    //////////
#if DEBUG_OUTPUT > 2
    cout << timeLearner.get_computable_object() << endl;
#endif

    /**********************************************************************
      DECOMPOSITION
     **********************************************************************/
    plVariablesConjunction knownConj = ObsConj^lambda^Time;
    plJointDistribution jd(X^Opening^knownConj,
            P_X*P_Opening*listObs*P_lambda
            *timeLearner.get_computable_object()); // <=> P_Time);
    jd.draw_graph("jd.fig");
#if DEBUG_OUTPUT > 0
    cout << "Joint distribution built." << endl;
#endif

    /**********************************************************************
      PROGRAM QUESTION
     **********************************************************************/
    ///***** 
    plVariablesConjunction X_Op = X^Opening;
    plCndDistribution Cnd_P_Time_X_knowing_Op;
    jd.ask(Cnd_P_Time_X_knowing_Op, Time^X, Opening);

    plCndDistribution Cnd_P_Time_knowing_X_Op;
    jd.ask(Cnd_P_Time_knowing_X_Op, Time, X_Op);
    for (unsigned int j = 0; j < openings.size(); j++) // Openings
    {
        //cout << jd.ask(Time, X_Op) << endl;
        plValues evidence(Opening);
        evidence[Opening] = j;
        plDistribution PP_Time_X;
        Cnd_P_Time_X_knowing_Op.instantiate(PP_Time_X, evidence);
        ///cout << "======== P(Time, X | Op) ========" << endl;
        ///cout << PP_Time_X.get_left_variables() << endl;
        ///cout << PP_Time_X.get_right_variables() << endl;
        ///cout << Cnd_P_Time_X_knowing_Op << endl;
        plDistribution T_P_Time_X;
        PP_Time_X.compile(T_P_Time_X);
        /// cout << T_P_Time_X << endl;
        std::stringstream tmp;
        tmp << "Opening" << openings[j] << ".gnuplot";
        T_P_Time_X.plot(tmp.str().c_str());

        evidence[X] = 10;
        Cnd_P_Time_knowing_X_Op.instantiate(PP_Time_X, evidence);
        PP_Time_X.compile(T_P_Time_X);
        tmp << "Opening" << openings[j] << "X10" << ".gnuplot";
        T_P_Time_X.plot(tmp.str().c_str());
    }
    return 0;
    /**/

    plCndDistribution Cnd_P_Opening_knowing_rest;
    jd.ask(Cnd_P_Opening_knowing_rest, Opening, knownConj);
#if DEBUG_OUTPUT > 0
    cout << jd.ask(Opening, knownConj) << endl;
#endif

    if (argc < 2)
        return 1;
    ifstream inputfile_test(argv[2]);
    string input;
    cout << endl;
    cout << ">>>> Testing from: " << argv[2] << endl;
    unsigned int noreplay = 0;
    while (getline(inputfile_test, input))
    {
        plValues evidence(knownConj);
        evidence[lambda] = 1;
        if (input.empty())
            break;
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening != "")
        {
            multimap<int, Building> tmpBuildings;
            getBuildings(input, tmpBuildings);
            tmpBuildings.erase(0); // key == 0 i.e. buildings not constructed
#if DEBUG_OUTPUT == 1
            cout << "************** Replay number: "
                << noreplay << " **************" << endl;
            ++noreplay;
#endif
            evidence[observed[0]] = 1; // the first Nexus/CC/Hatch exists
            // we assume we didn't see any buildings
            for (unsigned int i = 1; i < nbBuildings; ++i)
                evidence[observed[i]] = 0;
            
            // we assume we see the buildings as soon as they get constructed
            for (map<int, Building>::const_iterator it 
                    = tmpBuildings.begin(); 
                    it != tmpBuildings.end(); ++it)
            {
                evidence[observed[it->second.getEnumValue()]] = 1;
                evidence[Time] = it->first;
                
#if DEBUG_OUTPUT > 1
                cout << "====== evidence ======" << endl;
                cout << evidence << endl;
#endif

                plDistribution PP_Opening;
                Cnd_P_Opening_knowing_rest.instantiate(PP_Opening, evidence);
#if DEBUG_OUTPUT > 1
                cout << "====== P(Opening | rest).instantiate ======" << endl;
                cout << Cnd_P_Opening_knowing_rest << endl;
                cout << PP_Opening.get_left_variables() << endl;
                cout << PP_Opening.get_right_variables() << endl;
#endif
                plDistribution T_P_Opening;
                PP_Opening.compile(T_P_Opening);
#if DEBUG_OUTPUT >= 1
                cout << "====== P(Opening | evidence) ======" << endl;
                cout << T_P_Opening << endl;
#endif
            }
        }
    }


    /*plSerializer my_serializer();
    my_serializer.add_object("P_CB", P_CB);
    my_serializer.save("test.xml");*/

    // On Windows (Visual C++, MinGW) only.
#if defined(WIN32) || defined(_WIN32)
    cout << "Press any key to terminate..." << endl;
    getchar();
#endif
    /* */

    return 0;
}
