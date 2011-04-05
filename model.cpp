#include <pl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include "enums_name_tables.h"
#include "x_values.h"
#include "replays.h"
#include "parameters.h"

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

using namespace std;

typedef void iovoid;

/// POSSIBLE: use an iterator in values[] so that we directly
/// POSSIBLE: put a std::set instead of std::vector for vector_X
/// TODO: replace set by unordered_set (TR1 or boost) in a lot of places

std::vector<std::set<int> > vector_X;

#ifdef BENCH
/**
 * Returns plValues corresponding to the max of a map<plValues, plProbValues>
 */
plValues max(const map<plValues, plProbValue>& m)
{
    if (m.empty())
    {
        cout << "ERROR: given an empty map<plValues, plProValues> to max()" 
            << endl;
        return plValues();
    }
    plProbValue max = -1.0;
    plValues maxV;
    for (map<plValues, plProbValue>::const_iterator it = m.begin();
            it != m.end(); ++it)
    {
        if (it->second > max)
            maxV = it->first;
    }
    return maxV;
}
#endif

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
 * => X is compatible with observations if it covers them fully
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
 * Does all the learning for its two plCndLearnObject parameters
 * from the replays specified in inputstream
 * Relies heavily on:
 *  - pruneOpeningVal(string&): removes and returns the Opening
 *  - getBuildings(string, mmap<int, Building>, int): fill the mmap with  
 *    time at which each building has been built.
 */
template<class T>
void learn_T_and_X(ifstream& inputstream,
            plCndLearnObject<plLearnBellShape>& timeLearner,
            plCndLearnObject<plLearnHistogram>& xLearner,
            plSymbol& Opening, plSymbol& X, plSymbol& Time)
{
    string input;
    plValues vals_timeLearner(timeLearner.get_variables());
    plValues vals_xLearner(xLearner.get_variables());
#ifdef ERROR_CHECKS
    map<int, int> count_X_examples;
    map<int, plValues> one_value_per_X;
    map<pair<int, string>, int> count_X_Op_examples;
    int nbpts = 0; // count the number of added points (time, building)
#endif
    
    while (getline(inputstream, input))
    {
        if (input.empty())
            break;
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening != "")
        {
            multimap<int, Building> tmpBuildings;
            getBuildings(input, tmpBuildings, LEARN_TIME_LIMIT);
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
                vals_timeLearner[Opening] = tmpOpening;
                vals_xLearner[Opening] = tmpOpening;
#if DEBUG_OUTPUT > 1
                std::cout << "Opening: " << tmpOpening << std::endl;
#endif
                int tmp_ind = get_X_indice(tmpSet, vector_X);
                vals_timeLearner[X] = tmp_ind;
                vals_xLearner[X] = tmp_ind;
#if DEBUG_OUTPUT > 1
                std::cout << "X ind: " << tmp_ind
                    << std::endl;
#endif
                vals_timeLearner[Time] = it->first;
#if DEBUG_OUTPUT > 1
                std::cout << "Time: " << it->first << std::endl;
#endif
#ifdef ERROR_CHECKS
                if (count_X_examples.count(tmp_ind))
                {
                    count_X_examples[tmp_ind] = count_X_examples[tmp_ind] + 1;
                }
                else
                {
                    count_X_examples.insert(make_pair<int, int>(
                                tmp_ind, 1));
                    one_value_per_X.insert(make_pair<int, plValues>(
                                tmp_ind, vals_timeLearner));
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
#endif

                /// Add data point
                if (!timeLearner.add_point(vals_timeLearner))
                    cout << "ERROR: point not added to P(T|X,Op)" << endl;
                if (!xLearner.add_point(vals_xLearner))
                    cout << "ERROR: point not added to P(X|Op)" << endl;
                vals_timeLearner.reset();
                vals_xLearner.reset();
#ifdef ERROR_CHECKS
                ++nbpts;
#endif
            }
        }
    }
    
#ifdef ERROR_CHECKS
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
            /////////////////////////////////////////////////
            // put another (slightly different) value TODO CHANGE
            one_value_per_X[it->first][Time] = 
                one_value_per_X[it->first][Time] + 2; /// totally arbitrary 2
            timeLearner.add_point(one_value_per_X[it->first]);
            /////////////////////////////////////////////////
        }
    }
#endif
#if DEBUG_OUTPUT > 2
    cout << "*** Number of points (total), I counted: " << nbpts << endl;
    cout << "*** Number of different pairs (X, Opening): " 
        << count_X_Op_examples.size() << endl;
#endif
    /*
    for (unsigned int i = 0; i < vector_X.size(); i++)
    {
        plValues rightValues(X^Opening); 
        rightValues[Opening] = "FastExpand";
        rightValues[X] = i;
        cout << "Right values: " << rightValues << endl;
        cout << "Learnt parameters, mu: " 
        << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(
        rightValues)->get_distribution()).mean() 
        << ", stddev: " 
        << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(
        rightValues)->get_distribution()).standard_deviation() 
        << endl;
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

    if (argv[1] != NULL &&
            (argv[1][1] == 'P' || argv[1][1] == 'T' || argv[1][1] == 'Z'))
    {
        string argv1 = string(argv[1]);
        /// match up: Enemy vs Us
        /// For instance ZvT will get Zerg buildings
        stringstream extract_X_from;
        if (argv[1][3] == 'P' || argv[1][3] == 'T' || argv[1][3] == 'Z')
        {
            cout << "We are " << argv[1][3] 
                << " against " << argv[1][1] << endl;
            extract_X_from << argv[1][1] << "v" << argv[1][3] << ".txt";
        }
        else
        {
            extract_X_from << "l" << argv[1][1] << "all.txt";
        }
        if (argv[1][1] == 'P')
        {
            ifstream fin(extract_X_from.str().c_str()); // could be argv[1]
            vector_X = get_X_values(fin); /// Enemy race
            openings = protoss_openings;
            nbBuildings = NB_PROTOSS_BUILDINGS;
            buildings_name = protoss_buildings_name;
            cout << "X size: " << vector_X.size() << endl;
        }
        else if (argv[1][1] == 'T')
        {
            ifstream fin(extract_X_from.str().c_str()); // could be argv[1]
            vector_X = get_X_values(fin); /// Enemy race
            openings = terran_openings;
            nbBuildings = NB_TERRAN_BUILDINGS;
            buildings_name = terran_buildings_name;
        }
        else if (argv[1][1] == 'Z')
        {
            ifstream fin(extract_X_from.str().c_str()); // could be argv[1]
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
        tableOpening.push_back(1.0);
    plProbTable P_Opening(Opening, tableOpening, false);

    // Specification of P(X | Opening) (possible tech trees)
    plCndLearnObject<plLearnHistogram> xLearner(X, Opening);

    // Specification of P(O_1..NB_<RACE>_BUILDINGS)
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plProbValue tmp_table[] = {0.5, 0.5};
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
        learn_T_and_X<Protoss_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (argv[1][1] == 'T')
        learn_T_and_X<Terran_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (argv[1][1] == 'Z')
        learn_T_and_X<Zerg_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
#if DEBUG_OUTPUT > 2
    cout << "*** Number of possible pairs (X, Opening): "
        << vector_X.size()*openings.size();
    cout << timeLearner.get_computable_object() << endl;
#endif

    /**********************************************************************
      DECOMPOSITION
     **********************************************************************/
    plVariablesConjunction knownConj = ObsConj^lambda^Time;
    plJointDistribution jd(X^Opening^knownConj,
            //P_X*P_Opening*listObs*P_lambda
            xLearner.get_computable_object()*P_Opening*listObs*P_lambda
            *timeLearner.get_computable_object()); // <=> P_Time);
    jd.draw_graph("jd.fig");
#if DEBUG_OUTPUT > 0
    cout << "Joint distribution built." << endl;
#endif

    /**********************************************************************
      PROGRAM QUESTION
     **********************************************************************/
#if PLOT > 0
    plVariablesConjunction X_Op = X^Opening;
    plCndDistribution Cnd_P_Time_X_knowing_Op;
    jd.ask(Cnd_P_Time_X_knowing_Op, Time^X, Opening);
#if DEBUG_OUTPUT > 2
    cout << Cnd_P_Time_X_knowing_Op << endl;
#endif

#if PLOT > 1
    plCndDistribution Cnd_P_Time_knowing_X_Op;
    jd.ask(Cnd_P_Time_knowing_X_Op, Time, X_Op);
#if DEBUG_OUTPUT > 2
    cout << Cnd_P_Time_knowing_X_Op << endl;
#endif
#endif
    for (unsigned int i = 0; i < openings.size(); i++) // Openings
    {
        plValues evidence(Opening);
        evidence[Opening] = i;
        plDistribution PP_Time_X;
        Cnd_P_Time_X_knowing_Op.instantiate(PP_Time_X, evidence);
        plDistribution T_P_Time_X;
        PP_Time_X.compile(T_P_Time_X);
        std::stringstream tmp;
        tmp << "Opening" << openings[i] << ".gnuplot";
        T_P_Time_X.plot(tmp.str().c_str());

#if PLOT > 1
        for (unsigned int j = 0; j < vector_X.size(); ++j)
        {
            plValues evidence2(Opening^X);
            evidence2[Opening] = i;
            evidence2[X] = j; 
            // next line generates plWarning when == 0
            if (timeLearner.get_learnt_object_for_value(evidence2) != 0)
            {
                Cnd_P_Time_knowing_X_Op.instantiate(PP_Time_X, evidence2);
                PP_Time_X.compile(T_P_Time_X);
                std::stringstream tmp2;
                tmp2 << "Opening" << openings[i] << "X" << j << ".gnuplot";
                T_P_Time_X.plot(tmp2.str().c_str());
            }
        }
#endif
    }
    return 0;
#endif

    plCndDistribution Cnd_P_Opening_knowing_rest;
    jd.ask(Cnd_P_Opening_knowing_rest, Opening, knownConj);
#if DEBUG_OUTPUT > 0
    cout << Cnd_P_Opening_knowing_rest << endl;
#endif

    if (argc < 2)
        return 1;
    ifstream inputfile_test(argv[2]);
    string input;
    cout << endl;
    cout << ">>>> Testing from: " << argv[2] << endl;
    unsigned int noreplay = 0;
#ifdef BENCH
    unsigned int positive_classif_finale = 0;
    unsigned int cpositive_classif_finale = 0;
    map<plValues, plProbValue> cumulative_prob;
    for (vector<string>::const_iterator it = openings.begin();
            it != openings.end(); ++it)
    {
        plValues tmp(Opening);
        tmp[Opening] = *it;
        cumulative_prob.insert(make_pair<plValues, plProbValue>(tmp, 0.0));
    }
#endif

    while (getline(inputfile_test, input))
    {
        plValues evidence(knownConj);
        evidence[lambda] = 1;
        if (input.empty())
            break;
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening == "")
            continue;
        multimap<int, Building> tmpBuildings;
        getBuildings(input, tmpBuildings);
        tmpBuildings.erase(0); // key == 0 i.e. buildings not constructed
        if (tmpBuildings.empty())
            continue;
#if DEBUG_OUTPUT >= 1
        cout << "******** end replay number: " 
            << noreplay << " ********" << endl;
        ++noreplay;
        cout << endl << endl << endl;
        cout << "******** Real Opening: " << tmpOpening 
            << " replay number: " << noreplay
            << " ********" << endl;
#endif
        evidence[observed[0]] = 1; // the first Nexus/CC/Hatch exists
        // we assume we didn't see any buildings
        for (unsigned int i = 1; i < nbBuildings; ++i)
            evidence[observed[i]] = 0;

        plDistribution T_P_Opening;
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
            PP_Opening.compile(T_P_Opening);
#if DEBUG_OUTPUT >= 1
            cout << "====== P(Opening | evidence), building: "
                << it->second << " ======" << endl;
            cout << T_P_Opening << endl;
#endif
#ifdef BENCH
            if (T_P_Opening.is_null())
                continue;
            vector<pair<plValues, plProbValue> > outvals;
            T_P_Opening.sorted_tabulate(outvals);
            for (vector<pair<plValues, plProbValue> >::const_iterator 
                    jt = outvals.begin(); jt != outvals.end(); ++jt)
            {
                cumulative_prob[jt->first] += jt->second;
            }
#endif
        }
#ifdef BENCH
        for (map<plValues, plProbValue>::const_iterator 
                jt = cumulative_prob.begin(); jt != cumulative_prob.end(); ++jt)
        {
            cout << "|||| ";
            jt->first.Output(cout);
            cout << " => sum on probs: " << jt->second << endl;
        }
        plValues toTest(Opening);
        toTest[Opening] = tmpOpening;
        if (T_P_Opening.is_null())
            continue;
        if (toTest[Opening] == T_P_Opening.best()[Opening])
            ++positive_classif_finale;
        if (toTest[Opening] == max(cumulative_prob)[Opening])
            ++cpositive_classif_finale;
#endif
    }
#ifdef BENCH
    cout << ">>> Positive classif: " << positive_classif_finale
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(positive_classif_finale)/noreplay << endl;
    cout << ">>> Cumulative positive classif: " << cpositive_classif_finale
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(cpositive_classif_finale)/noreplay << endl;
#endif

    /*plSerializer my_serializer();
    my_serializer.add_object("P_CB", P_CB);
    my_serializer.save("test.xml");*/

    // On Windows (Visual C++, MinGW) only.
#if defined(WIN32) || defined(_WIN32)
    cout << "Press any key to terminate..." << endl;
    getchar();
#endif
    return 0;
}
