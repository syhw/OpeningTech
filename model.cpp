#include "model.h"

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

using namespace std;

/// POSSIBLE: use an iterator in values[] so that we directly
/// POSSIBLE: put a std::set instead of std::vector for vector_X
/// TODO: replace set by unordered_set (TR1 or boost) in a lot of places

std::vector<std::set<int> > vector_X;
int nbBuildings;
const char** buildings_name;

#ifdef BENCH
/**
 * Returns plValues corresponding to the max of a map<plValues, plProbValues>
 */
plValues max(const map<plValues, plProbValue>& m)
{
    if (m.empty())
    {
        cerr << "ERROR: given an empty map<plValues, plProValues> to max()" 
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

#if PLOT > 0
void gnuplot_vector_probas(const vector<vector<plProbValue> >& tab, 
        const vector<string>& openings, const string& filename)
{
    ofstream file;
    // gnuplot file
    file.open(filename.c_str(), ios::out);
    file << "set xlabel \"BuildingNumber\"" << endl;
    file << "set ylabel \"P(Opening)\"" << endl;
    file << "set title \"" << filename << "\"" << endl;
    file << "set style data linespoints" << endl;
    string cpfn = filename;
    cpfn.append(".data");
    file << "plot ";
        file << "\"" << cpfn << "\" using 1:" << 2 << " title " 
            << "\"" << openings[0];
    for (unsigned int i = 1; i < openings.size(); ++i)
    {
        file << "\", \"" << cpfn << "\" using 1:" << i+2 << " title " 
            << "\"" << openings[i];
    }
    file << endl;
    file << "pause -1 \"hit a button to continue\"" << endl;
    file.close();

    // data file
    file.open(cpfn.c_str(), ios::out);
    unsigned int i = 0;
    for (vector<vector<plProbValue> >::const_iterator it = tab.begin();
            it != tab.end(); ++it)
    {
        file << i++ << " ";
        for (vector<plProbValue>::const_iterator jt = it->begin();
                jt != it->end(); ++jt)
        {
            file << *jt << " ";
        }
        file << endl;
    }
    file.close();
}
#endif

/**
 * Prints the command line format/usage of the program
 */
iovoid usage()
{
    cout << "usage: ./model learn_from test_from" << endl;
    cout << "with names such as '.{1}R.{1}S*' / 'xRvS': " << endl;
    cout << " - x = l (learn) or t (test) [recommended]" << endl;
    cout << " - R = T or P or Z, race against/of the buildings [compulsory]"
        << endl;
    cout << " v stands for versus [compulsory placeholder]" << endl;
    cout << " - S = T or P or Z, other race of the matchup [compulsory]" 
        << endl;
    cout << "what is compulsory is to have R and S in positions 1 and 3!" 
        << endl;
}

/** 
 * Opening = LastOpening value
 */
void test_same_opening(plValues& op, const plValues& last_op)
{
    op[0] = last_op[0];
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
#ifdef BENCH
    clock_t start = clock();
#endif
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
                    /// TODO <1> init with bell shape this point as mean
                    ///timeLearner = plLearnBellShape(vals_timeLearner,
                    ///        it->first, 3.0, PL_ONE);
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
                    cerr << "ERROR: point not added to P(T|X,Op)" << endl;
                if (!xLearner.add_point(vals_xLearner))
                    cerr << "ERROR: point not added to P(X|Op)" << endl;
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
            // put another (slightly different) value: change, c.f. TODO <1>
            one_value_per_X[it->first][Time] = 
                one_value_per_X[it->first][Time] + 10; /// totally arbitrary 10
            timeLearner.add_point(one_value_per_X[it->first], 0.5);
            one_value_per_X[it->first][Time] = 
                one_value_per_X[it->first][Time] - 20; /// totally arbitrary -20
            timeLearner.add_point(one_value_per_X[it->first], 0.5);
            /////////////////////////////////////////////////
        }
    }
#endif
#if DEBUG_OUTPUT > 2
    cout << "*** Number of points (total), I counted: " << nbpts << endl;
    cout << "*** Number of different pairs (X, Opening): " 
        << count_X_Op_examples.size() << endl;
#endif
#ifdef BENCH
    clock_t end = clock();
    cout << "TIME: learning took: "
       << (double)(end - start) / CLOCKS_PER_SEC << " sec" << endl;
#endif
}

OpeningPredictor::OpeningPredictor(const vector<string>& op,
        const char* learningFileName)
: openings(op)
{
    /**********************************************************************
      VARIABLES SPECIFICATION
     **********************************************************************/
    X = plSymbol("X", plIntegerType(0, vector_X.size()-1));
    lambda = plSymbol("lambda", PL_BINARY_TYPE);
    // what has been observed
    for (unsigned int i = 0; i < nbBuildings; i++)
        observed.push_back(plSymbol(buildings_name[i], PL_BINARY_TYPE));
    Opening = plSymbol("Opening", plLabelType(openings));
#ifdef DIRAC_ON_LAST_OPENING
    LastOpening = plSymbol("LastOpening", plLabelType(openings));
#endif
    Time = plSymbol("Time", plIntegerType(1, LEARN_TIME_LIMIT)); 

    /**********************************************************************
      PARAMETRIC FORMS SPECIFICATION
     **********************************************************************/
    // Specification of P(Opening)
    for (unsigned int i = 0; i < openings.size(); i++) 
        tableOpening.push_back(1.0);
#ifdef DIRAC_ON_LAST_OPENING
    P_LastOpening = plMutableDistribution(
            plProbTable(LastOpening, tableOpening, false));
    same_opening = plExternalFunction(Opening, LastOpening,
            test_same_opening);
    P_Opening = plFunctionalDirac(Opening, LastOpening, same_opening);
#else
    P_Opening = plProbTable(Opening, tableOpening, false);
#endif

    // Specification of P(X | Opening) (possible tech trees)
    xLearner = plCndLearnObject<plLearnHistogram>(X, Opening);
    std::vector<plProbValue> tableX;
    for (unsigned int i = 0; i < vector_X.size(); i++) 
        tableX.push_back(1.0);
    P_X = plProbTable(X, tableX, false);

    // Specification of P(O_1..NB_<RACE>_BUILDINGS)
    plProbValue tmp_table[] = {0.5, 0.5};
    for (unsigned int i = 0; i < nbBuildings; i++)
    {
        P_Observed.push_back(plProbTable(observed[i], tmp_table, true));
        listObs *= plProbTable(observed[i], tmp_table, true);
    }

    // Specification of P(lambda | X, O_1..NB_<RACE>_BUILDINGS)
    for (std::vector<plSymbol>::const_iterator it = observed.begin();
            it != observed.end(); ++it)
        ObsConj ^= (*it);
    X_Obs_conj = X^ObsConj;
    coherence = plExternalFunction(lambda, X_Obs_conj, 
            test_X_possible);
    P_lambda = plFunctionalDirac(lambda, X_Obs_conj ,coherence);

    // Specification of P(T | X, Opening)
    timeLearner = plCndLearnObject<plLearnBellShape>(Time, X^Opening);
    cout << ">>>> Learning from: " << learningFileName << endl;
    ifstream inputstream(learningFileName);
    if (learningFileName[1] == 'P')
        learn_T_and_X<Protoss_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (learningFileName[1] == 'T')
        learn_T_and_X<Terran_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (learningFileName[1] == 'Z')
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
    knownConj = ObsConj^lambda^Time;
#ifdef DIRAC_ON_LAST_OPENING
    jd = plJointDistribution(X^Opening^LastOpening^knownConj, P_LastOpening*
#else
    jd = plJointDistribution(X^Opening^knownConj,
#endif
#ifdef X_KNOWING_OPENING
                xLearner.get_computable_object()*P_Opening*listObs*P_lambda
#else
                P_X*P_Opening*listObs*P_lambda
#endif
                *timeLearner.get_computable_object()); // <=> P_Time);
    jd.draw_graph("jd.fig");
#if DEBUG_OUTPUT > 0
    cout << "Joint distribution built." << endl;
#endif

    /**********************************************************************
      PROGRAM QUESTION
     **********************************************************************/
#if PLOT > 1
    X_Op = X^Opening;
    jd.ask(Cnd_P_Time_X_knowing_Op, Time^X, Opening);
#if DEBUG_OUTPUT > 2
    cout << Cnd_P_Time_X_knowing_Op << endl;
#endif

#if PLOT > 2
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

#if PLOT > 2
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
#ifdef PLOT_ONLY
    return 0;
#endif
#endif // endif of PLOT > 1

    jd.ask(Cnd_P_Opening_knowing_rest, Opening, knownConj);
#if DEBUG_OUTPUT > 0
    cout << Cnd_P_Opening_knowing_rest << endl;
#endif

#ifdef BENCH
    positive_classif_finale = 0;
    positive_classif_online = 0;
    positive_classif_online_after = 0;
    cpositive_classif_finale = 0;
    for (vector<string>::const_iterator it = openings.begin();
            it != openings.end(); ++it)
    {
        plValues tmp(Opening);
        tmp[Opening] = *it;
        cumulative_prob.insert(make_pair<plValues, plProbValue>(tmp, 0.0));
    }
#endif
}

OpeningPredictor::~OpeningPredictor()
{
}

void OpeningPredictor::init_game()
{
#ifdef BENCH
    times_label_predicted = 0;
    times_label_predicted_after = 0;
#endif
    evidence = plValues(knownConj);
    // we assume we didn't see any buildings
    for (unsigned int i = 1; i < nbBuildings; ++i)
        evidence[observed[i]] = 0;
    evidence[lambda] = 1; // we want coherence ;)
    evidence[observed[0]] = 1; // the first Nexus/CC/Hatch exists
#if PLOT > 0
    tmpProbV.clear();
#ifdef DIRAC_ON_LAST_OPENING
    P_LastOpening.tabulate(tmpProbV);
#else 
    P_Opening.tabulate(tmpProbV);
#endif
    T_P_Opening_v.push_back(tmpProbV);
#endif
}

int OpeningPredictor::instantiate_and_compile(int time,
        const Building& building, const string& tmpOpening)
{
#ifdef BENCH
    clock_t start = clock();
#endif
    evidence[observed[building.getEnumValue()]] = 1;
    evidence[Time] = time;

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
        << building << " ======" << endl;
    cout << T_P_Opening << endl;
#endif
#ifdef BENCH
    if (T_P_Opening.is_null())
        return -1;
    vector<pair<plValues, plProbValue> > outvals;
    T_P_Opening.sorted_tabulate(outvals);
#if PLOT > 0
    vector<plValues> dummy;
    tmpProbV.clear();
    T_P_Opening.tabulate(dummy, tmpProbV);
    T_P_Opening_v.push_back(tmpProbV);
#endif
    for (vector<pair<plValues, plProbValue> >::const_iterator 
            jt = outvals.begin(); jt != outvals.end(); ++jt)
    {
        cumulative_prob[jt->first] += jt->second;
    }
#endif
#ifdef DIRAC_ON_LAST_OPENING
    P_LastOpening.mutate(static_cast<plDistribution>(
                PP_Opening.compile().rename(LastOpening)));
#endif
#ifdef BENCH
    plValues toTest(Opening);
    toTest[Opening] = tmpOpening;
    if (T_P_Opening.best()[Opening] == toTest[Opening])
        //&& T_P_Opening[toTest[Opening]] > 0.5)
        ++times_label_predicted ;
    if (time > 180 && T_P_Opening.best()[Opening] == toTest[Opening])
        ++times_label_predicted_after;
#endif
#ifdef BENCH
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    time_taken_prediction.push_back(duration);
#if DEBUG_OUTPUT > 2
    cout << "TIME: instantiate+compile took: "
       << duration << " sec" << endl;
#endif
#endif
    return 0;
}

int OpeningPredictor::quit_game(const string& tmpOpening, int noreplay)
{
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
        return -1;
    if (toTest[Opening] == T_P_Opening.best()[Opening])
        ++positive_classif_finale;
    if (toTest[Opening] == max(cumulative_prob)[Opening])
        ++cpositive_classif_finale;
    if (times_label_predicted >= 2) // TODO change
        ++positive_classif_online;
    if (times_label_predicted_after >= 1)
        ++positive_classif_online_after;
#endif
#if PLOT > 0
    std::stringstream tmpfn;
#ifdef DIRAC_ON_LAST_OPENING
    tmpfn << "OpeningsRep" << noreplay << ".gnuplot";
#else
    tmpfn << "SFOpeningsRep" << noreplay << ".gnuplot";
#endif
    gnuplot_vector_probas(T_P_Opening_v, openings, tmpfn.str());
    T_P_Opening_v.clear();
#endif
#ifdef DIRAC_ON_LAST_OPENING
    P_LastOpening.mutate(static_cast<plDistribution>(
                plProbTable(LastOpening, tableOpening, false)));
#endif
    return 0;
}

void OpeningPredictor::results(int noreplay)
{
#ifdef BENCH
    cout << ">>> Positive classif online after 3 minutes: " 
        << positive_classif_online_after
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(positive_classif_online_after)/noreplay << endl;
    cout << ">>> Positive classif online: " << positive_classif_online
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(positive_classif_online)/noreplay << endl;
    cout << ">>> Positive classif: " << positive_classif_finale
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(positive_classif_finale)/noreplay << endl;
    cout << ">>> Cumulative positive classif: " << cpositive_classif_finale
        << " on " << noreplay << " replays, ratio: "
        << static_cast<double>(cpositive_classif_finale)/noreplay << endl;

    double mean_time_taken = 0.0;
    double stddev_time_taken = 0.0;
    for (vector<double>::const_iterator it = time_taken_prediction.begin();
            it != time_taken_prediction.end(); ++it)
        mean_time_taken += *it;
    mean_time_taken /= time_taken_prediction.size();
    for (vector<double>::const_iterator it = time_taken_prediction.begin();
            it != time_taken_prediction.end(); ++it)
        stddev_time_taken += (*it - mean_time_taken)*(*it - mean_time_taken);
    stddev_time_taken /= time_taken_prediction.size();
    stddev_time_taken = sqrt(stddev_time_taken);
    cout << "TIME: prediction mean: " << mean_time_taken << ", stddev: "
        << stddev_time_taken << endl;
#endif
}

int main(int argc, const char *argv[])
{
    /**********************************************************************
      INITIALIZATION
     **********************************************************************/
    std::vector<std::string> terran_openings;
#ifdef MY_OPENINGS_LABELS
    terran_openings.push_back("bio");
    terran_openings.push_back("rax_fe");
//    terran_openings.push_back("siege_exp");
    terran_openings.push_back("two_facto");
    terran_openings.push_back("vultures");
//    terran_openings.push_back("air");
    terran_openings.push_back("drop");
    terran_openings.push_back("unknown");
#else
    terran_openings.push_back("Bio");
    terran_openings.push_back("TwoFactory");
    terran_openings.push_back("VultureHarass");
    terran_openings.push_back("SiegeExpand");
    terran_openings.push_back("Standard");
    terran_openings.push_back("FastDropship");
    terran_openings.push_back("Unknown");
#endif

    std::vector<std::string> protoss_openings;
#ifdef MY_OPENINGS_LABELS
    protoss_openings.push_back("two_gates");
    protoss_openings.push_back("fast_dt");
    protoss_openings.push_back("templar");
    protoss_openings.push_back("speedzeal");
    protoss_openings.push_back("corsair");
    protoss_openings.push_back("nony");
    protoss_openings.push_back("reaver_drop");
    protoss_openings.push_back("unknown");
#else
    protoss_openings.push_back("FastLegs");
    protoss_openings.push_back("FastDT");
    protoss_openings.push_back("FastObs");
    protoss_openings.push_back("ReaverDrop");
    protoss_openings.push_back("Carrier");
    protoss_openings.push_back("FastExpand");
    protoss_openings.push_back("Unknown");
#endif


    std::vector<std::string> zerg_openings;
#ifdef MY_OPENINGS_LABELS
    //zerg_openings.push_back("speedlings");
    zerg_openings.push_back("fast_mutas");
    zerg_openings.push_back("mutas");
    zerg_openings.push_back("lurkers");
    zerg_openings.push_back("hydras");
    zerg_openings.push_back("unknown");
#else
    zerg_openings.push_back("TwoHatchMuta");
    zerg_openings.push_back("ThreeHatchMuta");
    zerg_openings.push_back("HydraRush");
    zerg_openings.push_back("Standard");
    zerg_openings.push_back("HydraMass");
    zerg_openings.push_back("Lurker");
    zerg_openings.push_back("Unknown");
#endif

    std::vector<std::string> openings;

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
            cerr << "ERROR in the first argument" << endl;
            usage();
            return 1;
        }
    }
    else
    {
        cerr << "ERROR in the first argument" << endl;
        usage();
        return 1;
    }

    OpeningPredictor op = OpeningPredictor(openings, argv[1]);

    if (argc < 2)
        return 1;
    ifstream inputfile_test(argv[2]);
    string input;
    cout << endl;
    cout << ">>>> Testing from: " << argv[2] << endl;
    unsigned int noreplay = 0;

    while (getline(inputfile_test, input))
    {
        if (input.empty())
            break;
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening == "")
            continue;
        multimap<int, Building> tmpBuildings;
        getBuildings(input, tmpBuildings, LEARN_TIME_LIMIT);
        tmpBuildings.erase(0); // key == 0 i.e. buildings not constructed
        if (tmpBuildings.empty())
            continue;

#if DEBUG_OUTPUT >= 1
        cout << "******** end replay number: " 
            << noreplay << " ********" << endl;
        cout << endl << endl << endl;
        cout << "******** Real Opening: " << tmpOpening 
            << " replay number: " << noreplay
            << " ********" << endl;
#endif
#ifdef BENCH
        ++noreplay;
#endif

        op.init_game();

        // we assume we see the buildings as soon as they get constructed
        for (multimap<int, Building>::const_iterator it 
                = tmpBuildings.begin(); 
                it != tmpBuildings.end(); ++it)
        {

            if (!op.instantiate_and_compile(it->first, it->second,
                        tmpOpening))
                continue;

        }

        if (!op.quit_game(tmpOpening, noreplay))
            continue;

    }
    
    op.results(noreplay);

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
