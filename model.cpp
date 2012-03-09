#include "model.h"
#define SIGNAL_NOISE_RATIO 1.0
#define FIXED_ERROR_RATE 1

#define MIN_POINTS_BELL_SHAPES // add virtual points to bell shapes w/o enough
#define MIN_STD_DEV 30
#define TIME_MULTIPLICATOR 1
#ifdef __SERIALIZE__
//http://www.boost.org/doc/libs/1_45_0/libs/serialization/doc/index.html
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/set.hpp"
#endif

// TODO add tree_distance to existing set_distances

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

using namespace std;

/// POSSIBLE: use an iterator in values[] so that we directly
/// POSSIBLE: put a std::set instead of std::vector for vector_X
/// TODO: replace set by unordered_set (TR1 or boost) in a lot of places

#ifdef __SERIALIZE__
struct serialized_tables
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & tabulated_P_Time_X_Op;
        ar & tabulated_P_X_Op;
        ar & openings;
        ar & vector_X;
        ar & distances_X;
    }
    vector<long double> tabulated_P_Time_X_Op;
    vector<long double> tabulated_P_X_Op;
    vector<string> openings;
    vector<set<int> > vector_X;
    vector<vector<int> > distances_X;
    serialized_tables() {};
    serialized_tables(const vector<long double>& time_x,
            const vector<long double>& x,
            const vector<string>& op,
            const vector<set<int> >& vx,
            const vector<vector<int> >& dx)
        : tabulated_P_Time_X_Op(time_x)
        , tabulated_P_X_Op(x)
        , openings(op)
        , vector_X(vx)
        , distances_X(dx)
    {}
};
#endif

tech_trees tt;
int nbBuildings;
const char** buildings_name;
typedef boost::minstd_rand base_generator_type;
boost::uniform_real<> uni_dist(0,1);
base_generator_type generator(42u);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uniform01(generator, uni_dist);

double fmean(const vector<double>& v)
{
    double m = 0.0;
    for (vector<double>::const_iterator it = v.begin();
            it != v.end(); ++it)
        m += *it;
    m /= v.size();
    return m;
}

double fstddev(const vector<double>& v, double m)
{
    double s = 0.0;
    for (vector<double>::const_iterator it = v.begin();
            it != v.end(); ++it)
        s += (*it - m)*(*it - m);
    s /= v.size();
    s = sqrt(s);
    return s;
}

template <class T>
std::vector<std::vector<T> > transpose(const std::vector<std::vector<T> >& t)
{
    std::vector<std::vector<T> > ret;
    for (typename std::vector<std::vector<T> >::const_iterator it = t.begin();
            it != t.end(); ++it)
    {
        for (typename std::vector<T>::const_iterator jt = it->begin();
                jt != it->end(); ++jt)
        {
            typename std::vector<T> ttmp;
            ttmp.reserve(it->size());
            ret.push_back(ttmp);
        }
    }
    for (unsigned int i = 0; i < ret.size(); ++i)
        for (unsigned int j = 0; j < ret[i].size(); ++j)
            ret[i][j] = t[j][i];
    return ret;
}

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
#ifdef TECH_TREES
void gnuplot_vector_probas_tt(const vector<vector<plProbValue> >& tab, 
        const string& filename)
{
    vector<unsigned int> to_count;
    vector<bool> to_c;
    vector<vector<plProbValue> >::const_iterator iit = tab.begin();
    for (vector<plProbValue>::const_iterator jt = iit->begin();
            jt != iit->end(); ++jt)
        to_c.push_back(false);
    ++iit;
    for (; iit != tab.end(); ++iit)
    {
        unsigned int i = 0;
        for (vector<plProbValue>::const_iterator jt = iit->begin();
                jt != iit->end(); ++jt)
        {
            if (*jt > 0.01)
                to_c[i] = true;
            ++i;
        }
    }
    for (unsigned int i =0; i < to_c.size(); ++i)
    {
        if (to_c[i])
            to_count.push_back(i);
    }
    if (to_count.empty())
        return;
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
            << "\"" << to_count[0];
    for (unsigned int i = 1; i < to_count.size(); ++i)
    {
        file << "\", \"" << cpfn << "\" using 1:" << i+2 << " title " 
            << "\"" << to_count[i];
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
        unsigned int j = 0;
        for (vector<plProbValue>::const_iterator jt = it->begin();
                jt != it->end(); ++jt)
        {
            if (to_c[j])
                file << *jt << " ";
            ++j;
        }
        file << endl;
    }
    file.close();
}
#endif
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
    set<int> setX = tt.vector_X[X_Obs_conj[0]];
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
#ifdef LAPLACE_LEARNING
            plCndLearnObject<plLearnLaplace>& timeLearner,
#else
            plCndLearnObject<plLearnBellShape>& timeLearner,
#endif
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
    map<pair<int, string>, plValues> one_value_per_X_Op;
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
                int tmp_ind = get_X_indice(tmpSet, tt.vector_X);
                vals_timeLearner[X] = tmp_ind;
                vals_xLearner[X] = tmp_ind;
#if DEBUG_OUTPUT > 1
                std::cout << "X ind: " << tmp_ind
                    << std::endl;
#endif
                vals_timeLearner[Time] = TIME_MULTIPLICATOR*it->first;
#if DEBUG_OUTPUT > 1
                std::cout << "Time: " << it->first << std::endl;
#endif
#ifdef ERROR_CHECKS
                pair<int, string> tmp_pair(tmp_ind, tmpOpening);
                if (count_X_Op_examples.count(tmp_pair))
                {
                    count_X_Op_examples[tmp_pair] = 
                        count_X_Op_examples[tmp_pair] + 1;
                }
                else
                {
                    count_X_Op_examples.insert(make_pair<pair<int, string>, 
                            int>(tmp_pair, 1));
                    one_value_per_X_Op.insert(make_pair<pair<int, string>,
                        plValues>(tmp_pair, vals_timeLearner));
                    /// TODO <1> init with bell shape this point as mean
                    ///timeLearner = plLearnBellShape(vals_timeLearner,
                    ///        it->first, 3.0, PL_ONE);
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
    for (map<pair<int, string>, int>::const_iterator it = count_X_Op_examples.begin();
            it != count_X_Op_examples.end(); ++it)
    {
        if (it->second == 0)
        {
            cout << "PROBLEM: We never encountered X: ";
            for (typename set<int>::const_iterator ibn
                    = tt.vector_X[it->first.first].begin();
                    ibn != tt.vector_X[it->first.first].end(); ++ibn)
            {
                Building tmpBuilding(static_cast<T>(*ibn));
                cout << tmpBuilding << ", ";
            }
            cout << endl;
        }
#ifdef MIN_POINTS_BELL_SHAPES
        else if (it->second < 3)
        {
            cout << "We encountered X " << it->first.first
               << " + Op " << it->first.second << " less than three times: ";
            for (typename set<int>::const_iterator ibn
                    = tt.vector_X[it->first.first].begin();
                    ibn != tt.vector_X[it->first.first].end(); ++ibn)
            {
                Building tmpBuilding(static_cast<T>(*ibn));
                cout << tmpBuilding << ", ";
            }
            cout << endl;
            /////////////////////////////////////////////////
            // put another (slightly different) value: change, c.f. TODO <1>
            one_value_per_X_Op[it->first][Time] = 
                one_value_per_X_Op[it->first][Time] + 10; /// totally arbitrary 10
            timeLearner.add_point(one_value_per_X_Op[it->first]);
            one_value_per_X_Op[it->first][Time] = 
                one_value_per_X_Op[it->first][Time] - 20; /// totally arbitrary -20 (-10)
            timeLearner.add_point(one_value_per_X_Op[it->first]);
            /////////////////////////////////////////////////
        }
#endif
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
    int shift = 0;
    if (learningFileName[0] == 'l')
        shift = 1;
    /**********************************************************************
      VARIABLES SPECIFICATION
     **********************************************************************/
    X = plSymbol("X", plIntegerType(0, tt.vector_X.size()-1));
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
#ifdef WITH_OPENINGS_PRIOR
    vector<double> tmp = prior_openings(learningFileName[0 + shift],
            learningFileName[2 + shift]);
    for (vector<double>::const_iterator i = tmp.begin(); i != tmp.end(); ++i)
        tableOpening.push_back(*i);
#else
    for (unsigned int i = 0; i < openings.size(); i++) 
        tableOpening.push_back(1.0);
#endif
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
    for (unsigned int i = 0; i < tt.vector_X.size(); i++) 
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
#ifdef LAPLACE_LEARNING
    timeLearner = plCndLearnObject<plLearnLaplace>(Time, X^Opening);
#else
    timeLearner = plCndLearnObject<plLearnBellShape>(Time, X^Opening);
#endif
    cout << ">>>> Learning from: " << learningFileName << endl;
    ifstream inputstream(learningFileName);
    if (learningFileName[0 + shift] == 'P')
        learn_T_and_X<Protoss_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (learningFileName[0 + shift] == 'T')
        learn_T_and_X<Terran_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
    else if (learningFileName[0 + shift] == 'Z')
        learn_T_and_X<Zerg_Buildings>(inputstream, timeLearner, xLearner,
                Opening, X, Time);
#if DEBUG_OUTPUT > 2
    cout << "*** Number of possible pairs (X, Opening): "
        << tt.vector_X.size()*openings.size();
    cout << timeLearner.get_computable_object() << endl;
#endif

#ifdef __MIN_STD_DEV_BELL_SHAPES__
    time_knowing_x_op = plDistributionMap(timeLearner.get_computable_object().get_left_variables(), timeLearner.get_computable_object().get_right_variables());
    time_knowing_x_op.push_default(plUniform(Time));
    plValues val(timeLearner.get_computable_object().get_right_variables());
    do {
#ifdef LAPLACE_LEARNING
        const plLearnLaplace* t = timeLearner.get_learnt_object_for_value(val);
#else
        const plLearnBellShape* t = timeLearner.get_learnt_object_for_value(val);
#endif
        if (t)
        {
            if (t->get_sigma() < MIN_STD_DEV)
            {
                cout << "--> min stddev" << endl;
                time_knowing_x_op.push(plBellShape(Time, t->get_mu(), MIN_STD_DEV), val);
            }
            else
            {
                cout << "--> bell shape" << endl;
                time_knowing_x_op.push(plBellShape(Time, t->get_mu(), t->get_sigma()), val);
            }
        }
        else
            cout << "-> uniform (default)" << endl;
    } while (val.next());
#endif
#ifdef __SERIALIZE__
    /* // Verification of tabulate alignment / order
    plValues val(timeLearner.get_computable_object().get_right_variables());
    do {
#ifdef LAPLACE_LEARNING
        const plLearnLaplace* t = timeLearner.get_learnt_object_for_value(val);
#else
        const plLearnBellShape* t = timeLearner.get_learnt_object_for_value(val);
#endif
        cout << val << endl;
        if (t)
        {
            cout << t->get_distribution() << endl;
            t->get_distribution().tabulate(cout);

            for (unsigned int i = 0; i < LEARN_TIME_LIMIT; ++i)
                cout << i+1 << " " << tabulated_P_Time_X_Op[val[X]*openings.size()*LEARN_TIME_LIMIT + val[Opening]*LEARN_TIME_LIMIT + i] << endl;
        }
        else
            cout << "P(Time) = Uniform: 1/" << LEARN_TIME_LIMIT << " = " << 1.0/LEARN_TIME_LIMIT << endl;
    } while (val.next());
    /**/

    vector<plProbValue> tmpTime_X;
#ifdef __MIN_STD_DEV_BELL_SHAPES__
    time_knowing_x_op.tabulate(tmpTime_X);
#else
    timeLearner.get_computable_object().tabulate(tmpTime_X);
#endif
    vector<plProbValue> tmpX;
    xLearner.get_computable_object().tabulate(tmpX);
    vector<long double> tmp1;
    tmp1.reserve(tmpTime_X.size());
    for (vector<plProbValue>::const_iterator it = tmpTime_X.begin();
            it != tmpTime_X.end(); ++it)
        tmp1.push_back(*it);
    vector<long double> tmp2;
    for (vector<plProbValue>::const_iterator it = tmpX.begin();
            it != tmpX.end(); ++it)
        tmp2.push_back(*it);
    serialized_tables st(tmp1, tmp2, op, tt.vector_X, tt.set_distances_X);

    string filename(learningFileName);
    filename = filename.substr(0, filename.find('.'));
    filename.append(".table");
    std::ofstream ofs(filename.c_str());
    {
        boost::archive::text_oarchive oa(ofs);
        oa << st;
    }
    cout << "Serialized learned tables as " << filename << endl;
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
#ifdef __MIN_STD_DEV_BELL_SHAPES__
                *time_knowing_x_op); // <=> P_Time
#else
                *timeLearner.get_computable_object()); // <=> P_Time);
#endif
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
        for (unsigned int j = 0; j < tt.vector_X.size(); ++j)
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

#ifdef TECH_TREES
    jd.ask(Cnd_P_X_knowing_obs, X, knownConj);
#endif
    jd.ask(Cnd_P_Opening_knowing_rest, Opening, knownConj);

#if DEBUG_OUTPUT > 0
#ifdef TECH_TREES
    cout << Cnd_P_X_knowing_obs << endl;
#endif
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
#ifdef TECH_TREES
    current_x.clear();
    current_x.insert(0);
    save_nbinferences = nbinferences;
    tmeanc_set_distance_X = 0.0;
    tbestc_set_distance_X = 0.0;
    tmeanc_tree_distance_X = 0.0;
    tbestc_tree_distance_X = 0.0;
    tbestp_set_distance_X = 0.0;
    tmeanp_set_distance_X = 0.0;
    tbestp_tree_distance_X = 0.0;
    tmeanp_tree_distance_X = 0.0;
    tprediction_best_set_X = 0.0;
    tprediction_mean_set_X = 0.0;
#endif
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
#ifdef TECH_TREES
    tmpProbV.clear();
    P_X.tabulate(tmpProbV);
    T_P_X_v.push_back(tmpProbV);
    old_T_P_X.clear();
#endif
#endif
}

int OpeningPredictor::instantiate_and_compile(int time,
        const Building& building, const string& tmpOpening)
{
    time = TIME_MULTIPLICATOR*time;
#ifdef BENCH
    clock_t start = clock();
#ifdef TECH_TREES
    current_x.insert(building.getEnumValue());
#endif
#endif
    if (uniform01() > SIGNAL_NOISE_RATIO)
        return -1;
    ++nbinferences;
    evidence[observed[building.getEnumValue()]] = 1;
    evidence[Time] = time;

#if DEBUG_OUTPUT > 1
    cout << "====== evidence ======" << endl;
    cout << evidence << endl;
#endif

#ifdef TECH_TREES
    plDistribution PP_X;
    Cnd_P_X_knowing_obs.instantiate(PP_X, evidence);
#endif
    plDistribution PP_Opening;
    Cnd_P_Opening_knowing_rest.instantiate(PP_Opening, evidence);

#if DEBUG_OUTPUT > 1
    cout << "====== P(Opening | rest).instantiate ======" << endl;
    cout << Cnd_P_Opening_knowing_rest << endl;
    cout << PP_Opening.get_left_variables() << endl;
    cout << PP_Opening.get_right_variables() << endl;
#endif
#ifdef TECH_TREES
    PP_X.compile(T_P_X);
#endif
    PP_Opening.compile(T_P_Opening);
#if DEBUG_OUTPUT >= 1
#ifdef TECH_TREES
    cout << "====== P(X | evidence), building: "
        << building << " ======" << endl;
    cout << T_P_X << endl << endl;
#endif
    cout << "====== P(Opening | evidence), building: "
        << building << " ======" << endl;
    cout << T_P_Opening << endl;
#endif
#ifdef BENCH
    vector<pair<plValues, plProbValue> > outvals;
    if (T_P_Opening.is_null())
        return -1;
    T_P_Opening.sorted_tabulate(outvals);
#if PLOT > 0
    vector<plValues> values_v;
    tmpProbV.clear();
    T_P_Opening.tabulate(values_v, tmpProbV);
    T_P_Opening_v.push_back(tmpProbV);
#ifdef TECH_TREES
    tmpProbV.clear();
    values_v.clear();
    T_P_X.tabulate(values_v, tmpProbV);
    T_P_X_v.push_back(tmpProbV);
    old_T_P_X.push_back(T_P_X);
#endif
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
        ++times_label_predicted;
    if (time > 180 && T_P_Opening.best()[Opening] == toTest[Opening])
        ++times_label_predicted_after;
#ifdef TECH_TREES
    plValues toTestX(X);
    int tmp = get_X_indice(current_x, tt.vector_X);
    toTestX[X] = tmp;
    tbestc_set_distance_X += tt.set_distances_X[T_P_X.best()[X]][tmp];
    for (unsigned int i = 0; i < tmpProbV.size(); ++i)
        tmeanc_set_distance_X += tt.set_distances_X[values_v[i][X]][tmp] 
            * tmpProbV[i];
#endif
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
#if DEBUG >= 1
    for (map<plValues, plProbValue>::const_iterator 
            jt = cumulative_prob.begin(); jt != cumulative_prob.end(); ++jt)
    {
        cout << "|||| ";
        jt->first.Output(cout);
        cout << " => sum on probs: " << jt->second << endl;
    }
#endif
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
#ifdef TECH_TREES
    if (current_x.size() > 3) 
    {
        int tmp = get_X_indice(current_x, tt.vector_X);
        plValues toTestXOld(X);
        toTestXOld[X] = tmp;
        int mindist = 10000000; // lol
        int minind = 1000000;
        for (unsigned int i = 0; i < old_T_P_X.size(); ++i)
        {
            int tmpdist = tt.set_distances_X[old_T_P_X[i].best()[X]][tmp];
            if (tmpdist < FIXED_ERROR_RATE)
            {
                tprediction_best_set_X = current_x.size() - (i + current_x.size() - (nbinferences - save_nbinferences));
                break;
            }
        }
        for (unsigned int i = 0; i < old_T_P_X.size(); ++i)
        {
            tmpProbV.clear();
            vector<plValues> tmpValues;
            old_T_P_X[i].tabulate(tmpValues, tmpProbV);
            int tmpdist = 0;
            for (unsigned int j = 0; j < tmpValues.size(); ++j)
                tmpdist += tt.set_distances_X[tmpValues[j][X]][tmp] 
                    * tmpProbV[j];
            if (tmpdist < FIXED_ERROR_RATE)
            {
                tprediction_mean_set_X = current_x.size() - (i + current_x.size() - (nbinferences - save_nbinferences));
                break;
            }
        }
        prediction_best_set_X.push_back(tprediction_best_set_X);
        prediction_mean_set_X.push_back(tprediction_mean_set_X);
    }
    double tmpmeanc;
    double tmpbestc;
    if (nbinferences != save_nbinferences)
    {
        tmpmeanc = tmeanc_set_distance_X
            / (nbinferences - save_nbinferences);
        tmpbestc = tbestc_set_distance_X
            / (nbinferences - save_nbinferences);

        meanc_set_distance_X.push_back(tmpmeanc);
        bestc_set_distance_X.push_back(tmpbestc);
    }
//    else
//    {
//        tmpmeanc = 0.0;
//        tmpbestc = 0.0;
//    }
#endif
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
#ifdef TECH_TREES
    std::stringstream tmpfn2;
    tmpfn2 << "TechTreeRep" << noreplay << ".gnuplot";
    gnuplot_vector_probas_tt(T_P_X_v, tmpfn2.str());
    T_P_X_v.clear();
#endif
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

    double mean_time_taken = fmean(time_taken_prediction);
    double stddev_time_taken = fstddev(time_taken_prediction, mean_time_taken);
    cout << "TIME: prediction mean: " << mean_time_taken << ", stddev: "
        << stddev_time_taken << endl;
#ifdef TECH_TREES
    double mean = fmean(meanc_set_distance_X);
    double stddev = fstddev(meanc_set_distance_X, mean);
    cout << endl << ">>> Mean of mean set-distance X: " << mean
        << " stddev/sigma: " << sqrt(stddev) << endl;

    mean = fmean(bestc_set_distance_X);
    stddev = fstddev(bestc_set_distance_X, mean);
    cout << endl << ">>> Mean of best set-distance X: " << mean
        << " stddev/sigma: " << sqrt(stddev) << endl;

    mean = fmean(prediction_best_set_X);
    stddev = fstddev(prediction_best_set_X, mean);
    cout << endl << ">>> Best of best set-distance predictive power (nb of observations ahead): " << mean
        << " stddev/sigma: " << sqrt(stddev) << endl;

    mean = fmean(prediction_mean_set_X);
    stddev = fstddev(prediction_mean_set_X, mean);
    cout << endl << ">>> Best of mean set-distance predictive power (nb of observations ahead): " << mean
        << " stddev/sigma: " << sqrt(stddev) << endl;

#endif

    cout << endl << ">>> Number of replays: " << noreplay << endl
        << "Number of inferences: " << nbinferences << endl
        << "Mean inference/replay: " << (double)nbinferences/(double)noreplay
        << endl;
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

    if (argv[1] != NULL)
    {
        char their_race = 'X';
        char our_race = 'X';
        if (argv[1][1] == 'P' || argv[1][1] == 'T' || argv[1][1] == 'Z')
        {
            their_race = argv[1][1];
            if (argv[1][3] == 'P' || argv[1][3] == 'T' || argv[1][3] == 'Z')
                our_race = argv[1][3];
        }
        else if (argv[1][0] == 'P' || argv[1][0] == 'T' || argv[1][0] == 'Z')
        {
            their_race = argv[1][0];
            if (argv[1][2] == 'P' || argv[1][2] == 'T' || argv[1][2] == 'Z')
                our_race = argv[1][2];
        }
        else
        {
            cerr << "ERROR in the first argument" << endl;
            usage();
            return 1;
        }
        /// match up: Enemy vs Us
        /// For instance ZvT will get Zerg buildings
        stringstream extract_X_from;
        if (our_race != 'X')
        {
            cout << "We are " << our_race
                << " against " << their_race << endl;
            extract_X_from << their_race << "v" << our_race << ".txt";
        }
        else
        {
            extract_X_from << "l" << their_race << "all.txt";
        }
        ifstream fin(extract_X_from.str().c_str());
        tt = tech_trees(fin); /// Enemy race
        cout << "X size: " << tt.vector_X.size() << endl;
        if (their_race == 'P')
        {
            openings = protoss_openings;
            nbBuildings = NB_PROTOSS_BUILDINGS;
            buildings_name = protoss_buildings_name;
        }
        else if (their_race == 'T')
        {
            openings = terran_openings;
            nbBuildings = NB_TERRAN_BUILDINGS;
            buildings_name = terran_buildings_name;
        }
        else if (their_race == 'Z')
        {
            openings = zerg_openings;
            nbBuildings = NB_ZERG_BUILDINGS;
            buildings_name = zerg_buildings_name;
        }
    }
    else
    {
        cerr << "ERROR in the first argument" << endl;
        usage();
        return 1;
    }

    OpeningPredictor op = OpeningPredictor(openings, argv[1]);

    if (argc < 3)
        return 0;
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

            op.instantiate_and_compile(it->first, it->second,
                    tmpOpening);
        }

        op.quit_game(tmpOpening, noreplay);

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

std::vector<double> OpeningPredictor::prior_openings(char them, char us)
{
    std::vector<double> r;
    if (them == 'P')
    {
        if (us == 'P') 
        {
            r.push_back(332);
            r.push_back(7);
            r.push_back(3);
            r.push_back(5);
            r.push_back(3);
            r.push_back(55);
            r.push_back(13);
            r.push_back(62);
        }
        else if (us == 'T')
        {
            r.push_back(252);
            r.push_back(87);
            r.push_back(17);
            r.push_back(100);
            r.push_back(4);
            r.push_back(449);
            r.push_back(86);
            r.push_back(35);
        }
        else if (us == 'Z')
        {
            r.push_back(304);
            r.push_back(20);
            r.push_back(22);
            r.push_back(98);
            r.push_back(190);
            r.push_back(114);
            r.push_back(19);
            r.push_back(99);
        }
    }
    else if (them == 'T')
    {
        if (us == 'P') 
        {
            r.push_back(62);
            r.push_back(438);
            r.push_back(240);
            r.push_back(122);
            r.push_back(52);
            r.push_back(93);
        }
        else if (us == 'T')
        {
            r.push_back(25);
            r.push_back(377);
            r.push_back(127);
            r.push_back(3);
            r.push_back(10);
            r.push_back(34);
        }
        else if (us == 'Z')
        {
            r.push_back(197);
            r.push_back(392);
            r.push_back(116);
            r.push_back(3);
            r.push_back(121);
            r.push_back(43);
        }
    }
    else if (them == 'Z')
    {
        if (us == 'P') 
        {
            r.push_back(109);
            r.push_back(253);
            r.push_back(181);
            r.push_back(329);
            r.push_back(16);
        }
        else if (us == 'T')
        {
            r.push_back(199);
            r.push_back(391);
            r.push_back(286);
            r.push_back(206);
            r.push_back(11);
        }
        else if (us == 'Z')
        {
            r.push_back(508);
            r.push_back(559);
            r.push_back(5);
            r.push_back(17);
            r.push_back(1);
        }
    }
    return r;
}
