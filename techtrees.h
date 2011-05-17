#include <pl.h>
#include "parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include "enums_name_tables_tt.h"
#include "x_values.h"
#include "replays.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

typedef void iovoid;

class TechTreePredictor
{
    /// Variables specification
    plSymbol X; // Build Tree
    std::vector<plSymbol> observed;
    plSymbol lambda;
#ifdef DIRAC_ON_LAST_X
    plSymbol LastX;
#endif
    plSymbol Time;
    /// Parametric forms specification
#ifdef DIRAC_ON_LAST_X
    plMutableDistribution P_LastX;
    plExternalFunction same_x;
    plFunctionalDirac P_X;
#else
    plProbTable P_X;
#endif
    plCndLearnObject<plLearnHistogram> xLearner;
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plVariablesConjunction ObsConj;
    plVariablesConjunction X_Obs_conj;
    plExternalFunction coherence;
    plFunctionalDirac P_lambda;
    plCndLearnObject<plLearnBellShape> timeLearner;
    /// Decomposition
    plVariablesConjunction knownConj;
    plJointDistribution jd;
    /// Program Question
#if PLOT > 1
    plVariablesConjunction X_Op;
    plCndDistribution Cnd_P_Time_X_knowing_Op;
#if PLOT > 2
    plCndDistribution Cnd_P_Time_knowing_X_Op;
#endif
#endif
    plCndDistribution Cnd_P_X_knowing_obs;

#ifdef BENCH
    unsigned int positive_classif_finale;
    unsigned int positive_classif_online;
    unsigned int positive_classif_online_after;
    unsigned int cpositive_classif_finale;
    std::map<plValues, plProbValue> cumulative_prob;
    unsigned int times_label_predicted;
    unsigned int times_label_predicted_after;
    std::vector<double> time_taken_prediction;
#endif

    // Game wise computations
    plValues evidence;
#if PLOT > 0
    std::vector<plProbValue> tmpProbV;
#endif

    public:
        plDistribution T_P_X;
        TechTreePredictor(const std::vector<std::string>& op,
                const char* learningFileName);
        ~TechTreePredictor();
        void init_game();
        int instantiate_and_compile(int time,
                const Building& building);
        int quit_game(int noreplay);
        void results(int noreplay);
};
