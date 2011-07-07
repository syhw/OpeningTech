#include <pl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include "x_values.h"
#include "enums_name_tables.h"
#include "replays.h"
#include "parameters.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

typedef void iovoid;

class OpeningPredictor
{
    std::vector<std::string> openings;
    /// Variables specification
    plSymbol X;
    std::vector<plSymbol> observed;
    plSymbol lambda;
    plSymbol Opening;
#ifdef DIRAC_ON_LAST_OPENING
    plSymbol LastOpening;
#endif
    plSymbol Time;
    /// Parametric forms specification
    std::vector<plProbValue> tableOpening;
#ifdef DIRAC_ON_LAST_OPENING
    plMutableDistribution P_LastOpening;
    plExternalFunction same_opening;
    plFunctionalDirac P_Opening;
#else
    plProbTable P_Opening;
#endif
    plCndLearnObject<plLearnHistogram> xLearner;
    plProbTable P_X;
    plComputableObjectList listObs;
    std::vector<plProbTable> P_Observed;
    plVariablesConjunction ObsConj;
    plVariablesConjunction X_Obs_conj;
    plExternalFunction coherence;
    plFunctionalDirac P_lambda;
    plCndLearnObject<plLearnBellShape> timeLearner;
#ifdef __MIN_STD_DEV_BELL_SHAPES__
    plDistributionMap time_knowing_x_op;
#endif
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
#ifdef TECH_TREES
    plCndDistribution Cnd_P_X_knowing_obs;
#endif
    plCndDistribution Cnd_P_Opening_knowing_rest;

#ifdef BENCH
    unsigned int positive_classif_finale;
    unsigned int positive_classif_online;
    unsigned int positive_classif_online_after;
    unsigned int cpositive_classif_finale;
    std::map<plValues, plProbValue> cumulative_prob;
#ifdef TECH_TREES
    std::vector<plDistribution> old_T_P_X;
    std::set<int> current_x;
    int save_nbinferences;
    double tmeanc_set_distance_X; // current
    double tbestc_set_distance_X; // current
    double tmeanc_tree_distance_X; // current
    double tbestc_tree_distance_X; // current
    double tmeanp_set_distance_X; // prediction
    double tbestp_set_distance_X; // prediction
    double tmeanp_tree_distance_X; // prediction
    double tbestp_tree_distance_X; // prediction
    double tprediction_best_set_X;
    double tprediction_mean_set_X;

    std::vector<double> meanc_set_distance_X;
    std::vector<double> bestc_set_distance_X;
    std::vector<double> meanc_tree_distance_X;
    std::vector<double> bestc_tree_distance_X;
    std::vector<double> meanp_set_distance_X;
    std::vector<double> bestp_set_distance_X;
    std::vector<double> meanp_tree_distance_X;
    std::vector<double> bestp_tree_distance_X;
    std::vector<double> prediction_best_set_X;
    std::vector<double> prediction_mean_set_X;
#endif
    unsigned int times_label_predicted;
    unsigned int times_label_predicted_after;
    std::vector<double> time_taken_prediction;
#endif

    // Game wise computations
    plValues evidence;
    int nbinferences;
#if PLOT > 0
    std::vector<std::vector<plProbValue> > T_P_Opening_v;
#ifdef TECH_TREES
    std::vector<std::vector<plProbValue> > T_P_X_v;
#endif
    std::vector<plProbValue> tmpProbV;
#endif

    std::vector<double> prior_openings(char them, char us);

    public:
#ifdef TECH_TREES
        plDistribution T_P_X;
#endif
        plDistribution T_P_Opening;
        OpeningPredictor(const std::vector<std::string>& op,
                const char* learningFileName);
        ~OpeningPredictor();
        void init_game();
        int instantiate_and_compile(int time,
                const Building& building, const std::string& tmpOpening);
        int quit_game(const std::string& tmpOpening, int noreplay);
        void results(int noreplay);
};
