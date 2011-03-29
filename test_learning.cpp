#include <pl.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include "enums_name_tables.h"
#include "x_values.h"
#include "replays.h"

using namespace std;

int main(int argc, const char *argv[])
{
    vector<int> vector_X;
    vector_X.push_back(0);
    vector_X.push_back(1);
    vector_X.push_back(2);
    std::vector<std::string> openings;
    openings.push_back("Op0");
    openings.push_back("Op1");

    plSymbol X("X", plIntegerType(0, vector_X.size()));
    std::vector<plProbValue> tableX;
    for (unsigned int i = 0; i < vector_X.size(); i++)
        tableX.push_back(1.0); // TOLEARN
    plProbTable P_X(X, tableX, false);

    plSymbol Opening("Opening", plLabelType(openings));
    std::vector<plProbValue> tableOpening;
    for (unsigned int i = 0; i < openings.size(); i++) 
        tableOpening.push_back(1.0); // TOLEARN
    plProbTable P_Opening(Opening, tableOpening, false);

    plSymbol Time("Time", plIntegerType(1,300)); // 18 minutes

    plCndLearnObject<plLearnBellShape> timeLearner(Time, X^Opening);

    plValues vals(timeLearner.get_variables());

    vals[Opening] = "Op0";
    vals[X] = 0;
    vals[Time] = 30;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 0;
    vals[Time] = 32;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 0;
    vals[Time] = 35;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 0;
    vals[Time] = 28;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op1";
    vals[X] = 0;
    vals[Time] = 35;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op1";
    vals[X] = 0;
    vals[Time] = 37;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op1";
    vals[X] = 0;
    vals[Time] = 34;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op1";
    vals[X] = 0;
    vals[Time] = 38;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 1;
    vals[Time] = 60;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 1;
    vals[Time] = 55;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 1;
    vals[Time] = 62;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 1;
    vals[Time] = 65;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 2;
    vals[Time] = 140;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 2;
    vals[Time] = 130;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 2;
    vals[Time] = 145;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();
    vals[Opening] = "Op0";
    vals[X] = 2;
    vals[Time] = 155;
    if (!timeLearner.add_point(vals))
        cout << "ERROR: point not added" << endl;
    vals.reset();

    plVariablesConjunction X_Op = X^Opening;
    plJointDistribution jd(Time^X_Op,
            timeLearner.get_computable_object()*P_X*P_Opening); 

    plCndDistribution Cnd_P_Time_knowing_X_Op;
    jd.ask(Cnd_P_Time_knowing_X_Op, Time, X_Op);
    plValues evidence(X_Op);
    evidence[X] = 0;
    evidence[Opening] = "Op0";
    plDistribution P_Time;
    Cnd_P_Time_knowing_X_Op.instantiate(evidence).compile(P_Time);
    P_Time.plot("test_Time.gnuplot");

    plCndDistribution Cnd_P_Time_knowing_X;
    jd.ask(Cnd_P_Time_knowing_X_Op, Time^X, Opening);
    plValues evidence2(Opening);
    evidence2[Opening] = "Op0";
    plDistribution P_Time_X;
    Cnd_P_Time_knowing_X_Op.instantiate(evidence2).compile(P_Time_X);
    P_Time_X.plot("test_Time_X.gnuplot");

    for (unsigned int i = 0; i < 3; i++)
    {
        plValues rightValues(X^Opening); 
        rightValues[Opening] = "Op0";
        rightValues[X] = i;
        cout << "Learnt parameters, mu: " 
             << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(
                         rightValues)->get_distribution()).mean() 
             << ", stddev: " 
             << static_cast<plBellShape>(timeLearner.get_learnt_object_for_value(
                         rightValues)->get_distribution()).standard_deviation() 
             << endl;
    }

    return 0;
}
