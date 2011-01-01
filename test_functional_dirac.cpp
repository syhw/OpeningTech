#include <pl.h>

void test_X_possible(plValues& lambda, const plValues& X_and_Y)
{
    // if X is possible w.r.t. observations
    if (X_and_Y[0] == X_and_Y[1])
        lambda[0] = 1; // true
    else
        lambda[0] = 0; // false
}

int main()
{
    plSymbol X("X", PL_BINARY_TYPE);
    plSymbol Y("Y", PL_BINARY_TYPE);
    plSymbol lambda("lambda", PL_BINARY_TYPE);

    plProbValue tmp_table[] = {0.5, 0.5};
    plProbTable P_X(X, tmp_table, true);
    plProbTable P_Y(Y, tmp_table, true);
    plExternalFunction coherence(lambda, X^Y, test_X_possible);
    plFunctionalDirac P_lambda(lambda, X^Y, coherence);
}
