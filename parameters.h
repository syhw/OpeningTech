#ifndef PARAMETERS_HEADER
#define PARAMETERS_HEADER

#define X_KNOWING_OPENING
#define DIRAC_ON_LAST_OPENING

//#define LEARN_TIME_LIMIT 1080 // 18 minutes, TODO change
#define LEARN_TIME_LIMIT 500
//#define GENERATE_X_VALUES
#define DEBUG_OUTPUT 1
#define PLOT 0

#if DEBUG_OUTPUT > 0
#define ERROR_CHECKS
#endif
#define ERROR_CHECKS // TODO change when probt-users will have answered
#define BENCH

#endif
