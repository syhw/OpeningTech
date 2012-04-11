#ifndef PARAMETERS_HEADER
#define PARAMETERS_HEADER

#define X_KNOWING_OPENING
#define DIRAC_ON_LAST_OPENING
//#define LAPLACE_LEARNING // learning histograms with laplace smoothing

#define LEARN_TIME_LIMIT 1080 // 18 minutes, TODO change
//#define LEARN_TIME_LIMIT 900 // 15 minutes
//#define LEARN_TIME_LIMIT 600 // 10 minutes
//#define GENERATE_X_VALUES
//#define DEBUG_OUTPUT 1
//#define TECH_TREES
//#define __SERIALIZE__
#ifndef LAPLACE_LEARNING
#define __MIN_STD_DEV_BELL_SHAPES__ // impose minimum standard deviations
#endif
/**
 * PLOT = 0 => no plotting
 * PLOT = 1 => plot P(Opening) over buildings seen for each replay
 * PLOT = 2 => adds plot P(Time,X|Opening)
 * PLOT = 3 => adds plot P(Time|X,Opening)
 */
//#define PLOT_ONLY
#define PLOT 1

#if DEBUG_OUTPUT > 0
#define ERROR_CHECKS
#endif
#define ERROR_CHECKS // TODO change when probt-users will have answered
#define BENCH

//#define MY_OPENINGS_LABELS
//#define WITH_OPENINGS_PRIOR
#ifndef MY_OPENINGS_LABELS
#undef WITH_OPENINGS_PRIOR
#endif

#endif
