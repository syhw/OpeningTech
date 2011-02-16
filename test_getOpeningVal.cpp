#include "replays.h"

using namespace std;

int main(int argc, const char *argv[])
{
    string input;
    while(cin)
    {
        getline(cin, input);
        if (input.empty())
            return 0;
        cout << pruneOpeningVal(input);
    }
    return 0;
}
