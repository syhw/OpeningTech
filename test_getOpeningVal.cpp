#include "replays.h"

using namespace std;

int main(int argc, const char *argv[])
{
    string input;
    while(cin)
    {
        getline(cin, input);
        cout << getOpeningVal(input);
    }
    return 0;
}
