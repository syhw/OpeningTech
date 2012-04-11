#include <iostream>
#include <map>
#include "x_values.h"
#include "replays.h"

using namespace std;

void mapPrint(const multimap<int, Building>& m)
{
    for (multimap<int, Building>::const_iterator it = m.begin();
            it != m.end(); ++it)
    {
        cout << "Time: " << it->first << ", Building: " 
            << it->second << endl;
    }
}

int main(int argc, const char *argv[])
{
    string input;
    while (cin)
    {
        getline(cin, input);
        string tmpOpening = pruneOpeningVal(input);
        if (tmpOpening != "")
        {
            multimap<int, Building> tmpBuildings;
            getBuildings(input, tmpBuildings);
            mapPrint(tmpBuildings);
        }
    }
    return 0;
}
