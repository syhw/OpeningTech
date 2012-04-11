#ifndef X_VALUES
#define X_VALUES
#include "parameters.h"
#include <set>
#include <vector>
#include <algorithm>
#include "enums_name_tables.h"

/// Copyright Gabriel Synnaeve 2011
/// This code is under 3-clauses (new) BSD License

#ifdef GENERATE_X_VALUES

#include <iostream>

std::vector<std::set<Terran_Buildings> > get_terran_X_values()
{
    std::vector<std::set<Terran_Buildings> > ret_vector;
    
    for(int expand = 0; expand <= 1; expand++) {
    for(int supply = 0; supply <= 1; supply++) {
    for(int ref = 0; ref <= 1; ref++) {
    for(int rax = 0; rax <= 1; rax++) {
    for(int rax2 = 0; rax2 <= 1; rax2++) {
        for(int bunker = 0; bunker == 0 || (bunker <= 1 && rax); bunker++) {
        for(int ebay = 0; ebay == 0 || (ebay <= 1 && rax); ebay++) {
            for(int missile_turret = 0; missile_turret == 0 || (missile_turret <= 1 && ebay); missile_turret++) {
        for(int academy = 0; academy == 0 || (academy <= 1 && rax && ref); academy++) { // && supply ?
            for(int cs = 0 ; cs == 0 || (cs <= 1 && academy); cs++) {
        for(int facto = 0; facto == 0 || (facto <= 1 && rax && ref); facto++) { // && supply ?
            for(int armory = 0; armory == 0 || (armory <= 1 && facto); armory++) {
            for(int machine_shop = 0; machine_shop == 0 || (machine_shop <= 1 && facto); machine_shop++) {
            for(int starport = 0; starport == 0 || (starport <= 1 && facto); starport++) {
                for(int control_tower = 0; control_tower == 0 || (control_tower <= 1 && starport); control_tower++) {
                for(int science = 0; science == 0 || (science <= 1 && starport); science++) { // && facto ;)
                    for(int covert = 0; covert == 0 || (covert <= 1 && science); covert++) {
                        for(int ns = 0; ns == 0 || (ns <= 1 && covert); ns++) {
                    for(int physics = 0; physics == 0 || (physics <= 1 && science); physics++) {

        std::set<Terran_Buildings> tmp_set;
        tmp_set.insert((Terran_Buildings)0); // CC
        tmp_set.insert((Terran_Buildings)(expand*Terran_Expansion));
        tmp_set.insert((Terran_Buildings)(cs*Terran_Comsat_Station));
        tmp_set.insert((Terran_Buildings)(ns*Terran_Nuclear_Silo)); 
        tmp_set.insert((Terran_Buildings)(supply*Terran_Supply_Depot)); 
        tmp_set.insert((Terran_Buildings)(ref*Terran_Refinery)); 
        tmp_set.insert((Terran_Buildings)(rax*Terran_Barracks)); 
        tmp_set.insert((Terran_Buildings)(rax2*Terran_Barracks2)); 
        tmp_set.insert((Terran_Buildings)(academy*Terran_Academy)); 
        tmp_set.insert((Terran_Buildings)(facto*Terran_Factory)); 
        tmp_set.insert((Terran_Buildings)(starport*Terran_Starport)); 
        tmp_set.insert((Terran_Buildings)(control_tower*Terran_Control_Tower)); 
        tmp_set.insert((Terran_Buildings)(science*Terran_Science_Facility)); 
        tmp_set.insert((Terran_Buildings)(covert*Terran_Covert_Ops)); 
        tmp_set.insert((Terran_Buildings)(physics*Terran_Physics_Lab)); 
        tmp_set.insert((Terran_Buildings)(machine_shop*Terran_Machine_Shop)); 
        tmp_set.insert((Terran_Buildings)(ebay*Terran_Engineering_Bay)); 
        tmp_set.insert((Terran_Buildings)(armory*Terran_Armory)); 
        tmp_set.insert((Terran_Buildings)(missile_turret*Terran_Missile_Turret)); 
        tmp_set.insert((Terran_Buildings)(bunker*Terran_Bunker)); 
        ret_vector.push_back(tmp_set);

    }}}}}}}}}}}}}}}}}}}
    return ret_vector;
}

std::vector<std::set<Protoss_Buildings> > get_protoss_X_values()
{
    std::vector<std::set<Protoss_Buildings> > ret_vector;

    for(int expand = 0; expand <= 1; expand++) {
    for(int pylon = 0; pylon <= 1; pylon++) {
    for(int pylon2 = 0; pylon2 <= 1; pylon2++) {
    for(int assim = 0; assim <= 1; assim++) {
    for(int gate = 0; gate == 0 || (gate == 1 && pylon); gate++) {
    for(int gate2 = 0; gate2 == 0 || (gate2 == 1 && pylon); gate2++) {
        for(int shield = 0; shield == 0 || (shield == 1 && gate); shield++) {
        for(int cyber = 0; cyber == 0 || (cyber == 1 && gate); cyber++) {
            for(int citadel = 0; citadel == 0 || (citadel == 1 && cyber && assim); citadel++) {
                for(int archives = 0; archives == 0 || (archives == 1 && citadel); archives++) {
            for(int robo = 0; robo == 0 || (robo == 1 && cyber && assim); robo++) {
                for(int bay = 0; bay == 0 || (bay == 1 && robo); bay++) {
                for(int obs = 0; obs == 0 || (obs == 1 && robo); obs++) {
            for(int stargate = 0; stargate == 0 || (stargate == 1 && cyber && assim); stargate++) {
                for(int beacon = 0; beacon == 0 || (beacon == 1 && stargate); beacon++) {
                for(int tribunal = 0; tribunal == 0 || (tribunal == 1 && stargate); tribunal++) {
    for(int forge = 0; forge <= 1; forge++) {
        for(int photon = 0; photon == 0 || (photon == 1 && forge); photon++) {

    std::set<Protoss_Buildings> tmp_set;
    tmp_set.insert((Protoss_Buildings)0); // Nexus
    tmp_set.insert((Protoss_Buildings)(expand*Protoss_Expansion)); 
    tmp_set.insert((Protoss_Buildings)(robo*Protoss_Robotics_Facility)); 
    tmp_set.insert((Protoss_Buildings)(pylon*Protoss_Pylon)); 
    tmp_set.insert((Protoss_Buildings)(pylon2*Protoss_Pylon2)); 
    tmp_set.insert((Protoss_Buildings)(assim*Protoss_Assimilator)); 
    tmp_set.insert((Protoss_Buildings)(obs*Protoss_Observatory)); 
    tmp_set.insert((Protoss_Buildings)(gate*Protoss_Gateway)); 
    tmp_set.insert((Protoss_Buildings)(gate2*Protoss_Gateway2)); 
    tmp_set.insert((Protoss_Buildings)(photon*Protoss_Photon_Cannon)); 
    tmp_set.insert((Protoss_Buildings)(citadel*Protoss_Citadel_of_Adun)); 
    tmp_set.insert((Protoss_Buildings)(cyber*Protoss_Cybernetics_Core)); 
    tmp_set.insert((Protoss_Buildings)(archives*Protoss_Templar_Archives)); 
    tmp_set.insert((Protoss_Buildings)(forge*Protoss_Forge)); 
    tmp_set.insert((Protoss_Buildings)(stargate*Protoss_Stargate)); 
    tmp_set.insert((Protoss_Buildings)(beacon*Protoss_Fleet_Beacon)); 
    tmp_set.insert((Protoss_Buildings)(tribunal*Protoss_Arbiter_Tribunal)); 
    tmp_set.insert((Protoss_Buildings)(bay*Protoss_Robotics_Support_Bay)); 
    tmp_set.insert((Protoss_Buildings)(shield*Protoss_Shield_Battery)); 
    ret_vector.push_back(tmp_set);

    }}}}}}}}}}}}}}}}}}
    return ret_vector;
}

std::vector<std::set<Zerg_Buildings> > get_zerg_X_values()
{
    std::vector<std::set<Zerg_Buildings> > ret_vector;
    
    for(int expand = 0; expand <= 1; expand++) {
    for(int expand2 = 0; expand2 <= 1; expand2++) {
    for(int overlord = 0; overlord <= 1; overlord++) {
    for(int extractor = 0; extractor <= 1; extractor++) {
    for(int creep_colony = 0; creep_colony <= 1; creep_colony++) {
    for(int pool = 0; pool <= 1; pool++) {
        for(int sunken = 0; sunken == 0 || (sunken == 1 && pool); sunken++) { // && creep_colony
        for(int hydraden = 0; hydraden == 0 || (hydraden == 1 && pool && extractor); hydraden++) {
        for(int lair = 0; lair == 0 || (lair == 1 && pool && extractor); lair++) {
            for(int spire = 0; spire == 0 || (spire == 1 && lair); spire++) {
            for(int nest = 0; nest == 0 || (nest == 1 && lair); nest++) {
                //for(int infestedcc = 0; infestedcc == 0 || (infestedcc == 1 && nest); infestedcc++) {
                for(int hive = 0; hive == 0 || (hive == 1 && nest); hive++) {
                    for(int nydus = 0; nydus == 0 || (nydus == 1 && hive); nydus++) {
                    for(int defiler_mound = 0; defiler_mound == 0 || (defiler_mound == 1 && hive); defiler_mound++) {
                    for(int greater_spire = 0; greater_spire == 0 || (greater_spire == 1 && hive && spire); greater_spire++) { 
                        // (greate_spire=1) => (spire=1)
                    for(int ultra_cavern = 0; ultra_cavern == 0 || (ultra_cavern == 1 && hive); ultra_cavern++) {
    for(int evo = 0; evo <= 1; evo++) {
        for(int spore = 0; spore == 0 || (spore == 1 && evo); spore++) { // && creep_colony

        std::set<Zerg_Buildings> tmp_set;
        tmp_set.insert((Zerg_Buildings)0); // Hatch
        //tmp_set.insert((Zerg_Buildings)(infestedcc*Zerg_Infested_Command_Center)); 
        tmp_set.insert((Zerg_Buildings)(expand*Zerg_Expansion)); 
        tmp_set.insert((Zerg_Buildings)(expand2*Zerg_Expansion2)); 
        tmp_set.insert((Zerg_Buildings)(overlord*Zerg_Building_Overlord)); 
        tmp_set.insert((Zerg_Buildings)(lair*Zerg_Lair)); 
        tmp_set.insert((Zerg_Buildings)(hive*Zerg_Hive)); 
        tmp_set.insert((Zerg_Buildings)(nydus*Zerg_Nydus_Canal)); 
        tmp_set.insert((Zerg_Buildings)(hydraden*Zerg_Hydralisk_Den)); 
        tmp_set.insert((Zerg_Buildings)(defiler_mound*Zerg_Defiler_Mound)); 
        tmp_set.insert((Zerg_Buildings)(greater_spire*Zerg_Greater_Spire)); 
        tmp_set.insert((Zerg_Buildings)(nest*Zerg_Queens_Nest)); 
        tmp_set.insert((Zerg_Buildings)(evo*Zerg_Evolution_Chamber)); 
        tmp_set.insert((Zerg_Buildings)(ultra_cavern*Zerg_Ultralisk_Cavern)); 
        tmp_set.insert((Zerg_Buildings)(spire*Zerg_Spire)); 
        tmp_set.insert((Zerg_Buildings)(pool*Zerg_Spawning_Pool)); 
        tmp_set.insert((Zerg_Buildings)(creep_colony*Zerg_Creep_Colony)); 
        tmp_set.insert((Zerg_Buildings)(spore*Zerg_Spore_Colony)); 
        tmp_set.insert((Zerg_Buildings)(sunken*Zerg_Sunken_Colony)); 
        tmp_set.insert((Zerg_Buildings)(extractor*Zerg_Extractor)); 
        ret_vector.push_back(tmp_set);

    }}}}}}}}}}}}}}}}}}//}
    return ret_vector;
}

#else

#include <fstream>
#include <map>
#include "replays.h"

std::string pruneOpeningVal(std::string& input);

struct tech_trees
{
    std::vector<std::set<int> > vector_X; // build tree (BT) as sets
    std::vector<std::vector<int> > set_distances_X; // distances between BT
    tech_trees(std::ifstream& fin)
    {
        // Fill vector_X
        std::set<std::set<int> > ret_set; // unordered_set
        std::string line;
        while (getline(fin, line))
        {
            pruneOpeningVal(line);
            std::multimap<int, Building> buildings;
            getBuildings(line, buildings, 0); // 0 for cutoffseconds
            buildings.erase(0); // key == 0 i.e. buildings not constructed
            std::set<int> tmpSet;
            tmpSet.insert(0); // first Nexus/CC/Hatch exists
            for (std::multimap<int, Building>::const_iterator it
                    = buildings.begin();
                    it != buildings.end(); ++it)
            {
                if (it->first > LEARN_TIME_LIMIT)
                    break;
                tmpSet.insert(it->second.getEnumValue());
                ret_set.insert(tmpSet);
            }
        }
        vector_X.reserve(ret_set.size());
        std::copy(ret_set.begin(), ret_set.end(),
                std::back_inserter(vector_X));
        // Fill set_distances_X
        for (unsigned int i = 0; i < vector_X.size(); ++i)
        {
            std::vector<int> tmp;
            for (unsigned int j = 0; j < vector_X.size(); ++j) // size/2+1
            {
                tmp.push_back(set_distance(i, j));
            }
            set_distances_X.push_back(tmp);
        }
    }
    int set_distance(unsigned int i, unsigned int j)
    {
        std::vector<int> symdiff;
        //std::set_symmetric_difference(vector_X[i].begin(), vector_X[i].end(),
        //        vector_X[j].begin(), vector_X[j].end(), symdiff.begin());
        for (std::set<int>::const_iterator it = vector_X[i].begin();
                it != vector_X[i].end(); ++it)
        {
            if (!vector_X[j].count(*it))
                symdiff.push_back(*it);
        }
        for (std::set<int>::const_iterator jt = vector_X[j].begin();
                jt != vector_X[j].end(); ++jt)
        {
            if (!vector_X[i].count(*jt))
                symdiff.push_back(*jt);
        }
        return symdiff.size();
    }
    tech_trees() {}
};

#endif

/// dumbest function evar
/// (search in O(n) in a vector)
/// TODO CHANGE
int get_X_indice(const std::set<int>& X,
        const std::vector<std::set<int> >& all_X)
{
    for (unsigned int i = 0; i < all_X.size(); ++i)
    {
        if (all_X[i] == X)
            return i;
    }
    std::cout << "ERROR: X not found in all existing X" << std::endl;
    return -1;
}

#endif
