#ifndef X_VALUES
#define X_VALUES
#include <set>
#include <vector>
#include "enums_name_tables.h"

std::vector<std::set<Terran_Buildings> > generate_terran_X_values()
{
    std::vector<std::set<Terran_Buildings> > ret_vector;
    
    //int cc = 0;
    int cs = 0;
    int ns = 0;
    int supply = 0;
    int ref = 0;
    int rax = 0;
    int academy = 0;
    int facto = 0;
    int starport = 0;
    int control_tower = 0;
    int science = 0;
    int covert = 0;
    int physics = 0;
    int machine_shop = 0;
    int ebay = 0;
    int armory = 0;
    int missile_turret = 0;
    int bunker = 0;

    //for(; cc < 1; cc++) {
    for(; supply < 1; supply++) {
    for(; ref < 1; ref++) {
    for(; rax < 1; rax++) {
        for(; bunker < 1 && rax; bunker++) {
        for(; ebay < 1 && rax; ebay++) {
            for(; missile_turret < 1 && ebay; missile_turret++) {
        for(; academy < 1 && rax; academy++) {
            for(; cs < 1 && academy; cs++) {
        for(; facto < 1 && rax; facto++) {
            for(; armory < 1 && facto; armory++) {
            for(; machine_shop < 1 && facto; machine_shop++) {
            for(; starport < 1 && facto; starport++) {
                for(; control_tower < 1 && starport; control_tower++) {
                for(; science < 1 && starport; science++) { // && facto ;)
                    for(; covert < 1 && science; covert++) {
                        for(; ns < 1 && covert; ns++) {
                    for(; physics < 1 && science; physics++) {

        std::set<Terran_Buildings> tmp_set;
        tmp_set.insert((Terran_Buildings)0); // CC
        tmp_set.insert((Terran_Buildings)(cs*Terran_Comsat_Station));
        tmp_set.insert((Terran_Buildings)(ns*Terran_Nuclear_Silo)); 
        tmp_set.insert((Terran_Buildings)(supply*Terran_Supply_Depot)); 
        tmp_set.insert((Terran_Buildings)(ref*Terran_Refinery)); 
        tmp_set.insert((Terran_Buildings)(rax*Terran_Barracks)); 
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

    }}}}}}}}}}}}}}}}}
    return ret_vector;
}

std::vector<std::set<Protoss_Buildings> > generate_protoss_X_values()
{
    std::vector<std::set<Protoss_Buildings> > ret_vector;
    
    //int nexus = 0;
    int robo = 0;
    int pylon = 0;
    int assim = 0;
    int obs = 0;
    int gate = 0;
    int photon = 0;
    int citadel = 0;
    int cyber = 0;
    int archives = 0;
    int forge = 0;
    int stargate = 0;
    int beacon = 0;
    int tribunal = 0;
    int bay = 0;
    int shield = 0; 

    //for(; nexus < 1; nexus++) {
    for(; pylon < 1; pylon++) {
    for(; assim < 1; assim++) {
    for(; gate < 1; gate++) {
        for(; shield < 1 && gate; shield++) {
        for(; cyber < 1 && gate; cyber++) {
            for(; citadel < 1 && cyber; citadel++) {
                for(; archives < 1 && citadel; archives++) {
            for(; robo < 1 && cyber; robo++) {
                for(; bay < 1 && robo; bay++) {
                for(; obs < 1 && robo; obs++) {
            for(; stargate < 1 && cyber; stargate++) {
                for(; beacon < 1 && stargate; beacon++) {
                for(; tribunal < 1 && stargate; tribunal++) {
    for(; forge < 1; forge++) {
        for(; photon < 1 && forge; photon++) {

    std::set<Protoss_Buildings> tmp_set;
    tmp_set.insert((Protoss_Buildings)0); // Nexus
    tmp_set.insert((Protoss_Buildings)(robo*Protoss_Robotics_Facility)); 
    tmp_set.insert((Protoss_Buildings)(pylon*Protoss_Pylon)); 
    tmp_set.insert((Protoss_Buildings)(assim*Protoss_Assimilator)); 
    tmp_set.insert((Protoss_Buildings)(obs*Protoss_Observatory)); 
    tmp_set.insert((Protoss_Buildings)(gate*Protoss_Gateway)); 
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

    }}}}}}}}}}}}}}}
    return ret_vector;
}

std::vector<std::set<Zerg_Buildings> > generate_zerg_X_values()
{
    std::vector<std::set<Zerg_Buildings> > ret_vector;
    
    //int hatch = 0;
    int infestedcc = 0;
    int lair = 0;
    int hive = 0;
    int nydus = 0;
    int hydraden = 0;
    int defiler_mound = 0;
    int greater_spire = 0;
    int nest = 0;
    int evo = 0;
    int ultra_cavern = 0;
    int spire = 0;
    int pool = 0;
    int creep_colony = 0;
    int spore = 0;
    int sunken = 0;
    int extractor = 0;

    //for(; hatch < 1; hatch++) {
    for(; extractor < 1; extractor++) {
    for(; creep_colony < 1; creep_colony++) {
    for(; pool < 1; pool++) {
        for(; sunken < 1 && pool; sunken++) { // && creep_colony
        for(; hydraden < 1 && pool; hydraden++) {
        for(; lair < 1 && pool; lair++) {
            for(; spire < 1 && lair; spire++) {
            for(; nest < 1 && lair; nest++) {
                for(; infestedcc < 1 && nest; infestedcc++) {
                for(; hive < 1 && nest; hive++) {
                    for(; nydus < 1 && hive; nydus++) {
                    for(; defiler_mound < 1 && hive; defiler_mound++) {
                    for(; greater_spire < 1 && hive && spire; greater_spire++) { 
                        // (greate_spire=1) => (spire=1)
                    for(; ultra_cavern < 1 && hive; ultra_cavern++) {
    for(; evo < 1; evo++) {
        for(; spore < 1 && evo; spore++) { // && creep_colony

        std::set<Zerg_Buildings> tmp_set;
        tmp_set.insert((Zerg_Buildings)0); // Hatch
        tmp_set.insert((Zerg_Buildings)(infestedcc*Zerg_Infested_Command_Center)); 
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

    }}}}}}}}}}}}}}}}
    return ret_vector;
}

#endif
