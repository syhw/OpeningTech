#ifndef ENUMS_NAME_TABLES
#define ENUMS_NAME_TABLES

#include <string>
#include <iostream>

#define TERRAN_X_UNITS \
    X(Terran_Marine, (const char*) "Terran_Marine") \
    X(Terran_Ghost,  (const char*) "Terran_Ghost") \
    X(Terran_Vulture,  (const char*) "Terran_Vulture") \
    X(Terran_Goliath,  (const char*) "Terran_Goliath") \
    X(Terran_Siege_Tank_Tank_Mode,  (const char*) "Terran_Siege_Tank_Tank_Mode") \
    X(Terran_SCV,  (const char*) "Terran_SCV") \
    X(Terran_Wraith,  (const char*) "Terran_Wraith") \
    X(Terran_Science_Vessel,  (const char*) "Terran_Science_Vessel") \
    X(Terran_Dropship,  (const char*) "Terran_Dropship") \
    X(Terran_Battlecruiser,  (const char*) "Terran_Battlecruiser") \
    X(Terran_Vulture_Spider_Mine,  (const char*) "Terran_Vulture_Spider_Mine") \
    X(Terran_Nuclear_Missile,  (const char*) "Terran_Nuclear_Missile") \
    X(Terran_Siege_Tank_Siege_Mode,  (const char*) "Terran_Siege_Tank_Siege_Mode") \
    X(Terran_Firebat,  (const char*) "Terran_Firebat") \
    X(Terran_Medic,  (const char*) "Terran_Medic") \
    X(Terran_Valkyrie,  (const char*) "Terran_Valkyrie")

#define X(a, b) a,
enum Terran_Units
{
    TERRAN_X_UNITS
};
#undef X

#define X(a, b) b,
const char* terran_units_name[] = 
{
    TERRAN_X_UNITS
};
#undef X

#define NB_TERRAN_UNITS 16

#define TERRAN_X_BUILDINGS \
    X(Terran_Command_Center, (const char*) "Terran_Command_Center") \
    X(Terran_Comsat_Station, (const char*) "Terran_Comsat_Station") \
    X(Terran_Nuclear_Silo, (const char*) "Terran_Nuclear_Silo") \
    X(Terran_Supply_Depot, (const char*) "Terran_Supply_Depot") \
    X(Terran_Supply_Depot2, (const char*) "Terran_Supply_Depot2") \
    X(Terran_Refinery, (const char*) "Terran_Refinery") \
    X(Terran_Barracks, (const char*) "Terran_Barracks") \
    X(Terran_Barracks2, (const char*) "Terran_Barracks2") \
    X(Terran_Academy, (const char*) "Terran_Academy") \
    X(Terran_Factory, (const char*) "Terran_Factory") \
    X(Terran_Starport, (const char*) "Terran_Starport") \
    X(Terran_Control_Tower, (const char*) "Terran_Control_Tower") \
    X(Terran_Science_Facility, (const char*) "Terran_Science_Facility") \
    X(Terran_Covert_Ops, (const char*) "Terran_Covert_Ops") \
    X(Terran_Physics_Lab, (const char*) "Terran_Physics_Lab") \
    X(Terran_Machine_Shop, (const char*) "Terran_Machine_Shop") \
    X(Terran_Engineering_Bay, (const char*) "Terran_Engineering_Bay") \
    X(Terran_Armory, (const char*) "Terran_Armory") \
    X(Terran_Missile_Turret, (const char*) "Terran_Missile_Turret") \
    X(Terran_Bunker, (const char*) "Terran_Bunker")

#define X(a, b) a,
enum Terran_Buildings
{
    TERRAN_X_BUILDINGS
};
#undef X

#define X(a, b) b,
const char* terran_buildings_name[] = 
{
    TERRAN_X_BUILDINGS
};
#undef X

#define NB_TERRAN_BUILDINGS 20 // 18 buildings + supply2 + rax2

#define PROTOSS_X_UNITS \
    X(Protoss_Corsair, (const char*) "Protoss_Corsair") \
    X(Protoss_Dark_Templar, (const char*) "Protoss_Dark_Templar") \
    X(Protoss_Dark_Archon, (const char*) "Protoss_Dark_Archon") \
    X(Protoss_Probe, (const char*) "Protoss_Probe") \
    X(Protoss_Zealot, (const char*) "Protoss_Zealot") \
    X(Protoss_Dragoon, (const char*) "Protoss_Dragoon") \
    X(Protoss_High_Templar, (const char*) "Protoss_High_Templar") \
    X(Protoss_Archon, (const char*) "Protoss_Archon") \
    X(Protoss_Shuttle, (const char*) "Protoss_Shuttle") \
    X(Protoss_Scout, (const char*) "Protoss_Scout") \
    X(Protoss_Arbiter, (const char*) "Protoss_Arbiter") \
    X(Protoss_Carrier, (const char*) "Protoss_Carrier") \
    X(Protoss_Interceptor, (const char*) "Protoss_Interceptor") \
    X(Protoss_Reaver, (const char*) "Protoss_Reaver") \
    X(Protoss_Observer, (const char*) "Protoss_Observer") \
    X(Protoss_Scarab, (const char*) "Protoss_Scarab")

#define X(a, b) a,
enum Protoss_Units
{
    PROTOSS_X_UNITS
};
#undef X

#define X(a, b) b,
const char* protoss_units_name[] =
{
    PROTOSS_X_UNITS
};
#undef X

#define NB_PROTOSS_UNITS 16

#define PROTOSS_X_BUILDINGS \
    X(Protoss_Nexus, (const char*) "Protoss_Nexus") \
    X(Protoss_Robotics_Facility, (const char*) "Protoss_Robotics_Facility") \
    X(Protoss_Pylon, (const char*) "Protoss_Pylon") \
    X(Protoss_Pylon2, (const char*) "Protoss_Pylon2") \
    X(Protoss_Assimilator, (const char*) "Protoss_Assimilator") \
    X(Protoss_Observatory, (const char*) "Protoss_Observatory") \
    X(Protoss_Gateway, (const char*) "Protoss_Gateway") \
    X(Protoss_Gateway2, (const char*) "Protoss_Gateway2") \
    X(Protoss_Photon_Cannon, (const char*) "Protoss_Photon_Cannon") \
    X(Protoss_Citadel_of_Adun, (const char*) "Protoss_Citadel_of_Adun") \
    X(Protoss_Cybernetics_Core, (const char*) "Protoss_Cybernetics_Core") \
    X(Protoss_Templar_Archives, (const char*) "Protoss_Templar_Archives") \
    X(Protoss_Forge, (const char*) "Protoss_Forge") \
    X(Protoss_Stargate, (const char*) "Protoss_Stargate") \
    X(Protoss_Fleet_Beacon, (const char*) "Protoss_Fleet_Beacon") \
    X(Protoss_Arbiter_Tribunal, (const char*) "Protoss_Arbiter_Tribunal") \
    X(Protoss_Robotics_Support_Bay, (const char*) "Protoss_Robotics_Support_Bay") \
    X(Protoss_Shield_Battery, (const char*) "Protoss_Shield_Battery")

#define X(a, b) a,
enum Protoss_Buildings
{
    PROTOSS_X_BUILDINGS
};
#undef X

#define X(a, b) b,
const char* protoss_buildings_name[] =
{
    PROTOSS_X_BUILDINGS
};
#undef X

#define NB_PROTOSS_BUILDINGS 18 // 16 buildings + pylon2 + gate2

#define ZERG_X_UNITS \
    X(Zerg_Larva, (const char*) "Zerg_Larva") \
    X(Zerg_Egg, (const char*) "Zerg_Egg") \
    X(Zerg_Zergling, (const char*) "Zerg_Zergling") \
    X(Zerg_Hydralisk, (const char*) "Zerg_Hydralisk") \
    X(Zerg_Ultralisk, (const char*) "Zerg_Ultralisk") \
    X(Zerg_Broodling, (const char*) "Zerg_Broodling") \
    X(Zerg_Drone, (const char*) "Zerg_Drone") \
    X(Zerg_Overlord, (const char*) "Zerg_Overlord") \
    X(Zerg_Mutalisk, (const char*) "Zerg_Mutalisk") \
    X(Zerg_Guardian, (const char*) "Zerg_Guardian") \
    X(Zerg_Queen, (const char*) "Zerg_Queen") \
    X(Zerg_Defiler, (const char*) "Zerg_Defiler") \
    X(Zerg_Scourge, (const char*) "Zerg_Scourge") \
    X(Zerg_Infested_Terran, (const char*) "Zerg_Infested_Terran") \
    X(Zerg_Cocoon, (const char*) "Zerg_Cocoon") \
    X(Zerg_Devourer, (const char*) "Zerg_Devourer") \
    X(Zerg_Lurker_Egg, (const char*) "Zerg_Lurker_Egg") \
    X(Zerg_Lurker, (const char*) "Zerg_Lurker")

#define X(a, b) a,
enum Zerg_Units
{
    ZERG_X_UNITS
};
#undef X

#define X(a, b) b,
const char* zerg_units_name[] =
{
    ZERG_X_UNITS
};
#undef X

#define NB_ZERG_UNITS 18

#define ZERG_X_BUILDINGS \
    X(Zerg_Infested_Command_Center, (const char*) "Zerg_Infested_Command_Center") \
    X(Zerg_Hatchery, (const char*) "Zerg_Hatchery") \
    X(Zerg_Lair, (const char*) "Zerg_Lair") \
    X(Zerg_Hive, (const char*) "Zerg_Hive") \
    X(Zerg_Nydus_Canal, (const char*) "Zerg_Nydus_Canal") \
    X(Zerg_Hydralisk_Den, (const char*) "Zerg_Hydralisk_Den") \
    X(Zerg_Defiler_Mound, (const char*) "Zerg_Defiler_Mound") \
    X(Zerg_Greater_Spire, (const char*) "Zerg_Greater_Spire") \
    X(Zerg_Queens_Nest, (const char*) "Zerg_Queens_Nest") \
    X(Zerg_Evolution_Chamber, (const char*) "Zerg_Evolution_Chamber") \
    X(Zerg_Ultralisk_Cavern, (const char*) "Zerg_Ultralisk_Cavern") \
    X(Zerg_Spire, (const char*) "Zerg_Spire") \
    X(Zerg_Spawning_Pool, (const char*) "Zerg_Spawning_Pool") \
    X(Zerg_Creep_Colony, (const char*) "Zerg_Creep_Colony") \
    X(Zerg_Spore_Colony, (const char*) "Zerg_Spore_Colony") \
    X(Zerg_Sunken_Colony, (const char*) "Zerg_Sunken_Colony") \
    X(Zerg_Extractor, (const char*) "Zerg_Extractor")\
    \
    X(Zerg_Building_Overlord, (const char*) "Zerg_Building_Overlord") \
    X(Zerg_Building_Overlord2, (const char*) "Zerg_Building_Overlord2")

#define X(a, b) a,
enum Zerg_Buildings
{
    ZERG_X_BUILDINGS
};
#undef X

#define X(a, b) b,
const char* zerg_buildings_name[] =
{
    ZERG_X_BUILDINGS
};
#undef X

#define NB_ZERG_BUILDINGS 19 // 17 buildings + 2 overlords

enum Spells
{
    Spell_Scanner_Sweep,
    Spell_Disruption_Web,
    Spell_Dark_Swarm
};

class Building
/// Not happy with this class
{
    friend std::ostream& operator <<(std::ostream& os, const Building& b);
    int _enumValue;
    int _tableSize;
    const char** _nameTable;
public:
    Building(Protoss_Buildings v)
        : _enumValue(v)
        , _nameTable(protoss_buildings_name)
        , _tableSize(NB_PROTOSS_BUILDINGS)
    {
    }
    Building(Terran_Buildings v)
        : _enumValue(v)
        , _nameTable(terran_buildings_name)
        , _tableSize(NB_TERRAN_BUILDINGS)
    {
    }
    Building(Zerg_Buildings v)
        : _enumValue(v)
        , _nameTable(zerg_buildings_name)
        , _tableSize(NB_ZERG_BUILDINGS)
    {
    }
    Building(const char* buildingName)
    {
        if (buildingName[0] == 'P')
        {
            _tableSize = NB_PROTOSS_BUILDINGS;
            _nameTable = protoss_buildings_name;
        } 
        else if (buildingName[0] == 'T')
        {
            _tableSize = NB_TERRAN_BUILDINGS;
            _nameTable = terran_buildings_name;
        } 
        else if (buildingName[0] == 'Z')
        {
            _tableSize = NB_ZERG_BUILDINGS;
            _nameTable = zerg_buildings_name;
        } 
        else
        {
            std::cout << 
                "ERROR: Building constructor failed to determine the race -> "
                << std::string(buildingName)
                << std::endl;
        }
        for (unsigned int i = 0; i < _tableSize; i++)
        {
            if (!strcmp(buildingName, _nameTable[i]))
            {
                _enumValue = i;
                return;
            }
        }
        std::cout << "ERROR: not found this building: "
            << "|" << std::string(buildingName) << "|"
            << std::endl;
    }
    std::ostream& operator <<(std::ostream& os) const
    {
        if (_enumValue < _tableSize)
            os << std::string(_nameTable[_enumValue]);
        else
            os << "ERROR: _enumValue too big: " << _enumValue;
        return os;
    }
    int getEnumValue() const
    {
        return _enumValue;
    }
};
inline std::ostream& operator <<(std::ostream& os, const Building& b)
{
    if (b._enumValue < b._tableSize)
        os << std::string(b._nameTable[b._enumValue]);
    else
        os << "ERROR: _enumValue too big: " << b._enumValue;
    return os;
}

#endif
