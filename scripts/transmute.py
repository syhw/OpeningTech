#!/opt/local/bin/python

"""
3-clauses BSD licence, 
Copyright 2010-2011 Gabriel Synnaeve

Script that transforms an arff file of SC:BW replays into a txt file 
with each line corresponding to a game.

Usage:
    python transmute.py input.arff output.txt
"""

import sys

input = open(sys.argv[1], 'r')
output = open(sys.argv[2], 'w')
race = 0

def adapt(list):
    if race == 1:
        for i in range(len(list)):
            if list[i] == 'ProtossPylon':
                list[i] = 'Protoss_Pylon'
                continue
            if list[i] == 'ProtossSecondPylon':
                list[i] = 'Protoss_Pylon2'
                continue
            if list[i] == 'ProtossFirstExpansion':
                list[i] = 'Protoss_Expansion'
                continue
            if list[i] == 'ProtossRoboBay':
                list[i] = 'Protoss_Robotics_Facility'
                continue
            if list[i] == 'ProtossFirstGas':
                list[i] = 'Protoss_Assimilator'
                continue
            if list[i] == 'ProtossObservatory':
                list[i] = 'Protoss_Observatory'
                continue
            if list[i] == 'ProtossGateway':
                list[i] = 'Protoss_Gateway'
                continue
            if list[i] == 'ProtossSecondGatway': # :)
                list[i] = 'Protoss_Gateway2'
                continue
            if list[i] == 'ProtossCannon':
                list[i] = 'Protoss_Photon_Cannon'
                continue
            if list[i] == 'ProtossCitadel':
                list[i] = 'Protoss_Citadel_of_Adun'
                continue
            if list[i] == 'ProtossCore':
                list[i] = 'Protoss_Cybernetics_Core'
                continue
            if list[i] == 'ProtossArchives':
                list[i] = 'Protoss_Templar_Archives'
                continue
            if list[i] == 'ProtossForge':
                list[i] = 'Protoss_Forge'
                continue
            if list[i] == 'ProtossStargate':
                list[i] = 'Protoss_Stargate'
                continue
            if list[i] == 'ProtossFleetBeason': # lol
                list[i] = 'Protoss_Fleet_Beacon'
                continue
            if list[i] == 'ProtossTribunal':
                list[i] = 'Protoss_Arbiter_Tribunal'
                continue
            if list[i] == 'ProtossRoboSupport':
                list[i] = 'Protoss_Robotics_Support_Bay'
                continue
            #if list[i] == '':
            #   list[i] = 'Protoss_Shield_Battery'
            #   continue
            if list[i] == 'midBuild':
                list[i] = 'Protoss_Opening'
    elif race == 2:
        for i in range(len(list)):
            if list[i] == 'TerranDepot':
                list[i] = 'Terran_Supply_Depot'
                continue
            if list[i] == 'TerranExpansion':
                list[i] = 'Terran_Expansion'
                continue
            if list[i] == 'TerranBarracks':
                list[i] = 'Terran_Barracks'
                continue
            if list[i] == 'TerranSecondBarracks':
                list[i] = 'Terran_Barracks2'
                continue
            if list[i] == 'TerranGas':
                list[i] = 'Terran_Refinery'
                continue
            if list[i] == 'TerranComsat':
                list[i] = 'Terran_Comsat_Station'
                continue
            if list[i] == 'TerranAcademy':
                list[i] = 'Terran_Academy'
                continue
            if list[i] == 'TerranFactory':
                list[i] = 'Terran_Factory'
                continue
            if list[i] == 'TerranStarport':
                list[i] = 'Terran_Starport'
                continue
            if list[i] == 'TerranControlTower':
                list[i] = 'Terran_Control_Tower'
                continue
            if list[i] == 'TerranScienceFacility':
                list[i] = 'Terran_Science_Facility'
                continue
            if list[i] == 'TerranMachineShop':
                list[i] = 'Terran_Machine_Shop'
                continue
            if list[i] == 'TerranEbay':
                list[i] = 'Terran_Engineering_Bay'
                continue
            if list[i] == 'TerranArmory':
                list[i] = 'Terran_Armory'
                continue
            if list[i] == 'TerranTurret':
                list[i] = 'Terran_Missile_Turret'
                continue
            if list[i] == 'TerranBunker':
                list[i] = 'Terran_Bunker'
                continue
            if list[i] == 'midBuild':
                list[i] = 'Terran_Opening'
    elif race == 3:
        for i in range(len(list)):
            if list[i] == 'ZergSecondHatch':
                list[i] = 'Zerg_Expansion'
                continue
            if list[i] == 'ZergThirdHatch':
                list[i] = 'Zerg_Expansion2'
                continue
            if list[i] == 'ZergLair':
                list[i] = 'Zerg_Lair'
                continue
            if list[i] == 'ZergHive':
                list[i] = 'Zerg_Hive'
                continue
            if list[i] == 'ZergHydraDen':
                list[i] = 'Zerg_Hydralisk_Den'
                continue
            if list[i] == 'ZergDefilerMound':
                list[i] = 'Zerg_Defiler_Mound'
                continue
            if list[i] == 'ZergGreaterSpire':
                list[i] = 'Zerg_Greater_Spire'
                continue
            if list[i] == 'ZergQueenNest':
                list[i] = 'Zerg_Queens_Nest'
                continue
            if list[i] == 'ZergevoDen':
                list[i] = 'Zerg_Evolution_Chamber'
                continue
            if list[i] == 'ZergUltraliskCavern':
                list[i] = 'Zerg_Ultralisk_Cavern'
                continue
            if list[i] == 'ZergSpire':
                list[i] = 'Zerg_Spire'
                continue
            if list[i] == 'ZergPool':
                list[i] = 'Zerg_Spawning_Pool'
                continue
            if list[i] == 'ZergCreep':
                list[i] = 'Zerg_Creep_Colony'
                continue
            if list[i] == 'ZergSpore':
                list[i] = 'Zerg_Spore_Colony'
                continue
            if list[i] == 'ZergSunken':
                list[i] = 'Zerg_Sunken_Colony'
                continue
            if list[i] == 'ZergGas':
                list[i] = 'Zerg_Extractor'
                continue
            if list[i] == 'ZergOverlord':
                list[i] = 'Zerg_Building_Overlord'
                continue
            if list[i] == 'midBuild':
                list[i] = 'Zerg_Opening'

lookup = []
games = []
data = 0
for line in input:
    if race == 0:
        if "Protoss" in line:
            race = 1
        elif "Terran" in line:
            race = 2
        elif "Zerg" in line:
            race = 3
    if data:
        game = {}
        l = line.split(',')
        for i in range(len(l)):
            game[lookup[i]] = l[i].rstrip('\n\r')
        games.append(game)
    else:
        if '@ATTRIBUTE' in line or '@attribute' in line:
            lookup.append(line.split(' ')[1])
        if '@DATA' in line or '@data' in line:
            adapt(lookup)
            data = 1
input.close()

for game in games:
    for (k,v) in game.iteritems():
        if '_' in k:
            if "Opening" in k:
                output.write("%s %s; " % (k, v))
            else:
                val = int(v)/24
                if val > 1079:
                    val = 0
                if val < 0:
                    val = 0
                output.write("%s %s; " % (k, str(val)))
    output.write('\n')
output.close()
