#!/opt/local/bin/python
import sys

"""
Copyright 2011 Gabriel Synnaeve
License: Python Software Foundation License (PSFL, BSD-like, GPL compatible)
http://docs.python.org/license.html

Transforms lord martin replay rips (one big file format) into:
    - one arff file: attributes are buildings/units/upgrades for the 2 players
    1 line = 1 game
    - one text file: one line for one player's action in 
    "Race_Building/UnitUpgrade Time;" format, 2 lines = 1 game

Winner/Loser = -1 is for unspecified/unfound/draw, 
               0/1 player number (incremental) for arff format
               0/1 false/true for txt format

Options:
    --generate-attributes only outputs attributes (buildings/upgrades/units)
    --multiple-buildings use second/third/fourth for some buildings...
    --generate-text output in txt (and not arff) format
"""

# pyt arff.py all.lmr [--generate-attributes] [--multiple-buildings]"
def usage():
    print "Usage is:"
    print "pyt arff.py FILE.lmr [--generate-attributes] [--multiple-buildings] [--generate-text]"

if len(sys.argv) < 2:
    usage()

write_arffs = True
multiple_buildings = True # second (and more) Gateways/Barracks/Hatches...
generate_attributes = False
if '--generate-attributes' in sys.argv:
    generate_attributes = True
if '--multiple-buildings' in sys.argv:
    multiple_buildings = True
if '--generate-text' in sys.argv:
    write_arffs = False

if write_arffs:
    pvp = open('2p'+sys.argv[1].split('.')[0]+'pvp.arff', 'w')
    pvt = open('2p'+sys.argv[1].split('.')[0]+'pvt.arff', 'w')
    pvz = open('2p'+sys.argv[1].split('.')[0]+'pvz.arff', 'w')
    tvt = open('2p'+sys.argv[1].split('.')[0]+'tvt.arff', 'w')
    tvz = open('2p'+sys.argv[1].split('.')[0]+'tvz.arff', 'w')
    zvz = open('2p'+sys.argv[1].split('.')[0]+'zvz.arff', 'w')
else:
    pvp = open('2p'+sys.argv[1].split('.')[0]+'pvp.txt', 'w')
    pvt = open('2p'+sys.argv[1].split('.')[0]+'pvt.txt', 'w')
    pvz = open('2p'+sys.argv[1].split('.')[0]+'pvz.txt', 'w')
    tvt = open('2p'+sys.argv[1].split('.')[0]+'tvt.txt', 'w')
    tvz = open('2p'+sys.argv[1].split('.')[0]+'tvz.txt', 'w')
    zvz = open('2p'+sys.argv[1].split('.')[0]+'zvz.txt', 'w')


s = set()
protoss = set()
terran = set()
zerg = set()

p = [{'game':{}}, {'game':{}}, -1]
nbp = 0

def extract_attribute(t):
    """
    Extract the building/unit/upgrade name and append the Race_ in front
    Works with a closure on p (players list of dict)
    """
    ret = t[len(t)-1]
    if 'player' in ret: # prune/filter player quit/disconnect
        return ''
    ret = ret.replace(' ', '_')
    if '(' in ret:
        ret = ret.split('(')[1].replace(')', '')
    ### Because upgrades sometimes are Protoss_Air_...
    if not 'Protoss_' in ret\
            and not 'Terran_' in ret \
            and not 'Zerg_' in ret:
        tr = p[int(t[2])]['race']
        if tr == 'P':
            ret = 'Protoss_' + ret
        elif tr == 'T':
            ret = 'Terran_' + ret
        elif tr == 'Z':
            ret = 'Zerg_' + ret
    return ret

def write_to(t):
    mu_ok = False
    for i in [0, 1]:
        if not mu_ok:
            if t[i]['race'] == 'P':
                p1 = i
                sp1 = protoss
                if t[1-i]['race'] == 'P':
                    to_write = pvp
                    p2 = 1-i
                    sp2 = protoss
                    mu_ok = True
                if t[1-i]['race'] == 'T':
                    to_write = pvt
                    p2 = 1-i
                    sp2 = terran
                    mu_ok = True
                if t[1-i]['race'] == 'Z':
                    to_write = pvz
                    p2 = 1-i
                    sp2 = zerg
                    mu_ok = True
    for i in [0, 1]:
        if not mu_ok:
            if t[i]['race'] == 'T':
                p1 = i
                sp1 = terran
                if t[1-i]['race'] == 'T':
                    to_write = tvt
                    p2 = 1-i
                    sp2 = terran
                    mu_ok = True
                if t[1-i]['race'] == 'Z':
                    to_write = tvz
                    p2 = 1-i
                    sp2 = zerg
                    mu_ok = True
    if not mu_ok:
        if t[0]['race'] == 'Z' and t[1]['race'] == 'Z':
            to_write = zvz
            p1 = 0
            p2 = 1
            sp1 = zerg
            sp2 = zerg
            mu_ok = True
    if write_arffs:
        if t[2] != -1:
            to_write.write(str(p1 - t[2]))
        else: # DRAW
            to_write.write('-1')
        for a in sp1:
            if a in t[p1]['game']:
                to_write.write(',' + t[p1]['game'][a])
            else:
                to_write.write(',0')
        for a in sp2:
            if a in t[p2]['game']:
                to_write.write(',' + t[p2]['game'][a])
            else:
                to_write.write(',0')
        to_write.write('\n')
    else:
        if 'winner' in t[p1]:
            to_write.write("Winner " + str(t[p1]['winner']) + '; ')
        else:
            to_write.write("Winner -1; ")
        for a in sp1:
            if a in t[p1]['game']:
                to_write.write(a + ' ' + str(t[p1]['game'][a]) + '; ')
            else:
                to_write.write(a + ' 0; ')
        to_write.write('\n')
        if 'winner' in t[p2]:
            to_write.write("Winner " + str(t[p2]['winner']) + '; ')
        else:
            to_write.write("Winner -1;")
        for a in sp2:
            if a in t[p2]['game']:
                to_write.write(a + ' ' + str(t[p2]['game'][a]) + '; ')
            else:
                to_write.write(a + ' 0; ')
        to_write.write('\n')

for step in [0,1]:
    f = open(sys.argv[1], 'r')
    for line in f:
        if line[0] == '[': # new replay name, write the old
            if p[0]['game'] != {} or p[1]['game'] != {}:
                write_to(p)
            # re-init for next game/replay
            p = [{'game':{}}, {'game':{}}, -1]
            nbp = 0
        if line[0] in '0123456789': # players' actions
            if nbp != 2:
                pass # SKIP (we are only interested in duel / 1vs1) 
            else:
                line = line.rstrip(' \n\r').split(',')
                tmp = extract_attribute(line)
                if tmp != '' and tmp not in s:
                    s.add(tmp)
                if step == 1 :
                    if 'player quit' in line[len(line) - 1]\
                            and not 'winner' in p[int(line[2])]:
                        p[int(line[2])]['winner'] = 0
                        p[1 - int(line[2])]['winner'] = 1
                        p[2] = 1-int(line[2])
                    else:
                        player = int(line[2])
                        if not tmp in p[player]['game']:
                            p[player]['game'][tmp] = line[0]
                        else: # search for Nexus2, Pylon2-3, Barracks2-4...
                            for i in range(2,5):
                                totest = tmp+str(i)
                                if totest in s\
                                        and totest not in p[player]['game']:
                                    p[player]['game'][totest] = line[0]
                                    break
        if line[0] == '_': # replay header
            if "Human" in line:
                nbp += 1
                tmp = line.strip('_').rstrip(' \n').split(',')
                if ('name' in p[0] and p[0]['name'] != ''):
                    p[1]['name'] = tmp[0]
                    p[1]['race'] = tmp[2]
                else:
                    p[0]['name'] = tmp[0]
                    p[0]['race'] = tmp[2]
    f.close()
    if multiple_buildings:
        s.add('Protoss_Gateway2')
        s.add('Protoss_Gateway3')
        s.add('Protoss_Gateway4')
        s.add('Protoss_Assimilator2')
        s.add('Protoss_Nexus2')
        s.add('Protoss_Nexus3')
        s.add('Protoss_Pylon2')
        s.add('Protoss_Pylon3')
        s.add('Terran_Supply_Depot2')
        s.add('Terran_Supply_Depot3')
        s.add('Terran_Command_Center2')
        s.add('Terran_Command_Center3')
        s.add('Terran_Barracks2')
        s.add('Terran_Barracks3')
        s.add('Terran_Barracks4')
        s.add('Terran_Refinery2')
        s.add('Zerg_Hatchery2')
        s.add('Zerg_Hatchery3')
        s.add('Zerg_Hatchery4')
        s.add('Zerg_Overlord2')
        s.add('Zerg_Overlord3')
        s.add('Zerg_Extractor2')
    if step == 0:
        for at in s:
            if 'Terran_' in at:
                terran.add(at)
        for at in s:
            if 'Zerg_' in at and at != 'Zerg_': # TODO small bug with "Zerg_"
                zerg.add(at)
        for at in s:
            # for mind control
            if 'Protoss_' in at\
                    and at.replace('Protoss_', 'Terran_') not in terran\
                    and at.replace('Protoss_', 'Zerg_') not in zerg:
                protoss.add(at)
        if generate_attributes:
            attr = open('attributes.txt', 'w')
            for at in protoss:
                attr.write(at + '\n')
            for at in terran:
                attr.write(at + '\n')
            for at in zerg:
                attr.write(at + '\n')
            break
        else:
            if write_arffs: # arff headers
                pvp.write('@RELATION Starcraft_PvP\n')
                pvt.write('@RELATION Starcraft_PvT\n') 
                pvz.write('@RELATION Starcraft_PvZ\n') 
                tvt.write('@RELATION Starcraft_TvT\n') 
                tvz.write('@RELATION Starcraft_TvZ\n') 
                zvz.write('@RELATION Starcraft_ZvZ\n') 
                pvp.write('\n@ATTRIBUTE Winner INTEGER\n')
                pvt.write('\n@ATTRIBUTE Winner INTEGER\n') 
                pvz.write('\n@ATTRIBUTE Winner INTEGER\n') 
                tvt.write('\n@ATTRIBUTE Winner INTEGER\n') 
                tvz.write('\n@ATTRIBUTE Winner INTEGER\n') 
                zvz.write('\n@ATTRIBUTE Winner INTEGER\n') 
                for a in protoss:
                    pvp.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in protoss:
                    pvp.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in protoss:
                    pvt.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in terran:
                    pvt.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in protoss:
                    pvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in zerg:
                    pvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in terran:
                    tvt.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in terran:
                    tvt.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in terran:
                    tvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in zerg:
                    tvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in zerg:
                    zvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                for a in zerg:
                    zvz.write('@ATTRIBUTE ' + a + ' INTEGER\n')
                pvp.write('\n@DATA\n')
                pvt.write('\n@DATA\n')
                pvz.write('\n@DATA\n')
                tvt.write('\n@DATA\n')
                tvz.write('\n@DATA\n')
                zvz.write('\n@DATA\n')
