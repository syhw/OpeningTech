#!/opt/local/bin/python
import sys

"""
Copyright 2011 Gabriel Synnaeve
License: Python Software Foundation License (PSFL, BSD-like, GPL compatible)
http://docs.python.org/license.html
"""

# pyt arff.py all.lmr [--generate-attributes]
def usage():
    print "Usage is:"
    print "pyt arff.py FILE.lmr [--generate-attributes]"

if len(sys.argv) < 2:
    usage()

generate_attributes = False
if sys.argv[2] == '--generate-attributes':
    generate_attributes = True

f = open(sys.argv[1], 'r')
pvp = open(sys.argv[1].split('.')[0]+'pvp.arff', 'w')
pvt = open(sys.argv[1].split('.')[0]+'pvt.arff', 'w')
pvz = open(sys.argv[1].split('.')[0]+'pvz.arff', 'w')
#tvp = open(sys.argv[1].split('.')[0]+'tvp.arff', 'w')
tvt = open(sys.argv[1].split('.')[0]+'tvt.arff', 'w')
tvz = open(sys.argv[1].split('.')[0]+'tvz.arff', 'w')
#zvp = open(sys.argv[1].split('.')[0]+'zvp.arff', 'w')
#zvt = open(sys.argv[1].split('.')[0]+'zvt.arff', 'w')
zvz = open(sys.argv[1].split('.')[0]+'zvz.arff', 'w')

s = set()

rep = ''
p = [{}, {}]
nbp = 0
skip = False

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

for line in f:
    if line[0] == '[': # new replay name, write the old
        #rep = ''
        p = [{}, {}]
        nbp = 0
    if line[0] in '0123456789':
        if nbp > 2:
            pass # SKIP (we are only interested in duel / 1vs1) 
        else:
            if generate_attributes:
                tmp = line.rstrip(' \n\r').split(',')
                tmp = extract_attribute(tmp)
                if tmp != '' and tmp not in s:
                    s.add(tmp)
                continue
            line = line.rstrip(' \n').split(',')
            if 'player quit' in line[len(line) - 1]:
                p[int(line[2])]['winner'] = 0
                p[1 - int(line[2])]['winner'] = 1
            else:
                pass

    if line[0] == '_': # header
        if "Human" in line:
            nbp += 1
            tmp = line.strip('_').rstrip(' \n').split(',')
            if ('name' in p[0] and p[0]['name'] != ''):
                p[1]['name'] = tmp[0]
                p[1]['race'] = tmp[2]
            else:
                p[0]['name'] = tmp[0]
                p[0]['race'] = tmp[2]

if generate_attributes:
    attr = open('attributes.txt', 'w')
    terran = []
    zerg = []
    for at in s:
        if 'Terran_' in at:
            terran.append(at)
    for at in s:
        if 'Zerg_' in at:
            zerg.append(at)
    for at in s:
        # for mind control
        if 'Protoss_' in at\
                and at.replace('Protoss_', 'Terran_') not in terran\
                and at.replace('Protoss_', 'Zerg_') not in zerg:
            attr.write(at + '\n')
    for at in terran:
        attr.write(at + '\n')
    for at in zerg:
        attr.write(at + '\n')
