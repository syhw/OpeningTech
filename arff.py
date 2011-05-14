import sys

# pyt arff.py all.lmr
generate_attributes = True

#new_rep = re.compile('^\[.*')
#action = re.compile('^\d.*')
f = open(sys.argv[1], 'r')
pvp = open(sys.argv[1].split('.')[0]+'pvp.arff', 'w')
pvt = open(sys.argv[1].split('.')[0]+'pvt.arff', 'w')
pvz = open(sys.argv[1].split('.')[0]+'pvz.arff', 'w')
tvp = open(sys.argv[1].split('.')[0]+'tvp.arff', 'w')
tvt = open(sys.argv[1].split('.')[0]+'tvt.arff', 'w')
tvz = open(sys.argv[1].split('.')[0]+'tvz.arff', 'w')
zvp = open(sys.argv[1].split('.')[0]+'zvp.arff', 'w')
zvt = open(sys.argv[1].split('.')[0]+'zvt.arff', 'w')
zvz = open(sys.argv[1].split('.')[0]+'zvz.arff', 'w')

s = set()

mu = ''
rep = ''
p1 = {}
p2 = {} 
nbp = 0
skip = False

for line in f:
    #if re.match(new_rep, line):
    if line[0] == '[': # new replay name, write the old
        #if mu != '':
        #mu = ''
        #rep = ''
        p1 = {}
        p2 = {}
        nbp = 0
    if line[0] in '0123456789':
        if nbp > 2:
            pass # SKIP 
        else:
            pass # TODO
        if generate_attributes:
            tmp = line.rstrip(' \n\r').split(',')
            tmpp = tmp[1]
            tmp = tmp[len(tmp)-1]
            if 'player' in tmp:
                continue
            tmp = tmp.replace(' ', '_')
            if '(' in tmp:
                tmp = tmp.split('(')[1].replace(')', '')
            if not 'Protoss' in tmp\
                    and (not 'Terran' in tmp or 'Infested' in tmp)\
                    and (not 'Zerg' in tmp or 'Zergling' in tmp):
                tmpr = ''
                if p1['name'] == tmpp:
                    tmpr = p1['race']
                elif p2['name'] == tmpp:
                    tmpr = p2['race']
                if tmpr == 'P':
                    tmp = 'Protoss_' + tmp
                elif tmpr == 'T':
                    tmp = 'Terran_' + tmp
                elif tmpr == 'Z':
                    tmp = 'Zerg_' + tmp
            if tmp != '' and tmp not in s:
                s.add(tmp)
    if line[0] == '_': # header
        if "Human" in line:
            nbp += 1
            tmp = line.strip('_').rstrip(' \n').split(',')
            if ('name' in p1 and p1['name'] != ''):
                p2['name'] = tmp[0]
                p2['race'] = tmp[2]
            else:
                p1['name'] = tmp[0]
                p1['race'] = tmp[2]

if generate_attributes:
    attr = open('attributes.txt', 'w')
    for at in s:
        if 'Protoss' in at:
            attr.write(at+'\n')
    for at in s:
        if 'Terran' in at:
            attr.write(at+'\n')
    for at in s:
        if 'Zerg' in at:
            attr.write(at+'\n')
