import sys, pickle, copy, math

# usage:
# python annotate.py matchs.txt -w
# will _w_rite matchsannotated.txt 
# with each player's most probable opening noted (last feature of each line)
# python annotate.py given_player_games.txt -s
# will print a _s_ummary of the players openings

f = open(sys.argv[1], 'r')
ptemplate = []
ttemplate = []
ztemplate = []
pdatal = []
tdatal = []
zdatal = []

def parse(l):
    if l == '':
        return
    ### Hack to remove names
    indn = l.find("Name")
    if (indn >= 0):
        l = l[:indn]+l[l.find('; ', indn)+2:]
    ### /End Hack
    lt = l.split(' ')
    d = []
    le = l.replace('; ',';').split(';')
    for elem in le:
        if len(elem.split(' ')) > 1:
            d.append(int(elem.split(' ')[1]))
    if "Protoss_" in l:
        if len(ptemplate) == 0:
            ptemplate.extend([lt[2*i] for i in range(len(lt)/2)])
        pdatal.append(d)
    elif "Terran_" in l:
        if len(ttemplate) == 0:
            ttemplate.extend([lt[2*i] for i in range(len(lt)/2)])
        tdatal.append(d)
    elif "Zerg_" in l:
        if len(ztemplate) == 0:
            ztemplate.extend([lt[2*i] for i in range(len(lt)/2)])
        zdatal.append(d)

def pnorm(x, m, s):
    """ 
    Compute the multivariate normal distribution with values vector x,
    mean vector m, sigma (variances/covariances) matrix s
    """
    xmt = np.matrix(x-m).transpose()
    for i in range(len(s)):
        if s[i,i] <= sys.float_info[3]: # min float
            s[i,i] = sys.float_info[3]
    sinv = np.linalg.inv(s)
    xm = np.matrix(x-m)
    return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
            *math.exp(-0.5*(xm*sinv*xmt))

def annotate(data, *args):
    def determine_cluster_ind(clusters):
        """ 
        The labeling cluster should be the one with globally smaller means
        __and__ the one with the smaller time to accomplish its first feature
        """
        ### /!\ shitty heuristic to determine which cluster is the one labeled
        cind1 = -1
        cind2 = -1
        minnorm = 10000000000000000000000000000.0 # ;)
        minff = 10000000000000000000000000000.0 # ;)
        if clusters.has_key('params'):
            params = clusters['params']
        elif clusters.has_key('centroids'):
            params = clusters['centroids']
        for i in range(len(params)):
            if clusters.has_key('params'):
                tmpnorm = np.linalg.norm(params[i]['mu']) # global cluster i means
                tmpff = params[i]['mu'][0] # mean time to have first feature
            elif clusters.has_key('centroids'):
                tmpnorm = np.linalg.norm(params[i]) # global cluster i means
                tmpff = params[i][0] # mean time to have first feature
            if tmpnorm < minnorm:
                minnorm = tmpnorm
                cind1 = i
            if tmpff < minff:
                minff = tmpff
                cind2 = i
        if cind1 == cind2:
            return cind1
        else:
            print "ERROR: Problem determining labeling cluster:",
            print clusters['name']
            print "with feature:", clusters['features'][0]
            print "min norm:", minnorm, " indice: ", cind1
            print "min first feature:", minff, " indice: ", cind2
            sys.exit(-1)
    annotations = {}
    annotations['openings'] = [{} for i in range(len(data))]
    annotations['games'] = copy.deepcopy(data)

    maxdim = 1.0 * max([len(d[1][0]) for d,useless in args if len(d[0]) > 0])

    for (d, clusters) in args:
        if len(d[0]) == 0:
            continue
        # determine the indice of the labeling cluster!
        cind = determine_cluster_ind(clusters)

        for i,game_data in enumerate(d[1]):
            tmpproba = pnorm(game_data, 
                    clusters['params'][cind]['mu'], 
                    clusters['params'][cind]['sigma'])
            tmpproba = tmpproba**(maxdim/len(game_data))
            annotations['openings'][d[0][i]][clusters['name']] = tmpproba
    return annotations

def most_probable_opening(di):
    mpo = ''
    mp = -0.1
    for k, v in di.iteritems():
        if v > mp:
            mpo = k
    return mpo

for line in f:
    parse(line.rstrip('\r\n'))
f.close()

import numpy as np
import clustering
clustering.formating = 'txt'

p_notes = {'openings' : {}}
t_notes = {'openings' : {}}
z_notes = {'openings' : {}}

if len(pdatal):
    print "Protoss player(s) detected in this file"
    f_ser = open("Protoss_models", 'r')
    two_gates = pickle.load(f_ser)
    fast_dt = pickle.load(f_ser)
    templar = pickle.load(f_ser)
    speedzeal = pickle.load(f_ser)
    corsair = pickle.load(f_ser)
    nony = pickle.load(f_ser)
    reaver_drop = pickle.load(f_ser)
    f_ser.close()
    pdata = np.ndarray([len(pdatal), len(pdatal[0])], np.float64)
    pdata /= 24
    for i in range(len(pdatal)):
        for j in range(len(pdatal[0])):
            pdata[i][j] = pdatal[i][j]
    two_gates_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Gateway2"),\
            ptemplate.index("Protoss_Gateway"),\
            ptemplate.index("Protoss_Zealot")], 1))
    fast_dt_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Dark_Templar")], 1))
    speedzeal_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Zealot_Speed"),\
            ptemplate.index("Protoss_Ground_Weapons")], 1))
    templar_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Psionic_Storm"),\
            ptemplate.index("Protoss_High_Templar")], 1))
    corsair_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Corsair")], 1))
    nony_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Dragoon"),\
            ptemplate.index("Protoss_Dragoon_Range")], 1))
    reaver_drop_data = clustering.filter_out_undef(pdata.take(
            [ptemplate.index("Protoss_Reaver"),\
            ptemplate.index("Protoss_Shuttle")], 1))

    p_notes = annotate(pdata,
            (two_gates_data, two_gates), (fast_dt_data, fast_dt),\
            (templar_data, templar), (speedzeal_data, speedzeal),\
            (corsair_data, corsair),\
            (nony_data, nony), (reaver_drop_data, reaver_drop))

if len(tdatal):
    print "Terran player(s) detected in this file"
    f_ser = open("Terran_models", 'r')
    bio = pickle.load(f_ser)
    rax_fe = pickle.load(f_ser)
    two_facto = pickle.load(f_ser)
    vultures = pickle.load(f_ser)
    drop = pickle.load(f_ser)
    f_ser.close()
    tdata = np.ndarray([len(tdatal), len(tdatal[0])], np.float64)
    tdata /= 24
    for i in range(len(tdatal)):
        for j in range(len(tdatal[0])):
            tdata[i][j] = tdatal[i][j]
    bio_data = clustering.filter_out_undef(tdata.take(
            [ttemplate.index("Terran_Barracks3"),\
            ttemplate.index("Terran_Barracks2"),\
            ttemplate.index("Terran_Barracks")], 1))
    rax_fe_data = clustering.filter_out_undef(tdata.take(
            [ttemplate.index("Terran_Expansion"),\
            ttemplate.index("Terran_Barracks")], 1))
    two_facto_data = clustering.filter_out_undef(tdata.take(
            [ttemplate.index("Terran_Factory2")], 1))
    vultures_data = clustering.filter_out_undef(tdata.take(
            [ttemplate.index("Terran_Spider_Mines"),\
            ttemplate.index("Terran_Vulture")], 1))
    drop_data = clustering.filter_out_undef(tdata.take(
            [ttemplate.index("Terran_Dropship")], 1))

    t_notes = annotate(tdata,
            (bio_data, bio), (rax_fe_data, rax_fe),\
            (two_facto_data, two_facto), (vultures_data, vultures),\
            (drop_data, drop))

if len(zdatal):
    print "Zerg player(s) detected in this file"
    f_ser = open("Zerg_models", 'r')
    speedlings = pickle.load(f_ser)
    fast_mutas = pickle.load(f_ser)
    mutas = pickle.load(f_ser)
    lurkers = pickle.load(f_ser)
    hydras = pickle.load(f_ser)
    f_ser.close()
    zdata = np.ndarray([len(zdatal), len(zdatal[0])], np.float64)
    zdata /= 24
    for i in range(len(zdatal)):
        for j in range(len(zdatal[0])):
            zdata[i][j] = zdatal[i][j]
    speedlings_data = clustering.filter_out_undef(zdata.take(
            [ztemplate.index("Zerg_Zergling_Speed"),\
            ztemplate.index("Zerg_Spawning_Pool"),\
            ztemplate.index("Zerg_Zergling")], 1))
    fast_mutas_data = clustering.filter_out_undef(zdata.take(
            [ztemplate.index("Zerg_Mutalisk"),\
            ztemplate.index("Zerg_Extractor")], 1))
    mutas_data = clustering.filter_out_undef(zdata.take(
            [ztemplate.index("Zerg_Expansion2"),\
            ztemplate.index("Zerg_Mutalisk")], 1))
    lurkers_data = clustering.filter_out_undef(zdata.take(
            [ztemplate.index("Zerg_Lurker")], 1))
    hydras_data = clustering.filter_out_undef(zdata.take(
            [ztemplate.index("Zerg_Hydralisk"),\
            ztemplate.index("Zerg_Hydralisk_Speed"),\
            ztemplate.index("Zerg_Hydralisk_Range")], 1))

    z_notes = annotate(zdata,
            (speedlings_data, speedlings), (fast_mutas_data, fast_mutas),\
            (mutas_data, mutas), (lurkers_data, lurkers),\
            (hydras_data, hydras))
        

if len(sys.argv) > 2:
    if sys.argv[2] == '-w':
        f = open(sys.argv[1], 'r')
        tow = open(sys.argv[1][:-4] + '_annotated.txt', 'w')
        i = -1
        j = -1
        k = -1
        for line in f:
            line = line.rstrip('\r\n') 
            op = ""
            if "Protoss_" in line:
                i += 1
                op = most_probable_opening(p_notes['openings'][i]) 
            elif "Terran_" in line:
                j += 1
                op = most_probable_opening(t_notes['openings'][j]) 
            elif "Zerg_" in line:
                k += 1
                op = most_probable_opening(z_notes['openings'][k]) 
            if op != "":
                tow.write(line + 'Opening ' + op + ';\n')
        tow.close()
    elif sys.argv[2] == '-s':
        openings = {}
        most_prob_openings = {}
        ngames = 0 # just for verification
        for game_openings in [t['openings'] for t in [p_notes,
                                                      t_notes, z_notes]]:
            for game in game_openings:
                ngames += 1
                mpop = most_probable_opening(game)
                if mpop in most_prob_openings:
                    most_prob_openings[mpop] += 1
                else:
                    most_prob_openings[mpop] = 1
                for o, p in game.iteritems():
                    if o in openings:
                        openings[o] = (openings[o][0]+1, openings[o][1]+p)
                    else:
                        openings[o] = (1, p)
        print most_prob_openings
        print openings
        print ngames

    
