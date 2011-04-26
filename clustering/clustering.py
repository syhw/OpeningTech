#!/opt/local/bin/python
# vim: set fileencoding=utf-8 :
# Playschool kode disclaimer! Completely untested!

# Python License 2.0.1 http://www.python.org/download/releases/2.0.1/license/
# Copyright 2011 Gabriel Synnaeve

# TODO a usage documentation (gather "DOC")

import sys, random, copy, math
try:
    import numpy as np
except:
    print "You need numpy for computations."
try:
    import pylab as pl
except:
    print "You need pylab/matplotlib for plotting."

def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()

def k_means(t, nbclusters=2, nbiter=3, medoids=False, soft=False, beta=200.0,\
        #distance=lambda x,y: np.linalg.norm(x-y),\
        distance=lambda x,y: math.sqrt(np.dot(x-y,(x-y).conj())),\
        responsability=lambda beta,d: math.exp(-1 * beta * d)):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters/centroids 
    'nbiter' is the number of iterations
    'medoids' tells is we use the medoids or centroids method
    'distance' is the function to use for comparing observations

    Overview of the algorithm ("hard k-means"):
    -> Place nbclusters points into the features space of the objects/t[i:]
    -> Assign each object to the group that has the closest centroid (distance)
    -> Recalculate the positions of the nbclusters centroids
    -> Repeat Steps 2 and 3 until the centroids no longer move

    We can change the distance function and change the responsability function
    -> distance will change the shape of the clusters
    -> responsability will change the breadth of the clusters (& associativity)
    """
    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    # find ranges for each features
    min_max = []
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))

    ### Soft => Normalization, otherwise "beta" has no meaning!
    if soft:
        for f in range(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    ### /Normalization # ugly

    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    tmpdist = np.ndarray([nbobs,nbclusters], np.float64) # distance obs<->clust
    tmpresp = np.ndarray([nbobs,nbclusters], np.float64) # responsability o<->c
    # iterate for the best quality
    for iteration in range(nbiter):
        clusters = [[] for c in range(nbclusters)]
        # Step 1: place nbclusters seeds for each features
        centroids = [np.array([random.uniform(min_max[f][0], min_max[f][1])\
                for f in range(nbfeatures)], np.float64)\
                for c in range(nbclusters)]
        old_centroids = [np.array([-1 for f in range(nbfeatures)], np.float64)\
                for c in range(nbclusters)] # should not be init, TODO
        new_sum = math.fsum([distance(centroids[c], old_centroids[c])\
                for c in range(nbclusters)])
        old_sum = sys.maxint
        np.seterr(invalid='raise')
        # iterate until convergence
        while new_sum < old_sum :
            old_centroids = copy.deepcopy(centroids)
            old_sum = new_sum
            for c in range(nbclusters):
                clusters[c] = []
            # precompute distance to all centroids/medoids for all observations
            for c in range(nbclusters):
                for o in range(nbobs):
                    tmpdist[o,c] = distance(centroids[c], t[o,:])
            if soft:
                # Step 2: compute the degree of assignment for each object
                for o in range(nbobs):
                    for c in range(nbclusters):
                        tmpresp[o,c] = responsability(beta, tmpdist[o,c])
                for o in range(nbobs):
                    tmpresp[o,:] /= math.fsum(tmpresp[o,:])
            else:
                # Step 2: assign each object to the closest centroid
                for o in range(nbobs):
                    clusters[tmpdist[o,:].argmin()].append(o)
            # Step 3: recalculate the positions of the nbclusters centroids
            for c in range(nbclusters):
                if medoids:
                    if soft:
                        print "Not implemented"
                        sys.exit(-1)
                    else:
                        tmpmin = sys.maxint
                        argmin = 0
                        for o in clusters[c]:
                            if tmpdist[o,c] < tmpmin:
                                tmpmin = tmpdist[o,c]
                                argmin = o
                        centroids[c] = t[argmin,:]
                else:
                    mean = np.array([0 for i in range(nbfeatures)], np.float64)
                    if soft:
                        for o in range(nbobs):
                            mean += tmpresp[o,c] * t[o,:]
                        mean /= math.fsum(tmpresp[:,c])
                    else:
                        for o in clusters[c]:
                            mean += t[o,:]
                        mean = map(lambda x: x/len(clusters[c]), mean)
                    centroids[c] = np.array(mean, np.float64)
            print centroids
            new_sum = math.fsum([distance(centroids[c], old_centroids[c])\
                    for c in range(nbclusters)])
            print "(k-means) old and new sum: ", old_sum, new_sum
        if soft:
            for o in range(nbobs):
                clusters[tmpdist[o,:].argmin()].append(o)
        quality = math.fsum([math.fsum([tmpdist[o][c] for o in clusters[c]])\
                /(len(clusters[c])+1) for c in range(nbclusters)])
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['centroids'] = centroids
            result['clusters'] = clusters
    return result

def r_em(t, nbclusters=0, plot=False):
    try:
        from rpy2.robjects import r
        import rpy2.robjects.numpy2ri # auto-translates numpy array to R ones
    except:
        print "You can't use 'r_em()' without rpy2 and the R library mclust"
        sys.exit(-1)
    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    result = {}
    r.library("mclust")
    if nbclusters:
        model = r.Mclust(t, G=nbclusters)
    else:
        model = r.Mclust(t)
    if plot:
        r.quartz("plot")
        r.plot(model, t)
    #for e in model:
    #    print e
    params = []
    if model[7][3][0][0] == "E":
        for i in range(nbclusters):
            params.append({'mu': np.array([model[7][2][i*nbfeatures+j]\
                                 for j in range(nbfeatures)], np.float64),\
                    'sigma': np.matrix(np.array([np.array([model[7][3][3]\
                    [jj*nbfeatures+j] for j in range(nbfeatures)],\
                    np.float64) for jj in range(nbfeatures)], np.float64)),\
                    'proba': model[7][1][i]})
    else:
        for i in range(nbclusters):
            params.append({'mu': np.array([model[7][2][i*nbfeatures+j]\
                                 for j in range(nbfeatures)], np.float64),\
                    'sigma': np.matrix(np.array([np.array([model[7][3][3]\
                    [i*(nbfeatures**2)+jj*nbfeatures+j] for j in range(nbfeatures)],\
                    np.float64) for jj in range(nbfeatures)], np.float64)),\
                    'proba': model[7][1][i]})
    result['quality'] = 'not computed'
    result['params'] = copy.deepcopy(params)
    result['clusters'] = [[o for o in range(nbobs)\
            if model[9][o] == c+1.0]\
            for c in range(nbclusters)]
    return result

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

def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,\
        epsilon=0.001, monotony=False, datasetinit=True):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters
    'nbiter' is the number of iterations
    'distance' is the function to use for comparing observations

    Overview of the algorithm:
    -> Draw nbclusters sets of (μ, σ, P_{#cluster}) at random (Gaussian 
       Mixture) [P(Cluster=0) = P_0 = (1/n).∑_{obs} P(Cluster=0|obs)]
    -> Compute P(Cluster|obs) for each obs, this is:
    [E] P(Cluster=0|obs)^t = P(obs|Cluster=0)*P(Cluster=0)^t
    -> Recalculate the mixture parameters with the new estimate
    [M] * P(Cluster=0)^{t+1} = (1/n).∑_{obs} P(Cluster=0|obs)
        * μ^{t+1}_0 = ∑_{obs} obs.P(Cluster=0|obs) / P_0
        * σ^{t+1}_0 = ∑_{obs} P(Cluster=0|obs)(obs-μ^{t+1}_0)^2 / P_0
    -> Compute E_t=∑_{obs} log(P(obs)^t)
       Repeat Steps 2 and 3 until |E_t - E_{t-1}| < ε
    """
    def draw_params():
            if datasetinit:
                tmpmu = np.array([1.0*t[random.uniform(0,nbobs),:]],np.float64)
            else:
                tmpmu = np.array([random.uniform(min_max[f][0], min_max[f][1])\
                        for f in range(nbfeatures)], np.float64)
            return {'mu': tmpmu,\
                    'sigma': np.matrix(np.diag(\
                    [(min_max[f][1]-min_max[f][0])/2.0\
                    for f in range(nbfeatures)])),\
                    'proba': 1.0/nbclusters}

    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    min_max = []
    # find ranges for each features
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    
    ### Normalization
    if normalize:
        for f in range(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    ### /Normalization # ugly

    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(clust|obs)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(obs|clust)
    # iterate nbiter times searching for the best "quality" clustering
    for iteration in range(nbiter):
        ##############################################
        # Step 1: draw nbclusters sets of parameters #
        ##############################################
        params = [draw_params() for c in range(nbclusters)]
        old_log_estimate = sys.maxint         # init, not true/real
        log_estimate = sys.maxint/2 + epsilon # init, not true/real
        estimation_round = 0
        # Iterate until convergence (EM is monotone) <=> < epsilon variation
        while (abs(log_estimate - old_log_estimate) > epsilon\
                and (not monotony or log_estimate < old_log_estimate)):
            restart = False
            old_log_estimate = log_estimate
            ########################################################
            # Step 2: compute P(Cluster|obs) for each observations #
            ########################################################
            for o in range(nbobs):
                for c in range(nbclusters):
                    # Px[o,c] = P(x|c)
                    Px[o,c] = pnorm(t[o,:],\
                            params[c]['mu'], params[c]['sigma'])
            #for o in range(nbobs):
            #    Px[o,:] /= math.fsum(Px[o,:])
            for o in range(nbobs):
                for c in range(nbclusters):
                    # Pclust[o,c] = P(c|x)
                    Pclust[o,c] = Px[o,c]*params[c]['proba']
            #    assert math.fsum(Px[o,:]) >= 0.99 and\
            #            math.fsum(Px[o,:]) <= 1.01
            for o in range(nbobs):
                tmpSum = 0.0
                for c in range(nbclusters):
                    tmpSum += params[c]['proba']*Px[o,c]
                Pclust[o,:] /= tmpSum
                #assert math.fsum(Pclust[:,c]) >= 0.99 and\
                #        math.fsum(Pclust[:,c]) <= 1.01
            ###########################################################
            # Step 3: update the parameters (sets {mu, sigma, proba}) #
            ###########################################################
            print "iter:", iteration, " estimation#:", estimation_round,\
                    " params:", params
            for c in range(nbclusters):
                tmpSum = math.fsum(Pclust[:,c])
                params[c]['proba'] = tmpSum/nbobs
                if params[c]['proba'] <= 1.0/nbobs:           # restart if all
                    restart = True                            # converges to
                    print "Restarting, p:",params[c]['proba'] # one cluster
                    break
                m = np.zeros(nbfeatures, np.float64)
                for o in range(nbobs):
                    m += t[o,:]*Pclust[o,c]
                params[c]['mu'] = m/tmpSum
                s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
                for o in range(nbobs):
                    s += Pclust[o,c]*(np.matrix(t[o,:]-params[c]['mu']).transpose()*\
                            np.matrix(t[o,:]-params[c]['mu']))
                    #print ">>>> ", t[o,:]-params[c]['mu']
                    #diag = Pclust[o,c]*((t[o,:]-params[c]['mu'])*\
                    #        (t[o,:]-params[c]['mu']).transpose())
                    #print ">>> ", diag
                    #for i in range(len(s)) :
                    #    s[i,i] += diag[i]
                params[c]['sigma'] = s/tmpSum
                print "------------------"
                print params[c]['sigma']

            ### Test bound conditions and restart consequently if needed
            if not restart:
                restart = True
                for c in range(1,nbclusters):
                    if not np.allclose(params[c]['mu'], params[c-1]['mu'])\
                    or not np.allclose(params[c]['sigma'], params[c-1]['sigma']):
                        restart = False
                        break
            if restart:                # restart if all converges to only
                old_log_estimate = sys.maxint         # init, not true/real
                log_estimate = sys.maxint/2 + epsilon # init, not true/real
                params = [draw_params() for c in range(nbclusters)]
                continue
            ### /Test bound conditions and restart

            ####################################
            # Step 4: compute the log estimate #
            ####################################
            log_estimate = math.fsum([math.log(math.fsum(\
                    [Px[o,c]*params[c]['proba'] for c in range(nbclusters)]))\
                    for o in range(nbobs)])
            print "(EM) old and new log estimate: ",\
                    old_log_estimate, log_estimate
            estimation_round += 1

        # Pick/save the best clustering as the final result
        quality = -log_estimate
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(params)
            result['clusters'] = [[o for o in range(nbobs)\
                    if Px[o,c] == max(Px[o,:])]\
                    for c in range(nbclusters)]
    return result

def parse(arff):
    template = []
    t = []
    data = False
    for line in arff:
        if data:
            tmp = []
            for elem in line.split(','):
                if elem[0] in "0123456789":
                    # Convert to int, and 24 frames per second:
                    # we don't need such a high resolution
                    tmp.append(int(elem))#/24) 
                else:
                    tmp.append(elem.rstrip('\n'))
            t.append(tmp)
        elif '@attribute' in line:
            template.append(line.split(' ')[1])
        elif '@data' in line:
            data = True
    return (template, t)

def filter_out_undef(tab, typ=np.float64):
    def not_undef(t):
        for e in t:
            if e < 0: # undef <=> < 0
                return False
        return True
    indices = []
    tmp = []
    for i in range(len(tab)):
        if not_undef(tab[i]):
            indices.append(i)
            tmp.append(tab[i])
    return (indices, np.array(tmp, typ))

def plot(clusterst, data, title='', gaussians=[], separate_plots=False):
    if clusterst.has_key('name'):
        title = clusterst['name']
    if clusterst.has_key('params'):
        gaussians = clusterst['params']
    clusters = clusterst['clusters']
    if separate_plots:
        ax = pl.subplot(212)
    else:
        ax = pl.subplot(111)
    colors = 'brgcymk'
    for k in range(len(clusters)):
        print ">>>> drawing ", k
        if len(clusters[k]):
            xy = [[data[i,j] for i in clusters[k]]\
                    for j in range(len(data[0]))]
            if len(data[0]) < 3:
                ax.scatter(xy[0], xy[len(data[0])-1],s=20,\
                        c=colors[k % len(colors)],\
                        marker='s', edgecolors='none')
            else:
                # [len(data[0]) * (len(data[0])+1)] / 2 plots
                total = math.sqrt(len(data[0])*(len(data[0])+1.0)/2.0)
                w = int(math.floor(total))
                h = int(math.floor(total+1))
                plotno = 0 
                for i in range(len(data[0])):
                    r = range(i+1, len(data[0]))
                    for j in r:
                        plotno += 1
                        ax = pl.subplot(h, w, plotno)
                        ax.scatter(xy[i], xy[j],s=20,\
                                c=colors[k % len(colors)], marker='s',\
                                edgecolors='none')
                        
    ranges = [min([data[:,i].min() for i in range(data.shape[1])]),\
            max([data[:,i].max() for i in range(data.shape[1])]),\
            min([data[i,:].min() for i in range(data.shape[0])]),\
            max([data[i,:].max() for i in range(data.shape[0])])]
    ax.axis(ranges)
    pl.title(title)

    ### Plot gaussians
    if len(gaussians):
        Z = []
        for g in gaussians:
            delta = (max(ranges) - min(ranges))/1000
            x = pl.arange(ranges[0], ranges[1], delta)
            y = pl.arange(ranges[2], ranges[3], delta)
            X,Y = pl.meshgrid(x, y)
            for i in range(len(g['sigma'])):
                if not g['sigma'][i,i]:                  # to put uncertainty on
                    g['sigma'][i,i] += 1.0/data.shape[0] # perfectly aligned data
            if len(g['sigma']) == 1:
                Z.append(pl.bivariate_normal(X, Y, g['sigma'][0,0],\
                        g['sigma'][0,0], g['mu'][0], g['mu'][0]))
            else:
                Z.append(pl.bivariate_normal(X, Y, g['sigma'][0,0],\
                        g['sigma'][1,1], g['mu'][0], g['mu'][1]))
        if separate_plots: # only supports 2 clusters currently, TODO
            cmap = pl.cm.get_cmap('jet', 10)    # 10 discrete colors
            ay = pl.subplot(221)
            ay.imshow(Z[0], cmap=cmap, interpolation='bilinear',\
                    origin='lower', extent=ranges)
            az = pl.subplot(222)
            az.imshow(Z[1], cmap=cmap, interpolation='bilinear',\
                    origin='lower', extent=ranges)
        else:
            #ZZ = sum(Z)
            if len(data[0]) < 3:
                for i in range(len(Z)):
                    ax.contour(X, Y, Z[i], 1, colors='k')# colors=colors[i%len(colors)])
    ### /Plot gaussians 

    pl.grid(True)
    pl.show()

def annotate(data, *args):
    annotations = {}
    annotations['openings'] = []
    annotations['games'] = copy.deepcopy(data)
    annotations['metadata'] = []
    labelind = len(data[0]) - 1
    # Remove the precedents labels/strategies/openings
    for game in annotations['games']:
        game[labelind] = ''
        annotations['metadata'].append({})
    # Determine which are the labels of brought by "clusters"
    for (data, clusters) in args:
        # data[0] are the true indices in data of data[1] (filtered data)
        # clusters['name'] / clusters['clusters'] / clusters['params']
        annotations['openings'].append(clusters['name'])

        ### /!\ shitty heuristic to determine which cluster is the one labeled
        # the labeling cluster should be the one with globally smaller means
        cind = -1
        minnorm = 100000000000000000000000000000000000000000000.0 # ;)
        if clusters.has_key('params'):
            params = clusters['params']
        elif clusters.has_key('centroids'):
            params = clusters['centroids']
        for i in range(len(params)):
            if clusters.has_key('params'):
                tmpnorm = np.linalg.norm(params[i]['mu'])
            elif clusters.has_key('centroids'):
                tmpnorm = np.linalg.norm(params[i])
            if tmpnorm < minnorm:
                minnorm = tmpnorm
                cind = i
        # clusters['clusters'][cind] is the list of the games labeled
        # clusters['name'] in data[1], their indices in data is in data[0]
        ### /!\ /shitty heuristic
        
        # Add each label to each game
        for g in clusters['clusters'][cind]:
            annotations['games'][data[0][g]][labelind] += clusters['name']+' '
            if clusters.has_key('params'):
                annotations['metadata'][data[0][g]][clusters['name']] = \
                        (clusters['params'], clusters['features'], cind)
            elif clusters.has_key('centroids'):
                annotations['metadata'][data[0][g]][clusters['name']] = \
                        (clusters['centroids'], clusters['features'], cind)
    # we can return here and print the full labels of replays
    # Make a selection of one label/strategy/opening per replay
    for game in annotations['games']:
        game[labelind].rstrip(' ')
        sp = game[labelind].split(' ')
        if len(sp) <= 1:
            if game[labelind] == '':
                game[labelind] == 'unknown'
            continue
        bestlabelp = ''
        bestproba = 0.0
        probat = 0.0
        bestlabelt = ''
        mintime = 10000000.0
        maxdim = 1.0 * max([len(v[1]) for v in annotations['metadata'][\
                annotations['games'].index(game)].itervalues()])
        npgame = np.array(game[:len(game)-1], np.float64)
        for (k, v) in annotations['metadata'][annotations['games']\
                .index(game)].iteritems():
            gp = v[0][v[2]]
            tmpproba = pnorm(npgame.take(v[1]), gp['mu'], gp['sigma'])
            tmpproba = tmpproba**(maxdim/len(v[1]))
            if tmpproba > bestproba:
                bestlabelp = k
                bestproba = tmpproba
            # v[1][0] =  # DOC: the first feature is always the "most important"
            if v[1][0] < mintime:
                mintime = v[1][0]
                bestlabelt = k
                probat = tmpproba
        if bestlabelp == bestlabelt or bestproba < (probat + 0.01**maxdim):
            game[labelind] = bestlabelt # if probat is only at 1% of bestproba
        else:
            game[labelind] = 'unknown'
    return annotations

def write_arff(template, annotations,fn):
    f = open(fn, 'w')
    r = fn.split('_')[1] # race that performs the openings
    f.write('@relation Starcraft_'+sys.argv[1][3:6]+'_'+r+'_openings\n')
    f.write('\n')
    for attr in template:
        if not 'midBuild' in attr:
            f.write('@attribute '+attr+' numeric\n')
        else:
            f.write('@attribute '+attr+' {'+\
                    ','.join(annotations['openings'])+'}\n')
    f.write('\n@data\n')
    for game in annotations['games']:
        #print ','.join([str(i) for i in game])
        f.write(','.join([str(i) for i in game])+'\n')
    return 

if __name__ == "__main__":
    if sys.argv[1] == "test":
        #t_data = np.array([[1,1],[11,11]], np.float64)
        #t_data = np.array([[1,1],[0,1],[1,0],[10,10],[11,9],\
        #                [11,12],[9,9],[1,2]], np.float64)
        t_data = np.array([[2,4],[1,2],[1,3],[1,4],[2,1],\
                        [3,1],[4,1],[4,2]], np.float64)
        #t_data = np.array([[1],[3],[1],[2],\
        #                [12],[11],[11],[10]], np.float64)
        t1 = k_means(t_data, nbiter=10)
        print t1
        plot(t1["clusters"],t_data, "test k-means")
        t2 = expectation_maximization(t_data, nbiter=10, normalize=False)
        print t2
        plot(t2["clusters"],t_data, "test EM", t2['params'])
        sys.exit(0)
    nbiterations = 5 # TODO 100 when clustering for real
    kmeans = False
    EM = False
    plotR = True
    plotM = False

    (template, datalist) = parse(open(sys.argv[1]))
    # build data without the "label"/opening/strategy column
    data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.float64)
    data /= 24
    # transform the kind & dynamic python list into a static numpy.ndarray
    for i in range(len(datalist)):
        for j in range(len(datalist[0]) - 1):
            data[i][j] = datalist[i][j]

    race = sys.argv[1].split('_')[1][0] # race that performs the openings
    matchup = sys.argv[1][5]            # race against which it performs

    if race == "P":
        # Main openings:
        # - 2 gateways aggression (proxy or not)
        # - fast DT
        # - FE into
        #   * +1 speed (legs) zealot push
        #   * templar tech: bisu build (sair/templar) or sair/DT
        #   * sair/reaver TODO
        # - gate/core/gate goons (nony)
        # - reaver drop 
        # - cannon rush [Disabled: can only scout it]

        ### 2 gates rush opening
        features_two_gates = [template.index("ProtossSecondGatway"),\
                template.index("ProtossGateway")]
        if kmeans:
            two_gates_data_int = filter_out_undef(data.take(\
                    features_two_gates, 1), typ=np.int64)
            two_gates_km = k_means(two_gates_data_int[1], nbiter=nbiterations)
        two_gates_data = filter_out_undef(data.take(features_two_gates, 1))
        if EM:
            two_gates_em = expectation_maximization(two_gates_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_gates = r_em(two_gates_data[1], nbclusters=2, plot=plotR)
        two_gates['features'] = features_two_gates

        ### Fast DT
        features_fast_dt = [template.index("ProtossDarkTemplar")]
        if kmeans:
            fast_dt_data_int = filter_out_undef(data.take(\
                    features_fast_dt, 1), typ=np.int64)
            fast_dt_km = k_means(fast_dt_data_int[1], nbiter=nbiterations,\
                    distance = lambda x,y: abs(x-y))
        fast_dt_data = filter_out_undef(data.take(features_fast_dt,1))
        if EM:
            fast_dt_em = expectation_maximization(fast_dt_data[1],\
                    nbiter=nbiterations, monotony=True, normalize=True)
        fast_dt = r_em(fast_dt_data[1], nbclusters=2, plot=plotR)
        fast_dt['features'] = features_fast_dt

        ### Fast Expand
        features_fast_exp = [template.index("ProtossFirstExpansion")]
        if kmeans:
            fast_exp_data_int = filter_out_undef(data.take(\
                    features_fast_exp, 1), typ=np.int64)
            fast_exp_km = k_means(fast_exp_data_int[1], nbiter=nbiterations,\
                    distance = lambda x,y: abs(x-y))
        fast_exp_data = filter_out_undef(data.take(features_fast_exp, 1))
        if EM:
            fast_exp_em = expectation_maximization(fast_exp_data[1],\
                    nbiter=nbiterations, monotony=True, normalize=True)
        fast_exp = r_em(fast_exp_data[1], nbclusters=2, plot=plotR)
        fast_exp['features'] = features_fast_exp

        ### +1 SpeedZeal
        features_speedzeal = [template.index("ProtossGroundWeapons1"),\
                    template.index("ProtossLegs")]
        if kmeans:
            speedzeal_data_int = filter_out_undef(data.take(\
                    features_speedzeal, 1), typ=np.int64)
            speedzeal_km = k_means(speedzeal_data_int[1], nbiter=nbiterations)
        speedzeal_data = filter_out_undef(data.take(features_speedzeal, 1))
        if EM:
            speedzeal_em = expectation_maximization(speedzeal_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedzeal = r_em(speedzeal_data[1], nbclusters=2, plot=plotR)
        speedzeal['features'] = features_speedzeal

        ### Bisu build
        features_bisu = [template.index("ProtossFirstExpansion"),\
                    template.index("ProtossCorsair"),\
                    template.index("ProtossDarkTemplar")]
        if kmeans:
            bisu_data_int = filter_out_undef(data.take(\
                    features_bisu, 1), typ=np.int64)
            bisu_km = k_means(bisu_data_int[1], nbiter=nbiterations)
        bisu_data = filter_out_undef(data.take(features_bisu, 1))
        if EM:
            bisu_em = expectation_maximization(bisu_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        bisu = r_em(bisu_data[1], nbclusters=2, plot=plotR)
        bisu['features'] = features_bisu

        ### Corsair opening
        features_corsair = [template.index("ProtossCorsair")]
        if kmeans:
            corsair_data_int = filter_out_undef(data.take(\
                    features_corsair, 1), typ=np.int64)
            corsair_km = k_means(corsair_data_int[1], nbiter=nbiterations)
        corsair_data = filter_out_undef(data.take(features_corsair, 1))
        if EM:
            corsair_em = expectation_maximization(corsair_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        corsair = r_em(corsair_data[1], nbclusters=2, plot=plotR)
        corsair['features'] = features_corsair

        ### Nony opening
        features_nony = [template.index("ProtossRange"),\
                    template.index("ProtossSecondGatway"),\
                    template.index("ProtossGoon")]
        if kmeans:
            nony_data_int = filter_out_undef(data.take(\
                    features_nony, 1), typ=np.int64)
            nony_km = k_means(nony_data_int[1], nbiter=nbiterations)
        nony_data = filter_out_undef(data.take(features_nony, 1))
        if EM:
            nony_em = expectation_maximization(nony_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        nony = r_em(nony_data[1], nbclusters=2, plot=plotR)
        nony['features'] = features_nony


        ### Reaver Drop
        features_reaver_drop = [template.index("ProtossReavor"),\
                template.index("ProtossShuttle")]
        if kmeans:
            reaver_drop_data_int = filter_out_undef(data.take(\
                    features_reaver_drop, 1), typ=np.int64)
            reaver_drop_km = k_means(reaver_drop_data_int[1], nbiter=nbiterations)
        reaver_drop_data = filter_out_undef(data.take(features_reaver_drop, 1))
        if EM:
            reaver_drop_em = expectation_maximization(reaver_drop_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        reaver_drop = r_em(reaver_drop_data[1], nbclusters=2, plot=plotR)
        reaver_drop['features'] = features_reaver_drop

        ### [Disabled] Cannon Rush
#        if kmeans:
#            cannon_rush_data_int = filter_out_undef(data.take([\
#                    template.index("ProtossForge"), template.index("ProtossCannon")\
#                    ], 1), typ=np.int64)
#            cannon_rush_km = k_means(cannon_rush_data_int[1], nbiter=nbiterations)
#        cannon_rush_data = filter_out_undef(data.take([\
#                template.index("ProtossForge"), template.index("ProtossCannon")\
#                ], 1))
#        if EM:
#            cannon_rush_em = expectation_maximization(cannon_rush_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        cannon_rush = r_em(cannon_rush_data[1], nbclusters=2, plot=plotR)

        two_gates['name'] = "two_gates"
        fast_dt['name'] = "fast_dt"
        fast_exp['name'] = "fast_exp" # TODO (not satisfied)
        speedzeal['name'] = "speedzeal"
        bisu['name'] = "bisu"
        corsair['name'] = "corsair"
        nony['name'] = "nony"
        reaver_drop['name'] = "reaver_drop"
        if plotM:
            print two_gates
            plot(two_gates, two_gates_data[1])
            print fast_dt
            plot(fast_dt, fast_dt_data[1])
            print fast_exp
            plot(fast_exp, fast_exp_data[1])
            print speedzeal
            plot(speedzeal, speedzeal_data[1])
            print bisu
            plot(bisu, bisu_data[1])
            print corsair
            plot(corsair, corsair_data[1])
            print nony
            plot(nony,nony_data[1])
            print reaver_drop
            plot(reaver_drop, reaver_drop_data[1])
            #print cannon_rush
            #plot(cannon_rush["clusters"],cannon_rush_data[1], "cannon rush",\
            #        cannon_rush["params"])

        write_arff(template, annotate(datalist,\
                (two_gates_data, two_gates), (fast_dt_data, fast_dt),\
                (fast_exp_data, fast_exp), (speedzeal_data, speedzeal),\
                (bisu_data, bisu), (corsair_data, corsair),\
                (nony_data, nony), (reaver_drop_data, reaver_drop)),\
                "my"+sys.argv[1])

    if race == "T":
        # Main openings:
        # - BBS rush (rax/rax/supply) / 8 rax [Disabled: can only scout it]
        # - Bio push (3 raxes at least)
        # - 1 Rax FE or 2 Rax FE
        # - Siege Expand (facto into siege + expand)
        # - 2 Factories (aggressive push / nada style)
        # - Vultures harass
        # - Wraith

        ### [Disabled] BBS rush
#        bbs_data = filter_out_undef(data.take([\
#                template.index("TerranBarracks"),\
#                template.index("TerranSecondBarracks"),\
#                template.index("TerranDepot")], 1))
#        if EM:
#            bbs_em = expectation_maximization(bbs_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        bbs = r_em(bbs_data[1], nbclusters=2, plot=plotR)

        ### Bio push
        features_bio = [template.index("TerranThirdBarracks"),\
                template.index("TerranSecondBarracks"),\
                template.index("TerranBarracks")]
        bio_data = filter_out_undef(data.take(features_bio, 1))
        if EM:
            bio_em = expectation_maximization(bio_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        bio = r_em(bio_data[1], nbclusters=2, plot=plotR)
        bio['features'] = features_bio

        ### 1/2 Rax FE
        features_rax_fe = [template.index("TerranExpansion"),\
                template.index("TerranBarracks")]
        rax_fe_data = filter_out_undef(data.take(features_rax_fe , 1))
        if EM:
            rax_fe_em = expectation_maximization(rax_fe_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        rax_fe = r_em(rax_fe_data[1], nbclusters=2, plot=plotR)
        rax_fe['features'] = features_rax_fe

        ### Siege Expand
        features_siege_exp = [template.index("TerranSiege"),\
                template.index("TerranExpansion")]
        siege_exp_data = filter_out_undef(data.take(features_siege_exp, 1))
        if EM:
            siege_exp_em = expectation_maximization(siege_exp_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        siege_exp = r_em(siege_exp_data[1], nbclusters=2, plot=plotR)
        siege_exp['features'] = features_siege_exp
        siege_exp['features'] = features_siege_exp

        ### 2 Facto
        features_two_facto = [template.index("TerranSecondFactory"),\
                template.index("TerranTank")]
        two_facto_data = filter_out_undef(data.take(features_two_facto, 1))
        if EM:
            two_facto_em = expectation_maximization(two_facto_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_facto = r_em(two_facto_data[1], nbclusters=2, plot=plotR)
        two_facto['features'] = features_two_facto

        ### Vultures harass
        features_vultures = [template.index("TerranMines"),\
                template.index("TerranVulture")]
        vultures_data = filter_out_undef(data.take(features_vultures, 1))
        if EM:
            vultures_em = expectation_maximization(vultures_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        vultures = r_em(vultures_data[1], nbclusters=2, plot=plotR)
        vultures['features'] = features_vultures

        ### Fast Wraith
        features_wraith = [template.index("TerranWraith"),\
                template.index("TerranStarport")]
        wraith_data = filter_out_undef(data.take(features_wraith, 1))
        if EM:
            wraith_em = expectation_maximization(wraith_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        wraith = r_em(wraith_data[1], nbclusters=2, plot=plotR)
        wraith['features'] = features_wraith

        bio['name'] = "bio"
        rax_fe['name'] = "rax_fe"
        siege_exp['name'] = "siege_exp"
        two_facto['name'] = "two_facto"
        vultures['name'] = "vultures"
        wraith['name'] = "wraith"
        if plotM:
            #bbs['name'] = "bbs"
            #print bbs
            #plot(bbs, bbs_data[1])
            print bio
            plot(bio, bio_data[1])
            print rax_fe
            plot(rax_fe, rax_fe_data[1])
            print siege_exp
            plot(siege_exp, siege_exp_data[1])
            print two_facto
            plot(two_facto, two_facto_data[1])
            print vultures
            plot(vultures, vultures_data[1])
            print wraith
            plot(wraith, wraith_data[1])

        write_arff(template, annotate(datalist,\
                (bio_data, bio), (rax_fe_data, rax_fe),\
                (siege_exp_data, siege_exp), (two_facto_data, two_facto),\
                (vultures_data, vultures), (wraith_data, wraith)),\
                "my"+sys.argv[1])

    if race == "Z":
        # Main openings:
        # - 4-6 pools very early glings rush [Disabled: can only scout it]
        # - ~9pool/9speed speedlings rush
        # - any kind of fast expand (overpool, 12 hatch...) into:
        #   * fast mutas (2 hatches muta, or even 1 hatch mutas in ZvZ
        #   * mass mutas (3 hatches, or more, into mutas)
        #   * fast lurkers (3 hatch lurker)
        #   * hydras push/drop

        ### [Disabled] Very early zerglings rush (4 to 6 pool)
#        glings_rush_data = filter_out_undef(data.take([\
#                template.index("ZergPool"),\
#                template.index("ZergZergling")], 1))
#        glings_rush_data_int = filter_out_undef(data.take([\
#                template.index("ZergPool"),\
#                template.index("ZergZergling")], 1), typ=np.int64)
#        if EM:
#            glings_rush_em = expectation_maximization(glings_rush_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        glings_rush_km = k_means(glings_rush_data_int[1], nbiter=nbiterations)
#        glings_rush = r_em(glings_rush_data[1], nbclusters=2, plot=plotR)

        ### Speedlings rush
        features_speedlings = [template.index("ZergZerglingSpeed"),\
                template.index("ZergPool")]
        speedlings_data = filter_out_undef(data.take(features_speedlings, 1))
        if EM:
            speedlings_em = expectation_maximization(speedlings_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedlings = r_em(speedlings_data[1], nbclusters=2, plot=plotR)
        speedlings['features'] = features_speedlings

        ### [Disabled] Hatch first
#        fast_exp_data = filter_out_undef(data.take([\
#                template.index("ZergSecondHatch"),\
#                template.index("ZergPool")], 1))
#        if EM:
#            fast_exp_em = expectation_maximization(fast_exp_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        fast_exp = r_em(fast_exp_data[1], nbclusters=2, plot=plotR)

        ### Fast mutas
        features_fast_mutas = [template.index("ZergMutalisk")]
        fast_mutas_data = filter_out_undef(data.take(features_fast_mutas, 1))
        if kmeans:
            fast_mutas_data_int = filter_out_undef(data.take([\
                    template.index("ZergMutalisk")], 1), typ=np.int64)
            fast_mutas_km = k_means(fast_mutas_data_int[1], nbiter=nbiterations)
        if EM:
            fast_mutas_em = expectation_maximization(fast_mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        fast_mutas = r_em(fast_mutas_data[1], nbclusters=2, plot=plotR)
        fast_mutas['features'] = features_fast_mutas

        ### 3 Hatch mutas / Mass mutas
        features_mutas = [template.index("ZergThirdHatch"),\
                template.index("ZergMutalisk")]
        mutas_data = filter_out_undef(data.take(features_mutas, 1))
        if EM:
            mutas_em = expectation_maximization(mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        mutas = r_em(mutas_data[1], nbclusters=2, plot=plotR)
        mutas['features'] = features_mutas

        ### Fast Lurkers (Third hatch should be late)
        features_lurkers = [template.index("ZergLurker")]
        lurkers_data = filter_out_undef(data.take(features_lurkers, 1))
        if EM:
            lurkers_em = expectation_maximization(lurkers_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        lurkers = r_em(lurkers_data[1], nbclusters=2, plot=plotR)
        lurkers['features'] = features_lurkers

        ### Mass hydras
        features_hydras = [template.index("ZergHydra"),\
                template.index("ZergHydraSpeed"),\
                template.index("ZergHydraRange")]
                #template.index("ZergThirdHatch")] # TODO remove 3rd hatch?
        hydras_data = filter_out_undef(data.take(features_hydras, 1))
        if EM:
            hydras_em = expectation_maximization(hydras_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        hydras = r_em(hydras_data[1], nbclusters=2, plot=plotR)
        hydras['features'] = features_hydras

        speedlings['name'] = "speedlings"
        fast_mutas['name'] = "fast_mutas"
        mutas['name'] = "mutas"
        lurkers['name'] = "lurkers"
        hydras['name'] = "hydras"
        if plotM:
            #glings_rush['name'] = "glings_rush"
            #print glings_rush
            #plot(glings_rush, glings_rush_data[1])
            print speedlings
            plot(speedlings, speedlings_data[1])
            #fast_exp['name'] = "fast_exp"
            #print fast_exp
            #plot(fast_exp, fast_exp_data[1])
            print fast_mutas
            plot(fast_mutas, fast_mutas_data[1])
            print mutas
            plot(mutas, mutas_data[1])
            print lurkers
            plot(lurkers, lurkers_data[1])
            print hydras
            plot(hydras, hydras_data[1])

        write_arff(template, annotate(datalist,\
                (speedlings_data, speedlings), (fast_mutas_data, fast_mutas),\
                (mutas_data, mutas), (lurkers_data, lurkers),\
                (hydras_data, hydras)), "my"+sys.argv[1])
        
