#!/opt/local/bin/python
# vim: set fileencoding=utf-8 :
# Playschool kode disclaimer! Completely untested!

# Python License 2.0.1 http://www.python.org/download/releases/2.0.1/license/
# Copyright 2011 Gabriel Synnaeve


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
    def pnorm(x, m, s):
        #return (1/(math.sqrt(2*math.pi*s**2)))\
        #        * math.exp((x-m)**2/(2*s**2))
        xmt = np.matrix(x-m).transpose()
        for i in range(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
                *math.exp(-0.5*(xm*sinv*xmt))

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
                    tmp.append(int(elem)/24) 
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

def plot(clusters, data, title='', gaussians=[], separate_plots=False):
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

    (template, datalist) = parse(open(sys.argv[1]))
    # ndarray([#lines, #columns], type) and here #columns without label/string
    data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.float64)
    # transform the kind & dynamic python list into a static numpy.ndarray
    for i in range(len(datalist)):
        for j in range(len(datalist[0]) - 1):
            data[i][j] = datalist[i][j]

    race = sys.argv[1].split('_')[1][0] # race that performs the openings
    matchup = sys.argv[1][5]            # race against which it performs

    ### FULL
    #full_data = filter_out_undef(data.take(\
    #        [template.index("ProtossPylon"), template.index("ProtossGateway"),\
    #        template.index("ProtossCore"), template.index("ProtossSecondPylon")]\
    #        , 1))
    #full = r_em(full_data[1], plot=True)

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
        # - cannon rush

        ### 2 gates rush opening
        if kmeans:
            two_gates_data_int = filter_out_undef(data.take([\
                    template.index("ProtossGateway"),\
                    template.index("ProtossSecondGatway")\
                    ], 1), typ=np.int64)
            two_gates_km = k_means(two_gates_data_int[1], nbiter=nbiterations)
        two_gates_data = filter_out_undef(data.take([\
                template.index("ProtossGateway"),\
                template.index("ProtossSecondGatway")], 1))
        if EM:
            two_gates_em = expectation_maximization(two_gates_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_gates = r_em(two_gates_data[1], nbclusters=2, plot=plotR)

        ### Fast DT
        if kmeans:
            fast_dt_data_int = filter_out_undef(data.take(\
                    [template.index("ProtossDarkTemplar")], 1), typ=np.int64)
            fast_dt_km = k_means(fast_dt_data_int[1], nbiter=nbiterations,\
                    distance = lambda x,y: abs(x-y))
        fast_dt_data = filter_out_undef(data.take(\
                [template.index("ProtossDarkTemplar")], 1))
        if EM:
            fast_dt_em = expectation_maximization(fast_dt_data[1],\
                    nbiter=nbiterations, monotony=True, normalize=True)
        fast_dt = r_em(fast_dt_data[1], nbclusters=2, plot=plotR)

        ### Fast Expand
        if kmeans:
            fast_exp_data_int = filter_out_undef(data.take(\
                    [template.index("ProtossFirstExpansion")], 1), typ=np.int64)
            fast_exp_km = k_means(fast_exp_data_int[1], nbiter=nbiterations,\
                    distance = lambda x,y: abs(x-y))
        fast_exp_data = filter_out_undef(data.take(\
                [template.index("ProtossFirstExpansion")], 1))
        if EM:
            fast_exp_em = expectation_maximization(fast_exp_data[1],\
                    nbiter=nbiterations, monotony=True, normalize=True)
        fast_exp = r_em(fast_exp_data[1], nbclusters=2, plot=plotR)

        ### +1 SpeedZeal
        if kmeans:
            speedzeal_data_int = filter_out_undef(data.take([\
                    template.index("ProtossGroundWeapons1"),\
                    template.index("ProtossLegs")\
                    ], 1), typ=np.int64)
            speedzeal_km = k_means(speedzeal_data_int[1], nbiter=nbiterations)
        speedzeal_data = filter_out_undef(data.take([\
                template.index("ProtossGroundWeapons1"),\
                template.index("ProtossLegs")\
                ], 1))
        if EM:
            speedzeal_em = expectation_maximization(speedzeal_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedzeal = r_em(speedzeal_data[1], nbclusters=2, plot=plotR)

        ### Bisu build
        if kmeans:
            bisu_data_int = filter_out_undef(data.take([\
                    template.index("ProtossFirstExpansion"),\
                    template.index("ProtossCorsair"),\
                    template.index("ProtossArchives")\
                    ], 1), typ=np.int64)
            bisu_km = k_means(bisu_data_int[1], nbiter=nbiterations)
        bisu_data = filter_out_undef(data.take([\
                template.index("ProtossFirstExpansion"),\
                template.index("ProtossCorsair"),\
                template.index("ProtossArchives")\
                ], 1))
        if EM:
            bisu_em = expectation_maximization(bisu_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        bisu = r_em(bisu_data[1], nbclusters=2, plot=plotR)

        ### Corsair opening
        if kmeans:
            corsair_data_int = filter_out_undef(data.take([\
                    template.index("ProtossCorsair")], 1), typ=np.int64)
            corsair_km = k_means(corsair_data_int[1], nbiter=nbiterations)
        corsair_data = filter_out_undef(data.take([\
                template.index("ProtossCorsair")], 1))
        if EM:
            corsair_em = expectation_maximization(corsair_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        corsair = r_em(corsair_data[1], nbclusters=2, plot=plotR)

        ### Nony opening
        if kmeans:
            nony_data_int = filter_out_undef(data.take([\
                    template.index("ProtossRange"),\
                    template.index("ProtossSecondGatway"),\
                    template.index("ProtossGoon")
                    ], 1), typ=np.int64)
            nony_km = k_means(nony_data_int[1], nbiter=nbiterations)
        nony_data = filter_out_undef(data.take([\
                template.index("ProtossRange"),\
                template.index("ProtossSecondGatway"),\
                template.index("ProtossGoon")
                ], 1))
        if EM:
            nony_em = expectation_maximization(nony_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        nony = r_em(nony_data[1], nbclusters=2, plot=plotR)


        ### Reaver Drop
        if kmeans:
            reaver_drop_data_int = filter_out_undef(data.take([\
                    template.index("ProtossShuttle"), template.index("ProtossReavor")\
                    ], 1), typ=np.int64)
            reaver_drop_km = k_means(reaver_drop_data_int[1], nbiter=nbiterations)
        reaver_drop_data = filter_out_undef(data.take([\
                template.index("ProtossShuttle"), template.index("ProtossReavor")\
                ], 1))
        if EM:
            reaver_drop_em = expectation_maximization(reaver_drop_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        reaver_drop = r_em(reaver_drop_data[1], nbclusters=2, plot=plotR)

        ### Cannon Rush
        if kmeans:
            cannon_rush_data_int = filter_out_undef(data.take([\
                    template.index("ProtossForge"), template.index("ProtossCannon")\
                    ], 1), typ=np.int64)
            cannon_rush_km = k_means(cannon_rush_data_int[1], nbiter=nbiterations)
        cannon_rush_data = filter_out_undef(data.take([\
                template.index("ProtossForge"), template.index("ProtossCannon")\
                ], 1))
        if EM:
            cannon_rush_em = expectation_maximization(cannon_rush_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        cannon_rush = r_em(cannon_rush_data[1], nbclusters=2, plot=plotR)

        print two_gates
        plot(two_gates["clusters"], two_gates_data[1], "two_gates",\
                two_gates["params"])
        print fast_dt
        plot(fast_dt["clusters"], fast_dt_data[1], "fast dark templar")
        print fast_exp
        plot(fast_exp["clusters"], fast_exp_data[1], "fast expand")
        print speedzeal
        plot(speedzeal["clusters"],speedzeal_data[1], "speedzeal",\
                speedzeal["params"])
        print bisu
        plot(bisu["clusters"], bisu_data[1], "bisu", bisu["params"])
        print corsair
        plot(corsair["clusters"], corsair_data[1], "corsair", corsair["params"])
        print nony
        plot(nony["clusters"],nony_data[1], "nony", nony["params"])
        print reaver_drop
        plot(reaver_drop["clusters"], reaver_drop_data[1], "reaver drop",\
                reaver_drop["params"])
        print cannon_rush
        plot(cannon_rush["clusters"],cannon_rush_data[1], "cannon rush",\
                cannon_rush["params"])

    if race == "T":
        # Main openings:
        # - BBS rush (rax/rax/supply) and other styles of 8 rax
        # - 1 Rax FE or 2 Rax FE
        # - Siege Expand (facto into siege + expand)
        # - 2 Factories (aggressive push / nada style)
        # - Vultures harass
        # - Fast Wraith cloak

        ### BBS rush
        bbs_data = filter_out_undef(data.take([\
                template.index("TerranBarracks"),\
                template.index("TerranSecondBarracks"),\
                template.index("TerranDepot")], 1))
        if EM:
            bbs_em = expectation_maximization(bbs_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        bbs = r_em(bbs_data[1], nbclusters=2, plot=plotR)

        ### 1/2 Rax FE
        rax_fe_data = filter_out_undef(data.take([\
                template.index("TerranBarracks"),\
                template.index("TerranExpansion")], 1))
        if EM:
            rax_fe_em = expectation_maximization(rax_fe_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        rax_fe = r_em(rax_fe_data[1], nbclusters=2, plot=plotR)

        ### Siege Expand
        siege_exp_data = filter_out_undef(data.take([\
                template.index("TerranSiege"),\
                template.index("TerranExpansion")], 1))
        if EM:
            siege_exp_em = expectation_maximization(siege_exp_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        siege_exp = r_em(siege_exp_data[1], nbclusters=2, plot=plotR)

        ### 2 Facto
        two_facto_data = filter_out_undef(data.take([\
                template.index("TerranFactory"),\
                template.index("TerranSecondFactory"),\
                template.index("TerranTank")], 1))
        if EM:
            two_facto_em = expectation_maximization(two_facto_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_facto = r_em(two_facto_data[1], nbclusters=2, plot=plotR)

        ### Vultures harass
        vultures_data = filter_out_undef(data.take([\
                template.index("TerranVulture"),\
                template.index("TerranMines")],1))
        if EM:
            vultures_em = expectation_maximization(vultures_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        vultures = r_em(vultures_data[1], nbclusters=2, plot=plotR)

        ### Fast Wraith cloak
        cloak_data = filter_out_undef(data.take([\
                template.index("TerranWraith"),\
                template.index("TerranCloak")], 1))
        if EM:
            cloak_em = expectation_maximization(cloak_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        cloak = r_em(cloak_data[1], nbclusters=2, plot=plotR)

        print bbs
        plot(bbs["clusters"], bbs_data[1], "bbs", bbs["params"])
        print rax_fe
        plot(rax_fe["clusters"], rax_fe_data[1], "rax_fe", rax_fe["params"])
        print siege_exp
        plot(siege_exp["clusters"], siege_exp_data[1], "siege_exp",\
                siege_exp["params"])
        print two_facto
        plot(two_facto["clusters"], two_facto_data[1], "two_facto",\
                two_facto["params"])
        print vultures
        plot(vultures["clusters"], vultures_data[1], "vultures",\
                vultures["params"])
        print cloak
        plot(cloak["clusters"], cloak_data[1], "cloak", cloak["params"])

    if race == "Z":
        # Main openings:
        # - 4-6 pools very early glings rush
        # - ~9pool/9speed speedlings rush
        # - any kind of fast expand (overpool, 12 hatch...) into:
        #   * fast mutas (2 hatches muta, or even 1 hatch mutas in ZvZ
        #   * mass mutas (3 hatches, or more, into mutas)
        #   * fast lurkers (3 hatch lurker)
        #   * hydras push/drop

        ### Very early zerglings rush (4 to 6 pool)
        glings_rush_data = filter_out_undef(data.take([\
                template.index("ZergPool"),\
                template.index("ZergZergling")], 1))
        if EM:
            glings_rush_em = expectation_maximization(glings_rush_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        glings_rush = r_em(glings_rush_data[1], nbclusters=2, plot=plotR)

        ### Speedlings rush
        speedlings_data = filter_out_undef(data.take([\
                template.index("ZergPool"),\
                template.index("ZergZerglingSpeed")], 1))
        if EM:
            speedlings_em = expectation_maximization(speedlings_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedlings = r_em(speedlings_data[1], nbclusters=2, plot=plotR)

        ### [Disabled] Hatch first
#        fast_exp_data = filter_out_undef(data.take([\
#                template.index("ZergSecondHatch"),\
#                template.index("ZergPool")], 1))
#        if EM:
#            fast_exp_em = expectation_maximization(fast_exp_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        fast_exp = r_em(fast_exp_data[1], nbclusters=2, plot=plotR)

        ### Fast mutas
        fast_mutas_data = filter_out_undef(data.take([\
                template.index("ZergMutalisk")], 1))
        if EM:
            fast_mutas_em = expectation_maximization(fast_mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        fast_mutas = r_em(fast_mutas_data[1], nbclusters=2, plot=plotR)

        ### 3 Hatch mutas / Mass mutas
        mutas_data = filter_out_undef(data.take([\
                template.index("ZergSecondHatch"),\
                template.index("ZergThirdHatch"),\
                template.index("ZergMutalisk")], 1))
        if EM:
            mutas_em = expectation_maximization(mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        mutas = r_em(mutas_data[1], nbclusters=2, plot=plotR)

        ### Lurkers
        lurkers_data = filter_out_undef(data.take([\
                template.index("ZergLurker")], 1))
        if EM:
            lurkers_em = expectation_maximization(lurkers_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        lurkers = r_em(lurkers_data[1], nbclusters=2, plot=plotR)

        ### Mass hydras
        hydras_data = filter_out_undef(data.take([\
                template.index("ZergHydra"),\
                template.index("ZergHydraSpeed"),\
                template.index("ZergHydraRange")], 1))
        if EM:
            hydras_em = expectation_maximization(hydras_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        hydras = r_em(hydras_data[1], nbclusters=2, plot=plotR)

        print glings_rush
        plot(glings_rush["clusters"], glings_rush_data[1], "glings_rush",\
                glings_rush["params"])
        print speedlings
        plot(speedlings["clusters"], speedlings_data[1], "speedlings",\
                speedlings["params"])
        #print fast_exp
        #plot(fast_exp["clusters"], fast_exp_data[1], "fast_exp", fast_exp["params"])
        print fast_mutas
        plot(fast_mutas["clusters"], fast_mutas_data[1], "fast_mutas",\
                fast_mutas["params"])
        print mutas
        plot(mutas["clusters"], mutas_data[1], "mutas", mutas["params"])
        print lurkers
        plot(lurkers["clusters"], lurkers_data[1], "lurkers", lurkers["params"])
        print hydras
        plot(hydras["clusters"], hydras_data[1], "hydras", hydras["params"])
