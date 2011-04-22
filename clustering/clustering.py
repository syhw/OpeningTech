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

def k_means(t, nbclusters=2, nbiter=3, medoids=False, soft=True, beta=1.0,\
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
    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    tmpdist = np.ndarray([nbobs,nbclusters], np.float64) # distance obs<->clust
    tmpresp = np.ndarray([nbobs,nbclusters], np.float64) # responsability o<->c
    # iterate for the best quality
    for i in range(nbiter):
        clusters = [[] for c in range(nbclusters)]
        # Step 1: place nbclusters seeds for each features
        centroids = [np.array([random.uniform(min_max[f][0], min_max[f][1])\
                for f in range(nbfeatures)], np.float64)\
                for c in range(nbclusters)]
        old_centroids = [np.array([-1 for f in range(nbfeatures)], np.float64)\
                for c in range(nbclusters)] # should not be init, TODO
        new_sum = sum([distance(centroids[c], old_centroids[c])\
                for c in range(nbclusters)])
        old_sum = sys.maxint
        np.seterr(invalid='raise')
        # iterate until convergence
        while new_sum < old_sum :
            old_centroids = copy.deepcopy(centroids)
            print "k-means iteration, old and new sum: ", old_sum, new_sum
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
                    tmpresp[o,:] /= sum(tmpresp[o,:])
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
                        mean /= sum(tmpresp[:,c])
                    else:
                        for o in clusters[c]:
                            mean += t[o,:]
                        mean = map(lambda x: x/len(clusters[c]), mean)
                    centroids[c] = np.array(mean, np.float64)
            print centroids
            new_sum = sum([distance(centroids[c], old_centroids[c])\
                    for c in range(nbclusters)])
        if soft:
            for o in range(nbobs):
                clusters[tmpdist[o,:].argmin()].append(o)
        quality = sum([sum([tmpdist[o][c] for o in clusters[c]])\
                /(len(clusters[c])+1) for c in range(nbclusters)])
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['centroids'] = centroids
            result['clusters'] = clusters
    return result

def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,\
        distance=lambda x,y: np.linalg.norm(x-y), epsilon=0.01):
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
        xmt = np.matrix((x-m)).transpose()
        for i in range(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*math.sqrt(np.linalg.det(s))\
                *math.exp(-0.5*(xm*sinv*xmt))

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
    ### /Normalization

    print t
    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(clust|obs)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(obs|clust)
    # iterate for the best quality
    for i in range(nbiter):
        # Step 1: draw nbclusters sets of parameters
        params = [{'mu': np.array(\
                [random.uniform(min_max[f][0], min_max[f][1])\
                for f in range(nbfeatures)], np.float64),\
                'sigma': np.matrix(np.diag(\
                [abs(random.gauss(min_max[f][1]/2.0,min_max[f][1]))\
                for f in range(nbfeatures)])),\
                'proba': 1.0/nbclusters}\
                for c in range(nbclusters)]
        old_log_estimate = 0.0 # init, not true/real
        log_estimate = epsilon+1.0 # init, not true/real
        while (abs(log_estimate - old_log_estimate) > epsilon):
            restart = False
            old_log_estimate = log_estimate
            # Step 2: compute P(Cluster|obs) for each observations
            for o in range(nbobs):
                for c in range(nbclusters):
                    # Px[o,c] = P(x|c)
                    Px[o,c] = pnorm(t[o,:],\
                            params[c]['mu'], params[c]['sigma'])
                    #print ">>>> Px[", o, ", ", c, "]: ", Px[o,c]
                    #print ">>>> mu: ", params[c]['mu']
                    #print ">>>> sigma: ", params[c]['sigma']
                    # Pclust[o,c] = P(c|x)
                    Pclust[o,c] = Px[o,c]*params[c]['proba']
            for o in range(nbobs):
                tmpSum = 0.0
                for c in range(nbclusters):
                    tmpSum += params[c]['proba']*Px[o,c]
                #print "Px: ", Px
                print tmpSum
                Pclust[o,:] /= tmpSum
            # Step 3: update the parameters (sets of mu, sigma, proba)
            for c in range(nbclusters):
                tmpSum = sum(Pclust[:,c])
                params[c]['proba'] = tmpSum/nbobs
                if params[c]['proba'] == 0.0:
                    restart = True
                    break
                m = np.zeros(nbfeatures, np.float64)
                for o in range(nbobs):
                    m += t[o,:]*Pclust[o,c]
                print tmpSum
                params[c]['mu'] = m/tmpSum
                print "%%%%%%%%%%%%%%%%: ", params[c]['mu']
                s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
                for o in range(nbobs):
                    diag = Pclust[o,c]*((t[o,:]-params[c]['mu'])*\
                            (t[o,:]-params[c]['mu']).transpose())
                    for i in range(len(s)):
                        s[i,i] += diag[i]
                print tmpSum
                params[c]['sigma'] = s/tmpSum
                print "################: ", params[c]['sigma']
            if restart:
                break
            # Step 4: compute the log estimate
#            restart = False
#            for c in range(nbclusters):
#                if params[c]['proba'] == 0.0:
#                    restart = True
#            if restart:
#                break
            log_estimate = sum([math.log(sum(\
                    [Px[o,c]*params[c]['proba'] for c in range(nbclusters)]))\
                    for o in range(nbobs)])
            print "log est: ", log_estimate
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

def filter_out_undef(tab):
    def not_undef(t):
        for e in t:
            if e < 0:
                return False
        return True
    indices = []
    tmp = []
    for i in range(len(tab)):
        if not_undef(tab[i]):
            indices.append(i)
            tmp.append(tab[i])
    return (indices, np.array(tmp, np.int64))

def plot(clusters, data, title='', gaussians=[]):
    ax = pl.subplot(111)
    xy = [[data[i,j] for i in clusters[0]] for j in range(len(data[0]))]
    ax.scatter(xy[0], xy[len(data[0])-1],\
            s=40, c='b', marker='s', edgecolors='none')
    xy = [[data[i,j] for i in clusters[1]] for j in range(len(data[0]))]
    ax.scatter(xy[0], xy[len(data[0])-1],\
            s=40, c='r', marker='s', edgecolors='none')
    Z = []
    for g in gaussians:
        delta = 0.001
        x = pl.arange(0.0, 1.0, delta)
        y = pl.arange(0.0, 1.0, delta)
        X,Y = pl.meshgrid(x, y)
        Z.append(pl.bivariate_normal(X, Y, float(g['sigma'][0,0]),\
                float(g['sigma'][1,1]),\
                float(g['mu'][0]), float(g['mu'][1])))
    ZZ = Z[0] + Z[1]
    cmap = pl.cm.get_cmap('jet', 10)    # 10 discrete colors
    ranges = [min([data[:,i].min() for i in range(data.shape[1])]),\
            max([data[:,i].max() for i in range(data.shape[1])]),\
            min([data[i,:].min() for i in range(data.shape[0])]),\
            max([data[i,:].max() for i in range(data.shape[0])])]
    ax.axis(ranges)
    ax.imshow(Z[0], cmap=cmap, interpolation='bilinear', origin='lower',\
            extent=ranges)
    ax.imshow(Z[1], cmap=cmap, interpolation='bilinear', origin='lower',\
            extent=ranges)
    pl.title(title)
    pl.grid(True)
    pl.show()

if __name__ == "__main__":
    if sys.argv[1] == "test":
        #t_data = np.array([[1,1],[0,1],[1,0],[10,10],[11,9],\
        #                [11,12],[9,9],[1,1]], np.float64)
        #t_data = np.array([[1,1],[11,11]], np.float64)
        t_data = np.array([[1,2],[1,3],[1,4],[2,1],\
                        [3,1],[4,1]], np.float64)
        #t1 = k_means(t_data, nbiter=10)
        #print t1
        #plot(t1["clusters"],t_data, "test k-means")
        t2 = expectation_maximization(t_data, nbiter=100, normalize=False)
        print t2
        plot(t2["clusters"],t_data, "test EM", t2['params'])

        sys.exit(0)
    nbiterations = 2 # TODO 100 when clustering for real
    (template, datalist) = parse(open(sys.argv[1]))
    # ndarray([#lines, #columns], type) and here #columns without label/string
    data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.int64)
    # transform the kind & dynamic python list into a static numpy.ndarray
    for i in range(len(datalist)):
        for j in range(len(datalist[0]) - 1):
            data[i][j] = datalist[i][j]

    race = sys.argv[1].split('_')[1][0] # race that performs the openings
    matchup = sys.argv[1][5]            # race against which it performs

    ### Fast DT
    fast_dt_data = filter_out_undef(data.take(\
            [template.index("ProtossDarkTemplar")], 1))
    fast_dt = k_means(fast_dt_data[1], nbiter=nbiterations,\
            distance = lambda x,y: abs(x-y))
    ### Fast Expand
    fast_exp_data = filter_out_undef(data.take(\
            [template.index("ProtossFirstExpansion")], 1))
    fast_exp = k_means(fast_exp_data[1], nbiter=nbiterations,\
            distance = lambda x,y: abs(x-y))
    ### Reaver Drop
    reaver_drop_data = filter_out_undef(data.take([\
            template.index("ProtossShuttle"), template.index("ProtossReavor")\
            ], 1))
    reaver_drop = k_means(reaver_drop_data[1], nbiter=nbiterations)
    #reaver_drop = expectation_maximization(reaver_drop_data[1], nbiter=nbiterations)

    ### Cannon Rush
    cannon_rush_data = filter_out_undef(data.take([\
            template.index("ProtossForge"), template.index("ProtossCannon")\
            ], 1))
#    cannon_rush = k_means(cannon_rush_data[1], nbiter=nbiterations)

    ### +1 SpeedZeal
    speedzeal_data = filter_out_undef(data.take([\
            template.index("ProtossGroundWeapons1"),\
            template.index("ProtossLegs")\
            ], 1))
#    speedzeal = k_means(speedzeal_data[1], nbiter=nbiterations)

    ### Nony opening
    nony_data = filter_out_undef(data.take([\
            template.index("ProtossRange"),\
            template.index("ProtossSecondGatway")\
            ], 1))
#    nony = k_means(nony_data[1], nbiter=nbiterations)

    ### Corsair opening
    corsair_data = filter_out_undef(data.take([\
            template.index("ProtossCorsair")], 1))
#    corsair = k_means(corsair_data[1], nbiter=nbiterations)

    print fast_dt
    plot(fast_dt["clusters"], fast_dt_data[1], "fast dark templar")

    print fast_exp
    plot(fast_exp["clusters"], fast_exp_data[1], "fast expand")

    print reaver_drop
    plot(reaver_drop["clusters"], reaver_drop_data[1], "reaver drop")

    print cannon_rush
    plot(cannon_rush["clusters"],cannon_rush_data[1], "cannon rush")

    print speedzeal
    plot(speedzeal["clusters"],speedzeal_data[1], "speedzeal")

    print nony
    plot(nony["clusters"],nony_data[1], "nony")

    print corsair
    plot(corsair["clusters"], corsair_data[1], "corsair")
