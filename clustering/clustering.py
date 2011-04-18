#!/opt/local/bin/python
# Playschool kode disclaimer! Completely untested!

# Python License 2.0.1 http://www.python.org/download/releases/2.0.1/license/
# Copyright 2011 Gabriel Synnaeve

import sys, random, copy, math
try:
    import numpy as np
except:
    print "You need numpy"
import pylab as pl

def k_means(t, nbclusters=2, nbiter=3, medoids=False, soft=True, beta=0.01,\
        #distance=lambda x,y: np.linalg.norm(x-y)):
        distance=lambda x,y: math.sqrt(np.dot(x-y,(x-y).conj()))):
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
                for f in range(nbfeatures)], np.int64)\
                for c in range(nbclusters)]
        old_centroids = [np.array([-1 for f in range(nbfeatures)], np.int64)\
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
                        tmpresp[o,c] = math.exp(-1 * beta * tmpdist[o,c])
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
                    mean = np.array([0 for i in range(nbfeatures)], np.int64)
                    if soft:
                        for o in range(nbobs):
                            mean += tmpresp[o,c] * t[o,:]
                        mean /= sum(tmpresp[:,c])
                    else:
                        for o in clusters[c]:
                            mean += t[o,:]
                        mean = map(lambda x: x/len(clusters[c]), mean)
                    centroids[c] = np.array(mean, np.int64)
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

def expectation_maximization(t, nbclusters=2, nbiter=10,\
        distance=lambda x,y: np.linalg.norm(x-y), epsilon=1):
    ### TODO
    return True

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

def plot(clusters, data):
    ax = pl.subplot(111)
    xy = [[data[i,0] for i in clusters[0]] for j in range(len(data[0]))]
    ax.scatter(xy[0], xy[len(data[0])-1],\
            s=40, c='b', marker='s', edgecolors='none')
    xy = [[data[i,0] for i in clusters[1]] for j in range(len(data[0]))]
    ax.scatter(xy[0], xy[len(data[0])-1],\
            s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

if __name__ == "__main__":
    nbiterations = 2 # TODO 100 when clustering for real
    (template, datalist) = parse(open(sys.argv[1]))
    # ndarray([#lines, #columns], type) and here #columns without label/string
    data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.int64)
    # transform the kind & dynamic python list into a static numpy.ndarray
    for i in range(len(datalist)):
        for j in range(len(datalist[0]) - 1):
            data[i][j] = datalist[i][j]
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

    ### Cannon Rush
    cannon_rush_data = filter_out_undef(data.take([\
            template.index("ProtossForge"), template.index("ProtossCannon")\
            ], 1))
    cannon_rush = k_means(cannon_rush_data[1], nbiter=nbiterations)

    ### +1 SpeedZeal
    speedzeal_data = filter_out_undef(data.take([\
            template.index("ProtossGroundWeapons1"), template.index("ProtossLegs")\
            ], 1))
    speedzeal = k_means(speedzeal_data[1], nbiter=nbiterations)

    ### Standard
    standard_data = filter_out_undef(data.take([\
            template.index("ProtossRange"), template.index("ProtossObs")\
            ], 1))
    standard = k_means(standard_data[1], nbiter=nbiterations)

    print fast_dt
    plot(fast_dt["clusters"], fast_dt_data[1])

    print fast_exp
    ax = pl.subplot(111)
    xs = [fast_exp_data[1][i,0] for i in fast_exp["clusters"][0]]
    ys = [fast_exp_data[1][i,0] for i in fast_exp["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [fast_exp_data[1][i,0] for i in fast_exp["clusters"][1]]
    ys = [fast_exp_data[1][i,0] for i in fast_exp["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

    print reaver_drop
    ax = pl.subplot(111)
    xs = [reaver_drop_data[1][i,0] for i in reaver_drop["clusters"][0]]
    ys = [reaver_drop_data[1][i,1] for i in reaver_drop["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [reaver_drop_data[1][i,0] for i in reaver_drop["clusters"][1]]
    ys = [reaver_drop_data[1][i,1] for i in reaver_drop["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

    print cannon_rush
    ax = pl.subplot(111)
    xs = [cannon_rush_data[1][i,0] for i in cannon_rush["clusters"][0]]
    ys = [cannon_rush_data[1][i,1] for i in cannon_rush["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [cannon_rush_data[1][i,0] for i in cannon_rush["clusters"][1]]
    ys = [cannon_rush_data[1][i,1] for i in cannon_rush["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

    print cannon_rush
    ax = pl.subplot(111)
    xs = [cannon_rush_data[1][i,0] for i in cannon_rush["clusters"][0]]
    ys = [cannon_rush_data[1][i,1] for i in cannon_rush["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [cannon_rush_data[1][i,0] for i in cannon_rush["clusters"][1]]
    ys = [cannon_rush_data[1][i,1] for i in cannon_rush["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

    print standard
    ax = pl.subplot(111)
    xs = [standard_data[1][i,0] for i in standard["clusters"][0]]
    ys = [standard_data[1][i,1] for i in standard["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [standard_data[1][i,0] for i in standard["clusters"][1]]
    ys = [standard_data[1][i,1] for i in standard["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

