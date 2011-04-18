#!/opt/local/bin/python
# Playschool kode disclaimer! Completely untested!
# t[first_line:last_line:increment,first_column:last_column:increment]

# Python License 2.0.1 http://www.python.org/download/releases/2.0.1/license/
# Copyright 2011 Gabriel Synnaeve

import sys, random, copy, math
try:
    import numpy as np
except:
    print "You need numpy"

def k_means(t, nbclusters=2, nbiter=10, medoids=False,\
        #distance=lambda x,y: np.linalg.norm(x-y),
        distance=lambda x,y: math.sqrt(np.dot(x-y,(x-y).conj())), 
        responsability=lambda x,c: 1):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters/centroids 
    'nbiter' is the number of iterations
    'medoids' tells is we use the medoids or centroids method
    'distance' is the function to use for comparing observations

    Overview of the algorithm:
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
    tmpdist = np.ndarray([nbobs,nbclusters], np.float64)
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
        old_sum = 1000000000000 # TODO clean
        np.seterr(invalid='raise')
        # iterate until convergence
        while new_sum < old_sum :
            old_centroids = copy.deepcopy(centroids)
            print "k-means iteration, old and new sum: ", old_sum, new_sum
            old_sum = new_sum
            for c in range(nbclusters):
                clusters[c] = []
            # distance to all centroids/medoids for all observations
            for c in range(nbclusters):
                for o in range(nbobs):
                    tmpdist[o,c] = distance(centroids[c], t[o,:])
            # Step 2: assign each object to the closest centroid
            for o in range(nbobs):
                clusters[tmpdist[o,:].argmin()].append(o)
            # Step 3: recalculate the positions of the nbclusters centroids
            for c in range(nbclusters):
                if medoids:
                    tmpmin = 1000000000000 # TODO clean #
                    argmin = 0                         #
                    for o in clusters[c]:              #
                        if tmpdist[o,c] < tmpmin:      #
                            tmpmin = tmpdist[o,c]      #
                            argmin = o                 #
                    centroids[c] = t[argmin,:]
                else:
                    mean = np.array([0 for i in range(nbfeatures)], np.int64)
                    for o in clusters[c]:
                        mean += t[o,:]
                    mean = map(lambda x: x/len(clusters[c]), mean)
                    centroids[c] = np.array(mean, np.int64)
            new_sum = sum([distance(centroids[c], old_centroids[c])\
                    for c in range(nbclusters)])
        quality = sum([sum([tmpdist[o][c] for o in clusters[c]])\
                /(len(clusters[c])+1) for c in range(nbclusters)])
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['centroids'] = centroids
            result['clusters'] = clusters
    return result

def expectation_maximization(t, nbclusters=2, nbiter=100,\
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
                    tmp.append(int(elem))
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

if __name__ == "__main__":
    (template, datalist) = parse(open(sys.argv[1]))
    # ndarray([#lines, #columns], type) and here #columns without label/string
    data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.int64)
    # transform the kind & dynamic python list into a static numpy.ndarray
    for i in range(len(datalist)):
        for j in range(len(datalist[0]) - 1):
            data[i][j] = datalist[i][j]
#    print k_means(filter_out_undef(data.take(\
#            [template.index("ProtossDarkTemplar")], 1))[1],\
#            distance = lambda x,y: abs(x-y))
#    print k_means(filter_out_undef(data.take(\
#            [template.index("ProtossFirstExpansion")], 1))[1],\
#            distance = lambda x,y: abs(x-y))
    fast_dt_data = filter_out_undef(data.take([\
            template.index("ProtossShuttle"), template.index("ProtossReavor")\
            ], 1))
    fast_dt = k_means(fast_dt_data[1])
    print fast_dt
    import pylab as pl
    ax = pl.subplot(111)
    xs = [fast_dt_data[1][i,0] for i in fast_dt["clusters"][0]]
    ys = [fast_dt_data[1][i,1] for i in fast_dt["clusters"][0]]
    ax.scatter(xs, ys, s=40, c='b', marker='s', edgecolors='none')
    xs = [fast_dt_data[1][i,0] for i in fast_dt["clusters"][1]]
    ys = [fast_dt_data[1][i,1] for i in fast_dt["clusters"][1]]
    ax.scatter(xs, ys, s=40, c='r', marker='s', edgecolors='none')
    pl.grid(True)
    pl.show()

