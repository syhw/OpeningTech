#!/opt/local/bin/python
# vim: set fileencoding=utf-8 :
# Playschool kode disclaimer! Completely untested!

# Python License 2.0.1 http://www.python.org/download/releases/2.0.1/license/
# Copyright 2011 Gabriel Synnaeve

# TODO a usage documentation (gather "DOC")
# First feature has a piority: the labels should correspond to some BO 
# attaining the first feature first.
# TODO opening clustering/labeling per match-up (and not race!), 9 cases

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

unique_labeling = True
prio_timing = 'early' # 'late'

def k_means(t, nbclusters=2, nbiter=3, medoids=False, soft=False, beta=200.0,\
        #distance=lambda x,y: np.linalg.norm(x-y),\
        distance=lambda x,y: math.sqrt(np.dot(x-y,(x-y).conj())),\
        responsability=lambda beta,d: math.exp(-1 * beta * d)):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters/centroids 
    'nbiter' is the number of iterations
    'medoids' tells if we use the medoids or centroids method
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
                        print "ERROR: Soft medoids not implemented"
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

def r_em(t, nbclusters=0, plot=False, name=""):
    try:
        from rpy2.robjects import r
        import rpy2.robjects.numpy2ri # auto-translates numpy array to R ones
        rpy2.robjects.numpy2ri.activate()
    except:
        print "ERROR: You can't use 'r_em()' without rpy2 and R+library mclust"
        sys.exit(-1)
    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    result = {}
    r.library("mclust")
    if nbclusters:
        model = r.Mclust(t, G=nbclusters)
        if name == 'templar':
            model = r.Mclust(t, G=nbclusters, modelName="VEV")
    else:
        model = r.Mclust(t)
    if plot:
        print "=============================="
        print "plotting: ", name
        print model
        r.quartz("plot")
        r.plot(model, t)
        print raw_input("type any key to pass")
        print "=============================="
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

def parse(data_file, form='arff'):
    template = []
    t = []
    data = False
    if form == 'arff':
        # arff header @relation, @attribute(s), @data 
        # -> [time_per_attr, ...] list
        for line in data_file:
            if data:
                tmp = []
                for elem in line.split(','):
                    if elem[0] in "0123456789":
                        # Convert to int, and 24 frames per second:
                        # we don't need such a high resolution
                        tmp.append(int(elem))#/24) 
                    else:
                        tmp.append(elem.rstrip('\r\n'))
                t.append(tmp)
            elif '@attribute' in line:
                template.append(line.split(' ')[1])
            elif '@data' in line:
                data = True
        return (template, t)
    elif form == 'txt':
        # [attribute_name time; ...] list
        for line in data_file:
            l = line.rstrip('\r\n').split(' ')
            if len(template) == 0:
                template.extend([l[2*i] for i in range(len(l)/2)])
            tmp = []
            #tmp.extend([l[2*i + 1] for i in range(len(l)/2)])
            l = line.rstrip('\r\n').replace('; ',';').split(';')
            for elem in l:
                if len(elem.split(' ')) > 1:
                    tmp.append(int(elem.split(' ')[1]))
            t.append(tmp)
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

def opening_probabilities(datalist, *args):
#    {'features': [37, 34], 'params': [{'mu': array([ 15754.64988219,  13728.3076984 ]), 'sigma': matrix([[  1.27961928e+08,   1.93321051e+07],
#                [  1.93321051e+07,   1.25958769e+07]]), 'proba': 0.4022966788465614}, {'mu': array([ 9786.01343733,  8517.94454639]), 'sigma': matrix([[ 4046531.0721206 ,  3836400.37897049],
#                            [ 3836400.37897049,  3834854.52409518]]), 'proba': 0.5977033211534386}], 'timing': 'late', 'clusters': [[0, 1, 2, 4, 6, 8, 10, 11, 19, 28, 34, 35, 40, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 60, 71, 72, 73, 74, 75, 77, 80, 83, 90, 91, 92, 93, 96, 98, 99, 100, 102, 110, 111, 112, 113, 114, 116, 121, 122, 127, 134, 136, 139], [3, 5, 7, 9, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 36, 37, 38, 39, 41, 42, 45, 46, 55, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 76, 78, 79, 81, 82, 84, 85, 86, 87, 88, 89, 94, 95, 97, 101, 103, 104, 105, 106, 107, 108, 109, 115, 117, 118, 119, 120, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 135, 137, 138, 140, 141]], 'quality': 'not computed', 'name': 'reaver_drop'#}
    return
    for game in datalist:
        m = -1.0
        op = ""
        for arg in args:
            p = multi_dim_gaussian_likelihood([game[i] for i in arg['features']], arg['params']['mu'], arg[
            if p > m:
                m = p
                op = arg['name']
            

def annotate(data, *args):
    def determine_cluster_ind():
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
        # clusters['clusters'][cind] is the list of the games labeled
        # clusters['name'] in data[1], their indices in data is in data[0]
        ### /!\ /shitty heuristic

    annotations = {}
    annotations['openings'] = []
    annotations['games'] = copy.deepcopy(data)
    annotations['metadata'] = []
    openings_timings = {}
    labelind = len(data[0]) - 1
    # Remove the precedents labels/strategies/openings
    for game in annotations['games']:
        game[labelind] = ''
        annotations['metadata'].append({})
    # Determine which are the labels brought by "clusters"
    for (data, clusters) in args:
        # data[0] are the true indices in data of data[1] (filtered data)
        # clusters['name'] / clusters['clusters'] / clusters['params']
        annotations['openings'].append(clusters['name'])
        openings_timings[clusters['name']] = clusters['timing']
        cind = determine_cluster_ind()
        # Add each label to each game
        for g in clusters['clusters'][cind]:
            annotations['games'][data[0][g]][labelind] += clusters['name']+' '
            if clusters.has_key('params'):
                annotations['metadata'][data[0][g]][clusters['name']] = \
                        (clusters['params'], clusters['features'], cind)
            elif clusters.has_key('centroids'):
                annotations['metadata'][data[0][g]][clusters['name']] = \
                        (clusters['centroids'], clusters['features'], cind)
    # We could return here and print the full labels of replays
    # Make a selection of one label/strategy/opening per replay
    for game in annotations['games']:
        game[labelind].rstrip(' ')
        sp = game[labelind].split(' ')
        if len(sp) == 1 and sp[0] == '':
            game[labelind] == 'unknown'
            continue
        # "else", len(sp) >= 2
        bestlabelp = '' # best label according to the proba of clustering
        bestproba = 0.0 # best probability
        bestlabelt = '' # best label according to the order of appearance
        probat = 0.0    # probability of the label appearing the first
        mintime = 10000000.0 # ;)
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
            # v[1][0] =  # DOC: the 1st feature is always the "most important"
            if game[v[1][0]] < mintime:
                mintime = game[v[1][0]]
                bestlabelt = k
                probat = tmpproba
        if unique_labeling:
            ### Picks the best label or put unknown
            ### label <- first appearing if most probable or not far (10%)
            if bestlabelp == bestlabelt or (probat/bestproba) > (0.9**maxdim):
                game[labelind] = bestlabelt # if probat is at 10% of bestproba
            elif openings_timings[bestlabelt] == prio_timing and\
                    openings_timings[bestlabelp] != prio_timing:
                game[labelind] = bestlabelt
            elif openings_timings[bestlabelp] == prio_timing and\
                    openings_timings[bestlabelt] != prio_timing:
                game[labelind] = bestlabelp
            else:
                #print "MARK: bestlabelt, bestlabelp:", bestlabelt, bestlabelp
                #print 'unknown: ', game
                game[labelind] = 'unknown'
            ### /Picks the best label 
        else:
            ### Filter out the less probable and compose openings (early/late)
            pass
            #print 'ERROR: Non unique labeling not implemented'
            #sys.exit(-1)
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
    plotR = False
    plotM = False

    ### q'n'd
    formating = 'UNKNOWN'
    if (sys.argv[1][-4:] == 'arff'):
        formating = 'arff'
    elif (sys.argv[1][-3:] == 'txt'):
        formating = 'txt'
    else:
        print "unknown input format/extension"
        sys.exit(-1)
    (template, datalist) = parse(open(sys.argv[1]), formating)

    # build data without the "label"/opening/strategy column
    if formating == 'arff':
        data = np.ndarray([len(datalist), len(datalist[0]) - 1], np.float64)
    elif formating == 'txt':
        data = np.ndarray([len(datalist), len(datalist[0])], np.float64)
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
        # - FE into (no "FE" opening)
        #   * +1 speed (legs) zealot push
        #   * templar tech: bisu build (sair/templar) or sair/DT
        #   * sair/reaver TODO
        # - gate/core/gate/gate goons (nony)
        # - reaver drop 
        # - cannon rush [Disabled: can only scout it]

        ### 2 gates rush opening
        features_two_gates = []
        if formating == 'arff':
            features_two_gates = [template.index("ProtossSecondGatway"),\
                    template.index("ProtossGateway"),\
                    template.index("ProtossZealot")]
        elif formating == 'txt':
            features_two_gates = [template.index("Protoss_Gateway2"),\
                    template.index("Protoss_Gateway"),\
                    template.index("Protoss_Zealot")]
        if kmeans:
            two_gates_data_int = filter_out_undef(data.take(\
                    features_two_gates, 1), typ=np.int64)
            two_gates_km = k_means(two_gates_data_int[1], nbiter=nbiterations)
        two_gates_data = filter_out_undef(data.take(features_two_gates, 1))
        if EM:
            two_gates_em = expectation_maximization(two_gates_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_gates = r_em(two_gates_data[1], nbclusters=2, plot=plotR, name="two_gates")
        two_gates['features'] = features_two_gates
        two_gates['timing'] = 'early'

        ### Fast DT
        features_fast_dt = []
        if formating == 'arff':
            features_fast_dt = [template.index("ProtossDarkTemplar")]
        elif formating == 'txt':
            features_fast_dt = [template.index("Protoss_Dark_Templar")]
        if kmeans:
            fast_dt_data_int = filter_out_undef(data.take(\
                    features_fast_dt, 1), typ=np.int64)
            fast_dt_km = k_means(fast_dt_data_int[1], nbiter=nbiterations,\
                    distance = lambda x,y: abs(x-y))
        fast_dt_data = filter_out_undef(data.take(features_fast_dt,1))
        if EM:
            fast_dt_em = expectation_maximization(fast_dt_data[1],\
                    nbiter=nbiterations, monotony=True, normalize=True)
        fast_dt = r_em(fast_dt_data[1], nbclusters=2, plot=plotR, name="fast_dt")
        fast_dt['features'] = features_fast_dt
        fast_dt['timing'] = 'late'

        ### +1 SpeedZeal
        features_speedzeal = []
        if formating == 'arff':
            features_speedzeal = [template.index("ProtossLegs"),\
                    template.index("ProtossGroundWeapons1")]
        elif formating == 'txt':
            features_speedzeal = [template.index("Protoss_Zealot_Speed"),\
                    template.index("Protoss_Ground_Weapons")]
        if kmeans:
            speedzeal_data_int = filter_out_undef(data.take(\
                    features_speedzeal, 1), typ=np.int64)
            speedzeal_km = k_means(speedzeal_data_int[1], nbiter=nbiterations)
        speedzeal_data = filter_out_undef(data.take(features_speedzeal, 1))
        if EM:
            speedzeal_em = expectation_maximization(speedzeal_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedzeal = r_em(speedzeal_data[1], nbclusters=2, plot=plotR, name="speedzeal")
        speedzeal['features'] = features_speedzeal
        speedzeal['timing'] = 'early'

        ### Fast templars
        features_templar = []
        if formating == 'arff':
            features_templar = [template.index("ProtossStorm"),\
                        template.index("ProtossTemplar")]
        elif formating == 'txt':
            features_templar = [template.index("Protoss_Psionic_Storm"),\
                        template.index("Protoss_High_Templar")]
        if kmeans:
            templar_data_int = filter_out_undef(data.take(\
                    features_templar, 1), typ=np.int64)
            templar_km = k_means(templar_data_int[1], nbiter=nbiterations)
        templar_data = filter_out_undef(data.take(features_templar, 1))
        if EM:
            templar_em = expectation_maximization(templar_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        templar = r_em(templar_data[1], nbclusters=2, plot=plotR, name="templar")
        templar['features'] = features_templar
        templar['timing'] = 'late'

        ### Corsair opening
        features_corsair = []
        if formating == 'arff':
            features_corsair = [template.index("ProtossCorsair")]
        elif formating == 'txt':
            features_corsair = [template.index("Protoss_Corsair")]
        if kmeans:
            corsair_data_int = filter_out_undef(data.take(\
                    features_corsair, 1), typ=np.int64)
            corsair_km = k_means(corsair_data_int[1], nbiter=nbiterations)
        corsair_data = filter_out_undef(data.take(features_corsair, 1))
        if EM:
            corsair_em = expectation_maximization(corsair_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        corsair = r_em(corsair_data[1], nbclusters=2, plot=plotR, name="corsair")
        corsair['features'] = features_corsair
        corsair['timing'] = 'early'

        ### Nony opening aka fast goons range
        features_nony = []
        if formating == 'arff':
            features_nony = [template.index("ProtossGoon"),\
                    template.index("ProtossRange")]
        elif formating == 'txt':
            features_nony = [template.index("Protoss_Dragoon"),\
                    template.index("Protoss_Dragoon_Range")]
        if kmeans:
            nony_data_int = filter_out_undef(data.take(\
                    features_nony, 1), typ=np.int64)
            nony_km = k_means(nony_data_int[1], nbiter=nbiterations)
        nony_data = filter_out_undef(data.take(features_nony, 1))
        if EM:
            nony_em = expectation_maximization(nony_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        nony = r_em(nony_data[1], nbclusters=2, plot=plotR, name="nony")
        nony['features'] = features_nony
        nony['timing'] = 'early'

        ### Reaver Drop
        features_reaver_drop = []
        if formating == 'arff':
            features_reaver_drop = [template.index("ProtossReavor"),\
                    template.index("ProtossShuttle")]
        elif formating == 'txt':
            features_reaver_drop = [template.index("Protoss_Reaver"),\
                    template.index("Protoss_Shuttle")]
        if kmeans:
            reaver_drop_data_int = filter_out_undef(data.take(\
                    features_reaver_drop, 1), typ=np.int64)
            reaver_drop_km = k_means(reaver_drop_data_int[1], nbiter=nbiterations)
        reaver_drop_data = filter_out_undef(data.take(features_reaver_drop, 1))
        if EM:
            reaver_drop_em = expectation_maximization(reaver_drop_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        reaver_drop = r_em(reaver_drop_data[1], nbclusters=2, plot=plotR, name="reaver_drop")
        reaver_drop['features'] = features_reaver_drop
        reaver_drop['timing'] = 'late'


        two_gates['name'] = "two_gates"
        fast_dt['name'] = "fast_dt"
        templar['name'] = "templar"
        speedzeal['name'] = "speedzeal"
        corsair['name'] = "corsair"
        nony['name'] = "nony"
        reaver_drop['name'] = "reaver_drop"
        if plotM:
            print two_gates
            plot(two_gates, two_gates_data[1])
            print fast_dt
            plot(fast_dt, fast_dt_data[1])
            print templar
            plot(templar, templar_data[1])
            print speedzeal
            plot(speedzeal, speedzeal_data[1])
            print corsair
            plot(corsair, corsair_data[1])
            print nony
            plot(nony,nony_data[1])
            print reaver_drop
            plot(reaver_drop, reaver_drop_data[1])
        write_arff(template, annotate(datalist,\
                (two_gates_data, two_gates), (fast_dt_data, fast_dt),\
                (templar_data, templar), (speedzeal_data, speedzeal),\
                (corsair_data, corsair),\
                (nony_data, nony), (reaver_drop_data, reaver_drop)),\
                "my"+sys.argv[1])

        ### Applying this learned model to the second argument file
        if len(sys.argv[2]) > 2:
            formating = 'UNKNOWN'
            if (sys.argv[2][-4:] == 'arff'):
                formating = 'arff'
            elif (sys.argv[2][-3:] == 'txt'):
                formating = 'txt'
            else:
                print "unknown input format/extension"
                sys.exit(-1)
            (template, datalist) = parse(open(sys.argv[2]), formating)
            datalist /= 24 #?????
            print opening_probabilities(datalist,\
                two_gates, fast_dt, templar, speedzeal,\
                corsair, nony, reaver_drop)

    if race == "T":
        # Main openings:
        # - BBS rush (rax/rax/supply) / 8 rax [Disabled: can only scout it]
        # - Bio push (3 raxes at least)
        # - 1 Rax FE or 2 Rax FE
        # - Siege Expand (facto into siege + expand)
        # - 2 Factories (aggressive push / nada style)
        # - Vultures harass
        # - Wraith

        ### Bio push
        features_bio = []
        if formating == 'arff':
            features_bio = [template.index("TerranThirdBarracks"),\
                    template.index("TerranSecondBarracks"),\
                    template.index("TerranBarracks")]
        elif formating == 'txt':
            features_bio = [template.index("Terran_Barracks3"),\
                    template.index("Terran_Barracks2"),\
                    template.index("Terran_Barracks")]
        bio_data = filter_out_undef(data.take(features_bio, 1))
        if EM:
            bio_em = expectation_maximization(bio_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        bio = r_em(bio_data[1], nbclusters=2, plot=plotR, name="bio")
        bio['features'] = features_bio
        bio['timing'] = 'early'

        ### Rax FE
        features_rax_fe = []
        if formating == 'arff':
            features_rax_fe = [template.index("TerranExpansion"),\
                    template.index("TerranBarracks")]
        elif formating == 'txt':
            features_rax_fe = [template.index("Terran_Expansion"),\
                    template.index("Terran_Barracks")]
        rax_fe_data = filter_out_undef(data.take(features_rax_fe , 1))
        if EM:
            rax_fe_em = expectation_maximization(rax_fe_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        rax_fe = r_em(rax_fe_data[1], nbclusters=2, plot=plotR, name="rax_fe")
        rax_fe['features'] = features_rax_fe
        rax_fe['timing'] = 'early'

        ### Siege Expand # TODO
#        features_siege_exp = [template.index("TerranSiege"),\
#                template.index("TerranExpansion")]
#        siege_exp_data = filter_out_undef(data.take(features_siege_exp, 1))
#        if EM:
#            siege_exp_em = expectation_maximization(siege_exp_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        siege_exp = r_em(siege_exp_data[1], nbclusters=2, plot=plotR, name="siege_exp")
#        siege_exp['features'] = features_siege_exp
#        siege_exp['timing'] = 'early'

        ### 2 Facto (the fast 2nd facto play)
        features_two_facto = []
        if formating == 'arff':
            features_two_facto = [template.index("TerranSecondFactory")]
        elif formating == 'txt':
            features_two_facto = [template.index("Terran_Factory2")]
                #template.index("TerranSiege")]
        two_facto_data = filter_out_undef(data.take(features_two_facto, 1))
        if EM:
            two_facto_em = expectation_maximization(two_facto_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        two_facto = r_em(two_facto_data[1], nbclusters=2, plot=plotR, name="two_facto")
        two_facto['features'] = features_two_facto
        two_facto['timing'] = 'late'

        ### Vultures harass
        features_vultures = []
        if formating == 'arff':
            features_vultures = [template.index("TerranMines"),\
                    template.index("TerranVulture")]
        elif formating == 'txt':
            features_vultures = [template.index("Terran_Spider_Mines"),\
                    template.index("Terran_Vulture")]
        vultures_data = filter_out_undef(data.take(features_vultures, 1))
        if EM:
            vultures_em = expectation_maximization(vultures_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        vultures = r_em(vultures_data[1], nbclusters=2, plot=plotR, name="vultures")
        vultures['features'] = features_vultures
        vultures['timing'] = 'late'

        ### Fast air ==> misses the TerranSecondStarport feature
#        features_air = [template.index("TerranWraith"),\
#                template.index("TerranStarport")]
#        air_data = filter_out_undef(data.take(features_air, 1))
#        if EM:
#            air_em = expectation_maximization(air_data[1],\
#                    nbiter=nbiterations, normalize=True, monotony=True)
#        air = r_em(air_data[1], nbclusters=2, plot=plotR, name="air")
#        air['features'] = features_air
#        air['timing'] = 'late'

        ### Fast Drop
        features_drop = []
        if formating == 'arff':
            features_drop = [template.index("TerranDropship")]
        elif formating == 'txt':
            features_drop = [template.index("Terran_Dropship")]
        drop_data = filter_out_undef(data.take(features_drop, 1))
        if EM:
            drop_em = expectation_maximization(drop_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        drop = r_em(drop_data[1], nbclusters=2, plot=plotR, name="drop")
        drop['features'] = features_drop
        drop['timing'] = 'late'

        bio['name'] = "bio"
        rax_fe['name'] = "rax_fe"
        two_facto['name'] = "two_facto"
        vultures['name'] = "vultures"
#        air['name'] = "air"
        drop['name'] = "drop"
        if plotM:
            print bio
            plot(bio, bio_data[1])
            print rax_fe
            plot(rax_fe, rax_fe_data[1])
            print two_facto
            plot(two_facto, two_facto_data[1])
            print vultures
            plot(vultures, vultures_data[1])
#            print air
#            plot(air, air_data[1])
            print drop
            plot(drop, drop_data[1])

        write_arff(template, annotate(datalist,\
                (bio_data, bio), (rax_fe_data, rax_fe),\
                (two_facto_data, two_facto),\
                (vultures_data, vultures),\
                (drop_data, drop)),\
                "my"+sys.argv[1])
                #(air_data, air),\

    if race == "Z":
        # Main openings:
        # - 4-6 pools very early glings rush [Disabled: can only scout it]
        # - ~9pool/9speed speedlings rush
        # - any kind of fast expand (overpool, 12 hatch...) into:
        #   * fast mutas (2 hatches muta, or even 1 hatch mutas in ZvZ
        #   * mass mutas (3 hatches, or more, into mutas)
        #   * fast lurkers (3 hatch lurker)
        #   * hydras push/drop

        ### Speedlings rush
        features_speedlings = []
        if formating == 'arff':
            features_speedlings = [template.index("ZergZerglingSpeed"),\
                    template.index("ZergPool"),\
                    template.index("ZergZergling")]
        elif formating == 'txt':
            features_speedlings = [template.index("Zerg_Zergling_Speed"),\
                    template.index("Zerg_Spawning_Pool"),\
                    template.index("Zerg_Zergling")]
        speedlings_data = filter_out_undef(data.take(features_speedlings, 1))
        if EM:
            speedlings_em = expectation_maximization(speedlings_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        speedlings = r_em(speedlings_data[1], nbclusters=2, plot=plotR, name="speedlings")
        speedlings['features'] = features_speedlings
        speedlings['timing'] = 'early'

        ### Fast mutas
        features_fast_mutas = []
        if formating == 'arff':
            features_fast_mutas = [template.index("ZergMutalisk"),\
                    template.index("ZergGas")] # TODO
        elif formating == 'txt':
            features_fast_mutas = [template.index("Zerg_Mutalisk"),\
                    template.index("Zerg_Extractor")] # TODO
        fast_mutas_data = filter_out_undef(data.take(features_fast_mutas, 1))
        if kmeans:
            fast_mutas_data_int = filter_out_undef(data.take(\
                    features_fast_mutas, 1), typ=np.int64)
            #fast_mutas_data_int = filter_out_undef(data.take([\
                    #features_fast_mutas[0]], 1), typ=np.int64)
            fast_mutas_km = k_means(fast_mutas_data_int[1], nbiter=nbiterations)
        if EM:
            fast_mutas_em = expectation_maximization(fast_mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        fast_mutas = r_em(fast_mutas_data[1], nbclusters=2, plot=plotR, name="fast_mutas")
        fast_mutas['features'] = features_fast_mutas
        fast_mutas['timing'] = 'early'

        ### 3 Hatch mutas / Mass mutas
        features_mutas = []
        if formating == 'arff':
            features_mutas = [template.index("ZergThirdHatch"),\
                    template.index("ZergMutalisk")]
        elif formating == 'txt':
            features_mutas = [template.index("Zerg_Expansion2"),\
                    template.index("Zerg_Mutalisk")]
        mutas_data = filter_out_undef(data.take(features_mutas, 1))
        if EM:
            mutas_em = expectation_maximization(mutas_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        mutas = r_em(mutas_data[1], nbclusters=2, plot=plotR, name="mutas")
        mutas['features'] = features_mutas
        mutas['timing'] = 'late'

        ### Fast Lurkers (Third hatch should be late)
        features_lurkers = []
        if formating == 'arff':
            features_lurkers = [template.index("ZergLurker")]
        elif formating == 'txt':
            features_lurkers = [template.index("Zerg_Lurker")]
        lurkers_data = filter_out_undef(data.take(features_lurkers, 1))
        if EM:
            lurkers_em = expectation_maximization(lurkers_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        lurkers = r_em(lurkers_data[1], nbclusters=2, plot=plotR, name="lurkers")
        lurkers['features'] = features_lurkers
        lurkers['timing'] = 'late'

        ### Hydras
        features_hydras = []
        if formating == 'arff':
            features_hydras = [template.index("ZergHydra"),\
                    template.index("ZergHydraSpeed"),\
                    template.index("ZergHydraRange")]
                    #template.index("ZergThirdHatch")] # TODO remove 3rd hatch?
        elif formating == 'txt':
            features_hydras = [template.index("Zerg_Hydralisk"),\
                    template.index("Zerg_Hydralisk_Speed"),\
                    template.index("Zerg_Hydralisk_Range")]
                    #template.index("ZergThirdHatch")] # TODO remove 3rd hatch?
        hydras_data = filter_out_undef(data.take(features_hydras, 1))
        if EM:
            hydras_em = expectation_maximization(hydras_data[1],\
                    nbiter=nbiterations, normalize=True, monotony=True)
        hydras = r_em(hydras_data[1], nbclusters=2, plot=plotR, name="hydras")
        hydras['features'] = features_hydras
        hydras['timing'] = 'early'

        speedlings['name'] = "speedlings"
        fast_mutas['name'] = "fast_mutas"
        mutas['name'] = "mutas"
        lurkers['name'] = "lurkers"
        hydras['name'] = "hydras"
        if plotM:
            print speedlings
            plot(speedlings, speedlings_data[1])
            print fast_mutas
            plot(fast_mutas, fast_mutas_data[1])
            print mutas
            plot(mutas, mutas_data[1])
            print lurkers
            plot(lurkers, lurkers_data[1])
            print hydras
            plot(hydras, hydras_data[1])

        write_arff(template, annotate(datalist,\
                (fast_mutas_data, fast_mutas),\
                (mutas_data, mutas), (lurkers_data, lurkers),\
                (hydras_data, hydras)), "my"+sys.argv[1])



