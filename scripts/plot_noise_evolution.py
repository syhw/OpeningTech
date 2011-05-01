import sys, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import polyfit, polyval

results = {}
n = 0
first = True
r = [0.0, 0.0, 0.0]
current = ''

for line in sys.stdin:
    if len(line) > 1 and line[1] == 'v':
        n = 0
        if first:
            current = line.rstrip('\n ')
            first = False
        else:
            tmp = copy.deepcopy(r)
            if results.has_key(current):
                results[current].append(tmp)
            else:
                results[current] = [tmp]
            current = line.rstrip('\n ')
        continue
    if '0' in line and not 'noise' in line: # not prob to 1.0 :D
        l = line.split(' ')
        l = l[len(l) - 1]
        r[n] = float(l.rstrip('\n '))
        n += 1
tmp = copy.deepcopy(r)
if results.has_key(current):
    results[current].append(tmp)
else:
    results[current] = [tmp]
current = line.rstrip('\n ')

#sys.exit(0)
#print results
width = 0.25
for k in results.iterkeys():
    fig = plt.figure()
    indrange = range(len(results[k]))
    print k
    print indrange
    ind = np.array(indrange)
    ax = fig.add_subplot(111)
    threemins = [results[k][i][0] for i in indrange]
    positive = [results[k][i][1] for i in indrange]
    allgame = [results[k][i][2] for i in indrange]
    #xt = polyval(polyfit(indrange, threemins, 4), indrange)
    #xp = polyval(polyfit(indrange, positive, 4), indrange)
    #xa = polyval(polyfit(indrange, allgame, 4), indrange)
    xt = polyval(polyfit(indrange, threemins, 2), indrange)
    xp = polyval(polyfit(indrange, positive, 2), indrange)
    xa = polyval(polyfit(indrange, allgame, 2), indrange)
    #rect0 = ax.bar(ind, threemins, width, color='r')
    #rect1 = ax.bar(ind+width+0.05, positive, width, color='g')
    #rect2 = ax.bar(ind+2*width+0.1, allgame, width, color='b')
    #rect0 = ax.plot(indrange, threemins, 'o-', color='r')
    rect0 = ax.plot(indrange, threemins, 'o', ms=8, color='#FF0000', label='online once > 3min')
    rect1 = ax.plot(indrange, positive, 's', ms=8, color='#008000', label='online twice')
    rect2 = ax.plot(indrange, allgame, '^', ms=8, color='#0000FF', label='final')
    rect0 = ax.plot(indrange, xt, 'k--', linewidth=3, color='r', label='OO3')
    rect1 = ax.plot(indrange, xp, '-', linewidth=3, color='g', label='OT')
    rect2 = ax.plot(indrange, xa, 'k-.', linewidth=3, color='b', label='final')
    ax.set_title('prediction probability, test set, ' + k)
    plt.legend()
    plt.show()
