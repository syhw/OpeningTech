import sys, copy
import numpy as np
import matplotlib.pyplot as plt

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
    if '0' in line: # not prob to 1.0 :D
        l = line.split(' ')
        l = l[len(l) - 1]
        r[n] = float(l.rstrip('\n '))
        n += 1
#sys.exit(0)
print results
fig = plt.figure()
width = 0.25
indrange = range(len(results['PvP']))
ind = np.array(indrange)
ax = fig.add_subplot(111)
threemins = [results['PvZ'][i][0] for i in indrange]
positive = [results['PvZ'][i][1] for i in indrange]
allgame = [results['PvZ'][i][2] for i in indrange]
rect0 = ax.bar(ind, threemins, width, color='r')
rect1 = ax.bar(ind+width+0.05, positive, width, color='g')
rect2 = ax.bar(ind+2*width+0.1, allgame, width, color='b')
plt.show()
