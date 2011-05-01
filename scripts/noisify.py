#!/opt/local/bin/python
import sys, random, math

if len(sys.argv) > 1:
    random.seed()
    for i in range(1,16):
        fil = sys.argv[1].split('.')[0]+'n'+str(i)+'.txt'
        f = open(fil, 'w')
        for line in open(sys.argv[1]):
            elems = line.split(';')
            for n in range(i):
                ind = int(math.floor(random.uniform(0, len(elems)-1.0)))
                if not "Opening" in elems[ind]:
                    #if not 'T' in elems[ind] or not 'P' in elems[ind]\
                    #        or not 'Z' in elems[ind]:
                    if elems[ind] == '' or elems[ind] == ' ':
                        n -= 1
                    else:
                        if '\n' in elems[ind]:
                            elems[ind-1] = elems[ind-1] + '\n'
                        elems.pop(ind)
                else:
                    n -= 1
            f.write(';'.join(elems))
        f.close()
        print "wrote: ", f
else:
    print "need one argument file"

