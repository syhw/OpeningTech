import sys
divisor = int(sys.argv[1])

for line in sys.stdin:
    t = line.rstrip('\n').split(' ')
    for (i, e) in enumerate(t):
        if e and e[0] in '0123456789':
            t[i] = e.rstrip(';')
            t[i] = int(t[i])
            t[i] /= divisor
            t[i] = str(t[i]) + ';'
    print ' '.join(t)

