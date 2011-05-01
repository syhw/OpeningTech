#!/opt/local/bin/python
# Usage: cat t[PVZ]v[PVZ].txt | pyt count_min.py
import sys
m = 1000000
for line in sys.stdin:
    c = len(line.split(';'))
    c -= 1
    if c < m:
        m = c
print m

