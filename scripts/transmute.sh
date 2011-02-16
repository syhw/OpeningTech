#!/bin/bash

python transmute.py scmPvP_Protoss_Mid.arff PvP.txt
python transmute.py scmPvT_Terran_Mid.arff TvP.txt
python transmute.py scmPvZ_Zerg_Mid.arff ZvP.txt
python transmute.py scmTvZ_Terran_Mid.arff TvZ.txt
python transmute.py scmZvZ_Zerg_Mid.arff ZvZ.txt
python transmute.py scmPvT_Protoss_Mid.arff PvT.txt
python transmute.py scmPvZ_Protoss_Mid.arff PvZ.txt
python transmute.py scmTvT_Terran_Mid.arff TvT.txt
python transmute.py scmTvZ_Zerg_Mid.arff ZvT.txt

