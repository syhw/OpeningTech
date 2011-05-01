#!/bin/bash

PROBT_INCLUDE=/Users/gabrielsynnaeve/these/code/probt/include
PROBT_LIB=/Users/gabrielsynnaeve/these/code/probt/lib

model: model.cpp
	g++ -ggdb -arch i386 -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o model

tests: test_x_values.cpp test_functional_dirac.cpp test_lambda.cpp test_getOpeningVal.cpp test_getBuildings.cpp
	g++ -ggdb test_x_values.cpp -o test_x_values
	g++ -arch i386 -I$(PROBT_INCLUDE) test_functional_dirac.cpp \
		-L$(PROBT_LIB) -lpl -o test_functional_dirac
	g++ -arch i386 -I$(PROBT_INCLUDE) test_lambda.cpp \
		-L$(PROBT_LIB) -lpl -o test_lambda
	g++ -arch i386 -I$(PROBT_INCLUDE) test_learning.cpp \
		-L$(PROBT_LIB) -lpl -o test_learning
	g++ -ggdb test_getOpeningVal.cpp -o test_getOpeningVal
	g++ -ggdb test_getBuildings.cpp -o test_getBuildings

all: tests model
	make run
	[ -x /usr/bin/say ] && say "Battlecruiser operational!"

run:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./model lPvP.txt tPvP.txt
	echo $(PROBT_LIB)

debugrun:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH gdb ./model < PvP.txt
	echo $(PROBT_LIB)

mymodel: model.cpp
	g++ -ggdb -arch i386 -DMY_OPENINGS_LABELS -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o mymodel

test_x_values: tests
	./test_x_values

test_functional_dirac: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./test_functional_dirac

test_lambda: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./test_lambda

test_learning: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./test_learning
	sed -i '' 's#set data style lines#set style data lines#' *.gnuplot
	gnuplot *.gnuplot

gnuplot_format: 
	sed -i '' 's#set data style lines#set style data lines#' *.gnuplot
	sed -i '' 's#X#TechTree#' *.gnuplot
	sed -i '' 's#plTabulatedDistribution#P#' *.gnuplot
	for file in `ls *.gnuplot`; do echo 'set terminal aqua font "sans,12"' | cat - $$file > /tmp/out && mv /tmp/out $$file ; done

gnuplot_heatmap: gnuplot_format
	sed -i '' 's#set style data lines#set pm3d map#' *.gnuplot

gnuplot_bw_log_heatmap: gnuplot_heatmap
	for file in `ls *.gnuplot`; do echo 'set palette gray negative' | cat - $$file > /tmp/out && mv /tmp/out $$file ; done
	for file in `ls *.gnuplot`; do echo 'set logscale zcb' | cat - $$file > /tmp/out && mv /tmp/out $$file ; done

test_getOpeningVal: tests
	./test_getOpeningVal < lPall.txt

test_getBuildings: tests
	./test_getBuildings < lPall.txt | less

runtests: test_x_values test_functional_dirac test_lambda test_getOpeningVal test_getBuildings

fullbenchs: model mymodel
	echo "TODO"

benchs: model mymodel
	echo "Benchmarks with Ben Weber labels:\n" > benchs.txt
	for name in [TPZ]v[TPZ].txt; do echo "$${name%.*}" >> benchs.txt &&\
		./model l$$name t$$name | grep ">>> Positive classif" >> benchs.txt\
		&& echo "\n" >> benchs.txt; done
	echo "Benchmarks with my labels:\n" >> benchs.txt
	for name in [TPZ]v[TPZ]x.txt; do echo "$${name%.*}" >> benchs.txt &&\
		./mymodel l$$name t$$name | grep ">>> Positive classif" >> benchs.txt\
		&& echo "\n" >> benchs.txt; done

noisebenchs: 
	for ((i=1; i<16; i++)); do \
		./noisy.sh $$i >> benchs.txt; done

.PHONY: model mymodel tests

clean:
	rm -rf ./-* ./:* ./[* prefix option illegal mktemp: c++-header *~ \
		*.dSYM \
		*.gnuplot* \
		*.fig \
		test_x_values test_getOpeningVal test_getBuildings \
		test_functional_dirac test_lambda test_learning\
		protoss_possible_tech_trees.txt \
		terran_possible_tech_trees.txt \
		zerg_possible_tech_trees.txt \
	 	model \
		t*n*.txt \
		x_values

