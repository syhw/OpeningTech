#!/bin/bash

PROBT_INCLUDE=/Users/gabrielsynnaeve/these/code/probt22/include
PROBT_LIB=/Users/gabrielsynnaeve/these/code/probt22/lib
BOOST_STAGE_LIB=/Users/gabrielsynnaeve/labs/boost_1_45_0/stage/lib
BOOST_INCLUDE=/Users/gabrielsynnaeve/labs/boost_1_45_0

model: model.cpp
	g++ -ggdb -arch i386 -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o model 

model_with_serialization: model.cpp
	g++ -ggdb -arch i386 -D__SERIALIZE__ -I$(BOOST_INCLUDE) -I$(PROBT_INCLUDE) model.cpp -L$(BOOST_STAGE_LIB) -L$(PROBT_LIB) -lpl -lboost_serialization -o model 

techtrees: techtrees.cpp
	g++ -ggdb -arch i386 -DTECH_TREES -I$(PROBT_INCLUDE) techtrees.cpp -L$(PROBT_LIB) -lpl -o techtrees

tt: model.cpp
	g++ -ggdb -arch i386 -DTECH_TREES -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o tt

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

all: tests model tt
	#techtrees
	make run
	[ -x /usr/bin/say ] && say "Battlecruiser operational!"

run:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./model lPvP.txt tPvP.txt
	echo $(PROBT_LIB)

run_with_serialization:
	DYLD_LIBRARY_PATH=$(BOOST_STAGE_LIB):$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./model lPvP.txt tPvP.txt
	echo $(BOOST_STAGE_LIB)
	echo $(PROBT_LIB)

debugrun:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) gdb ./model < PvP.txt
	echo $(PROBT_LIB)

mymodel: model.cpp
	g++ -ggdb -arch i386 -DMY_OPENINGS_LABELS -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o mymodel

mymodel_with_serialization: model.cpp
	g++ -ggdb -arch i386 -DMY_OPENINGS_LABELS -I$(BOOST_INCLUDE) -I$(PROBT_INCLUDE) model.cpp -L$(BOOST_STAGE_LIB) -L$(PROBT_LIB) -lpl -lboost_serialization-xgcc42-mt -o mymodel 

test_x_values: tests
	./test_x_values

test_functional_dirac: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./test_functional_dirac

test_lambda: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./test_lambda

test_learning: tests
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./test_learning
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
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH)
	echo "TODO"

benchs: model mymodel
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH)
	echo "Benchmarks with Ben Weber labels:\n" > benchs.txt
	for name in [TPZ]v[TPZ].txt; do echo "$${name%.*}" >> benchs.txt &&\
		DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./model l$$name t$$name | grep ">>> Positive classif" >> benchs.txt\
		&& echo "\n" >> benchs.txt; done
	echo "Benchmarks with my labels:\n" >> benchs.txt
	for name in [TPZ]v[TPZ]x.txt; do echo "$${name%.*}" >> benchs.txt &&\
		DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH) ./mymodel l$$name t$$name | grep ">>> Positive classif" >> benchs.txt\
		&& echo "\n" >> benchs.txt; done

noisebenchs: 
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH)
	for ((i=1; i<16; i++)); do \
		./noisy.sh $$i >> benchs.txt; done

ttbenchs: tt
	DYLD_LIBRARY_PATH=$(PROBT_LIB):$(DYLD_LIBRARY_PATH)
	echo "Launching benchmarks:\n" > ttbenchs.txt
	for name in [TPZ]v[TPZ].txt; do echo "$${name%.*}" >> ttbenchs.txt &&\
		./tt l$$name t$$name | grep ">>>" >> ttbenchs.txt\
		&& echo "\n" >> ttbenchs.txt; done

.PHONY: model mymodel tests techtrees tt

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
	 	model tt techtrees\
		t*n*.txt \
		x_values

