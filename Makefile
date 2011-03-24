
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

run:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./model lPvP.txt tPvP.txt
	echo $(PROBT_LIB)

debugrun:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH gdb ./model < PvP.txt
	echo $(PROBT_LIB)

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

test_getOpeningVal: tests
	./test_getOpeningVal < testP.txt

test_getBuildings: tests
	./test_getBuildings < testP.txt | less

runtests: test_x_values test_functional_dirac test_lambda test_getOpeningVal test_getBuildings

.PHONY: model tests

clean:
	rm -rf ./-* ./:* ./[* prefix option illegal mktemp: c++-header *~ \
		*.dSYM \
		*.gnuplot* \
		test_x_values test_getOpeningVal test_getBuildings \
		test_functional_dirac test_lambda\
		protoss_possible_tech_trees.txt \
		terran_possible_tech_trees.txt \
		zerg_possible_tech_trees.txt \
	 	model \
		x_values

