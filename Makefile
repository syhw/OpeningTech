
PROBT_INCLUDE=/Users/gabrielsynnaeve/these/code/probt/include
PROBT_LIB=/Users/gabrielsynnaeve/these/code/probt/lib

all:
	g++ test_x_values.cpp -o test_x_values
	g++ -arch i386 -I$(PROBT_INCLUDE) test_functional_dirac.cpp \
		-L$(PROBT_LIB) -lpl -o test_functional_dirac
	g++ test_getOpeningVal.cpp -o test_getOpeningVal
	g++ test_getBuildings.cpp -o test_getBuildings
	g++ -arch i386 -I$(PROBT_INCLUDE) model.cpp -L$(PROBT_LIB) -lpl -o model

run:
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./model
	echo $(PROBT_LIB)

test_x_values: all
	./test_x_values

test_functional_dirac: all
	DYLD_LIBRARY_PATH=$(PROBT_LIB):DYLD_LIBRARY_PATH ./test_functional_dirac

test_getOpeningVal: all
	./test_getOpeningVal < testP.txt

test_getBuildings: all
	./test_getBuildings < testP.txt | less

clean:
	rm ./-* ./:* ./[* prefix option illegal mktemp: c++-header *~

