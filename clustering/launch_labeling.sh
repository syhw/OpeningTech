#!/bin/bash
for name in `ls scm*.arff`
    do echo $name && python clustering.py $name
done
