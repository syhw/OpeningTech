#!/bin/bash
for name in `ls scm*.arff`
    do echo $name && /opt/local/bin/python clustering.py $name
done
