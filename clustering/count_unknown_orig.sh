#!/bin/bash
for name in `ls scm*.arff`; do echo $name && cat $name | grep 'Unknown' | wc -l; done
