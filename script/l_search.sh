#!/bin/bash

../build/transfer 1>stdout.txt 2>stderr.txt
#../build/load_search 1>lout.txt 2>lerr.txt
../build/compaction 1>comout.txt 2>comerr.txt