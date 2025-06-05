#!/bin/bash

../build/raid_test -n 1  2>err.txt
../build/raid_test -n 2  2>err.txt
../build/raid_test -n 4  2>err.txt
../build/raid_test -n 8  2>err.txt
../build/raid_test -n 16  2>err.txt
../build/raid_test -n 32  2>err.txt
../build/raid_test -n 64  2>err.txt
../build/raid_test -n 128  2>err.txt

echo "ssd test finished"

