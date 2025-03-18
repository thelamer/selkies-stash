#! /bin/bash

set -e

cd module
rm -Rf build/
mkdir build
cd build
cmake ..
make
mv screen_capture_module.so ../../
cd ../../

echo "Compilation complete please run screen_cap_example.py to continue"
