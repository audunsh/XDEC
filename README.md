# XDEC-PRI : eXtended DEC using the Resolution of Identity approximation 
## General installation instructions
1. Compile + install libint: <a href="https://github.com/evaleev/libint">https://github.com/evaleev/libint</a>
2. Compile the libint wrapper for python
3. Set environment variables according to instructions in section.

## Detailed installation on fram.sigma2.no
1. Load required modules:
'GMP/6.1.2-GCCcore-8.3.0 => GMP/6.1.2-GCCcore-7.3.0'
'module load Boost/1.68.0-intel-2018b-Python-3.6.6'
'module load Anaconda3/2019.07'
2. Clone libint:
'git clone https://github.com/evaleev/libint.git'
2a. From inside libint-folder:
'./autogen.sh'
2b. Make a build-directory outside of the libint-tree, enter it and run
'/path/to/libint/configure CC=gcc CXX=g++ CXXFLAGS="-O3" --with-pic --with-cxxgen-optflags="-O3" --prefix=$HOME/libint_install/
2c. Run 'make', optionally with '-jN' for parallel compilation
2d. Run 'make install' 
3. Clone Pybind11:
'git clone https://github.com/pybind/pybind11.git'
4. Clone the XDEC repository:
5. Install the libint wrapper:
'c++ -O3 -Wall -shared -fPIC -std=c++11 lwrap.cpp -o lwrap`python3-config --extension-suffix`  -I/cluster/software/Anaconda3/2019.07/include/python3.7m/ -I/cluster/home/[username]/pybind11/include/ -I/cluster/home/[username]/include -I/cluster/home/[username]/libint_install/include -L/cluster/home/[username]/libint_install/lib -L/cluster/home/[username]/lib -L/usr/local/lib -I/usr/local/include -lm -lgmpxx -lgmp -lint2'
