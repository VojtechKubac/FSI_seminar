This repository contains codes and results for CSM validation tests dor *Turek&Hron FSI benchmark*.

The usage is as follows. At first run *Makefile* be typing `make` to create the meshes. 
The meshes are created in three levels, one corase *L0* and once *L1* or twice *L2* refined, they 
aee stored in the *meshes* repository. For creating meshes, *gmsh* and *FEniCS* 
need to be installed on the computer. Then you can start the computation.

The main code *solid.py* is writtten in *FEniCS* and the results are stored in directories 
*results-...*, where the three dots represents the name of benchmark.


