# Tutorial for an FSI simulation of an elastic flap perpendicular to a channel flow

This *Fluid* folder, *runFluid* and *precice-configure* files are just copied from preCICE project, [Clculix-OpenFOAM](https://github.com/precice/tutorials/tree/master/FSI/cylinderFlap/OpenFOAM-CalculiX)
version of this benchmark.


This tutorial is described in the [preCICE wiki](https://github.com/precice/precice/wiki/Tutorial-for-FSI-with-OpenFOAM-and-CalculiX).

The *Solid* folder is written by myself and uses *FEniCS* to solve the structural equtions. For pure solid code and CSM 
validation see the [CSM benchmark](https://github.com/VojtechKubac/FSI_seminar/tree/master/CSM_benchmark) folder in this
porject.

To run the two simulations in two different terminals and watch their output on the screen, use the (simpler) scripts `runFluid` (or `runFluid -parallel`) and `python3 Solid/solid.py`. Please always run the script `runFluid` first.

There is an [open issue](https://github.com/precice/openfoam-adapter/issues/26) that leads to additional "empty" result directories when running with some OpenFOAM versions, leading to inconveniences during post-processing. Please run the script `removeObsoleteSolvers.sh` to delete the additional files.

You may adjust the end time in the precice-config_*.xml, or interupt the execution earlier if you want.

The OpenFOAM and precCICE codes were contributed by Derek Risseeuw (TU Delft).

## Disclaimer

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.
