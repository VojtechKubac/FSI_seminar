# FSI_seminar

This is a repository with my computations for FSI seminar at TUM. It's about Turek&Hron FSI benchmark[1],
for which I wrote solid solver, validated in the directory *CSM_benchmark*. Then, in the directory
*FSI*, FSI benchmarks are computed. The FSI comptetions are done by partitioned approach with the use
of *preCICE* software [2]. The OpenFOAM fluid solver is taken from preCICE tutorial repository.
Finally, the *fenicsadapter.py* is a slightly modified version of official preCICE adapter for FEniCS.


[1] S. Turek and J. Hron, “Proposal for numerical benchmarking of fluid–structure interaction between an elastic object and laminar incompressible flow,” in Fluid-Structure Interaction - Modelling, Simulation, Optimization, ser. Lecture Notes in Computational Science and Engineering, H. Bungartz and M. Schäfer, Eds., no. 53. Springer, 2006, pp. 371-385, iSBN 3-540-34595-7.

[2] H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann, “preCICE – a fully parallel library for multi-physics surface coupling,” Computers and Fluids, vol. 141, pp. 250–258, 2016, advances in Fluid-Structure Interaction. Available:http://www.sciencedirect.com/science/article/pii/S0045793016300974
