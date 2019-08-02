#!/usr/bin/env python

import sys
import os.path
from dolfin import *
from mshr import *
import numpy as np
from dolfin import __version__

if __version__[:4] == '2018' or  __version__[:4] == '2019':
    comm = MPI.comm_world
else:
    comm = mpi_comm_world()

parameters["refinement_algorithm"] = "plaza_with_parent_facets"
#parameters['allow_extrapolation'] = True

def read_mesh_xml(name):
    mesh = Mesh(name+'.xml')
    bd=None
    sd=None
    
    if os.path.isfile(name+"_facet_region.xml") : 
        bd = MeshFunction("size_t", mesh, name+"_facet_region.xml")

    if os.path.isfile(name+"_physical_region.xml") : 
        sd = MeshFunction("size_t", mesh, name+"_physical_region.xml")

    mesh.init()
    #plot(bd, interactive=True)
    #plot(sd, interactive=True)
    return(mesh,bd,sd)

# for parsing input arguments
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--mesh", dest="mesh_name", default='bench3D')
(options, args) = parser.parse_args()

name=options.mesh_name

mesh, bd, sd =read_mesh_xml(name)

print("Save the hdf mesh and boundaries...")
hdf = HDF5File(mesh.mpi_comm(), name+".h5", "w")
hdf.write(mesh, "/mesh")
if(sd) : hdf.write(sd, "/domains")
if(bd) : hdf.write(bd, "/bndry")

#save mesh just for paraview if needed
print("Save the xdmf mesh...")    
meshfile = XDMFFile(comm,name+"_pv_mesh.xdmf")
meshfile.write(mesh)
if(bd) :
    bdfile = XDMFFile(comm,name+"_pv_facets.xdmf")
    bdfile.write(bd)
if(sd) :
    sdfile = XDMFFile(comm,name+"_pv_cells.xdmf")
    sdfile.write(sd)

print("Done.")
