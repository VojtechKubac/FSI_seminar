from dolfin import *
import sys

if len(sys.argv) == 2:
    name = sys.argv[1]
else:
    name = 'mesh_feelpp'

mesh = Mesh(name+".xml")                                                  
domains = MeshFunction("size_t", mesh, name+"_physical_region.xml")    
bndry = MeshFunction("size_t", mesh, name+"_facet_region.xml")
hdf = HDF5File(mesh.mpi_comm(), name+".h5", "w")
hdf.write(mesh, "/mesh")
hdf.write(domains, "/domains")
hdf.write(bndry, "/bndry")
