import dolfin as df
import sys
import os.path
from optparse import OptionParser

def read_xml_files(name):
    mesh = df.Mesh(name+".xml")                                                  
    bndry = None
    domains = None

    if os.path.isfile(name+"_facet_region.xml") : 
        bndry = df.MeshFunction("size_t", mesh, name+"_facet_region.xml")

    if os.path.isfile(name+"_physical_region.xml") : 
        domains = df.MeshFunction("size_t", mesh, name+"_physical_region.xml")    

    mesh.init()

    return(mesh, bndry, domains)


parser = OptionParser()
parser.add_option("--mesh", dest="mesh_name", default='mesh_structure_L1')
(options, args) = parser.parse_args()

name=options.mesh_name

(mesh, bndry, domains) = read_xml_files(name)

df.info('Save the hdf mesh and boundaries ...')
hdf = df.HDF5File(mesh.mpi_comm(), name+".h5", "w")
hdf.write(mesh, "/mesh")
if(domains):  hdf.write(domains, "/domains")
if(bndry):    hdf.write(bndry, "/bndry")

df.info('Done.')
