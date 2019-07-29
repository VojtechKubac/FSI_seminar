from dolfin import *
#dolfin-convert mesh_feelpp.msh mesh_feelpp.xml

from dolfin_utils import meshconvert

# Convert to XML
ifilename = 'mesh_feelpp.msh'
ofilename = 'mesh_feelpp.xml'
meshconvert.convert2xml(ifilename, ofilename)


# Order mesh
import os
os.system("dolfin-order %s" % ofilename)
