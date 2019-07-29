"""
Mesh processing procedures for the code solid.py
"""
from dolfin import *
from dolfin import __version__
import mshr
import os.path

# mesh specific constants
gW = 0.41		# width of the domain
gL = 2.5		# length of the domain
gX = 0.2		# x coordinate of the centre of the circle
gY = 0.2		# y coordinate of the centre of the circle
g_radius  = 0.05	# radius of the circle
gEL = 0.35		# length of the elastic part (left end fully attached to the circle)
gEH = 0.02		# hight of the elastic part

A = Point(0.6, 0.2)	# point at the end of elastic beam - for pressure comparison
B = Point(0.15, 0.2)	# point at the surface of rigid circle - for pressure comparison

# boundary marks
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# interface marks
_FSI = 1
_FLUID_CYLINDER = 2

def give_gmsh_mesh(name):
    name = name + '.h5'
    if not os.path.isfile(name):
        raise ValueError('Invalid name for gmsh mesh.')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)
    domains = MeshFunction('size_t', mesh, 2, mesh.domains())
    hdf.read(domains, '/domains')
    bndry = MeshFunction('size_t', mesh, 1)
    hdf.read(bndry, '/bndry')    
    
    return(mesh, bndry, domains, A)

def get_benchmark_specification(benchmark = 'CSM1'):
    if benchmark == 'CSM1':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        result = "results-CSM1"	
    elif benchmark == 'CSM2':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(2e06)
        result = "results-CSM2"	
    elif benchmark == 'CSM3':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        result = "results-CSM3"	
    else:
        raise ValueError('"{}" is a wrong name for problem specification.'.format(benchmark))
    E_s = Constant(2*mu_s*(1+nu_s))
    #info('E_s = {}'.format(E_s.values()))
    lambda_s = Constant((nu_s*E_s)/((1+nu_s)*(1-2*nu_s)))
    g = 2.0
    return g, lambda_s, mu_s, rho_s, result
