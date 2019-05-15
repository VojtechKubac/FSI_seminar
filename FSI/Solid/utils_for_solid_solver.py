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
    if not os.path.isfile(name):
        raise ValueError('Invalid name for gmsh mesh.')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)

    # reads MeshFunctions
    #domains = MeshFunction('size_t', mesh, 2, mesh.domains())
    #hdf.read(domains, '/domains')
    #bndry = MeshFunction('size_t', mesh, 1)
    #hdf.read(bndry, '/bndry')    

    # devide boundary into subdomains
    class CouplingBoundary(SubDomain):
        """
        Determines if the point is at the coupling boundary

        :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
        """
        def inside(self, x, on_boundary):
            tol = 1e-14
            if on_boundary and (near(x[1], gY - 0.5*gEH) or near(x[1], gY + 0.5*gEH)
                    or near(x[0], 0.6)):
                return True
            else:
                return False

    class ComplementaryBoundary(SubDomain):
        """
        Determines if a point is at the complementary boundary with tolerance of
        1E-14.
        :func inside(): returns True if point belongs to the boundary, otherwise
                        returns False
        """
        def __init__(self, subdomain):
            SubDomain.__init__(self)
            self.complement = subdomain

        def inside(self, x, on_boundary):
            tol = 1E-14
            if on_boundary and not self.complement.inside(x, on_boundary):
                return True
            else:
                return False

    coupling_boundary = CouplingBoundary()
    complementary_boundary = ComplementaryBoundary(coupling_boundary)
    
    return(mesh, coupling_boundary, complementary_boundary, A)

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
    elif benchmark == 'FSI1':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        result = "results-FSI1"		
    elif benchmark == 'FSI2':
        rho_s = Constant(1e04)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        result = "results-FSI2"		
    elif benchmark == 'FSI3':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(2e06)
        result = "results-FSI3"		
    else:
        raise ValueError('"{}" is a wrong name for problem specification.'.format(benchmark))
    E_s = Constant(2*mu_s*(1+nu_s))
    lambda_s = Constant((nu_s*E_s)/((1+nu_s)*(1-2*nu_s)))
    g = Constant(2.0)
    return g, lambda_s, mu_s, rho_s, result
