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

def get_benchmark_specification(benchmark = 'FSI3'):
    if benchmark == 'FSI1':
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 0.2
        T_end = 60.0
        result = "results-FSI1/"
    elif benchmark == 'FSI2':
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 1.0
        T_end = 15.0
        result = "results-FSI2/"		
    elif benchmark == 'FSI3':
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 2.0
        T_end = 20.0
        result = "results-FSI3/"		
    else:
        raise ValueError('"{}" is a wrong name for problem specification.'.format(benchmark))
    v_max = Constant(1.5*U)     # mean velocity to maximum velocity 
                                #      (we have parabolic profile)
    mu_f = Constant(nu_f*rho_f)
    return v_max, mu_f, rho_f, T_end, result

def give_gmsh_mesh(name):
    if not os.path.isfile(name):
        raise ValueError('Invalid name for gmsh mesh.')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)

    #hdf.read(domains, '/subdomains')
    bndry = MeshFunction('size_t', mesh, 1)
    #bndry.set_all(0)
    hdf.read(bndry, '/bndry')
    #hdf.read(bndry, '/boundaries')

    # devide boundary into subdomains
    #   - for preCICE needs SubDomains
    class CouplingBoundary(SubDomain):
        """
        Determines if the point is at the coupling boundary

        :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
        """
        def inside(self, x, on_boundary):
            tol = 1e-14
            if on_boundary and ((near(x[1], gY - 0.5*gEH) and x[0] < 0.8 and x[0] > 0.2)
                    or (near(x[1], gY + 0.5*gEH) and x[0] < 0.8 and x[0] > 0.2)
                    or (near(x[0], 0.6) and x[1] < 0.3 and x[1] > 0.1 )):
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
    
    return(mesh, bndry, coupling_boundary, complementary_boundary, A, B)
