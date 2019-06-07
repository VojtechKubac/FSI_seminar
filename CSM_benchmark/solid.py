from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD
from optparse import OptionParser

if __version__[:4] == '2017':       # works with FEniCS 2017
    comm = mpi_comm_world()
else:                               # works with FEniCS 2018, 2019
    comm = MPI.comm_world
my_rank = comm.Get_rank()

# Use UFLACS to speed-up assembly and limit quadrature degree
parameters["std_out_all_processes"] = False
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4

parameters['ghost_mode'] = 'shared_facet'

PETScOptions.set('mat_mumps_icntl_24', 1)	# detects null pivots
PETScOptions.set('mat_mumps_cntl_1', 0.01)	# set treshold for partial treshold pivoting, 0.01 is default value


class Structure(object):
    def __init__(self, mesh, bndry, dt, theta, g, lambda_s, mu_s, rho_s, result, *args, **kwargs):

        self.mesh  = mesh
        self.dt    = Constant(dt)
        self.theta = theta
        self.f     = Constant((0.0, f))
        self.f     = Expression(( "0.0", "g"), g = Constant(g), degree=0)

        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_s    = rho_s
        
        self.bndry = bndry

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)		# velocity space
        eU = VectorElement("CG", mesh.ufl_cell(), 2)		# displacement space
        eB = VectorElement("Bubble", mesh.ufl_cell(), mesh.geometry().dim()+1) # Bubble element

        eW = MixedElement([eV, eU])			# function space
        W  = FunctionSpace(self.mesh, eW)
        self.W = W

        bc_u = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, 1)
        self.bcs = [bc_u]

        #info("Normal and Circumradius.")
        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define functions
        self.w  = Function(self.W)      
        self.w0 = Function(self.W)

        (v_, u_) = TestFunctions(self.W)
        (self.v, self.u) = split(self.w)
        (self.v0, self.u0) = split(self.w0)

        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        self.JJ  = det(self.FF)
        self.JJ0 = det(self.FF0)


        # approximate time derivatives
        du = (1.0/self.dt)*(self.u - self.u0)
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        B_s0 = self.FF0.T*self.FF0
        S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))
        S_s0 = self.FF0*(0.5*self.lambda_s*tr(B_s0 - I)*I + self.mu_s*(B_s0 - I))

        # write equation for solid
        self.F_solid = self.rho_s*inner(dv, v_)*dx \
                   + self.theta*inner(S_s , grad(v_))*dx \
                   + (1.0 - self.theta)*inner(S_s0, grad(v_))*dx \
                   + inner(du - (self.theta*self.v + (1.0 - self.theta)*self.v0), u_)*dx \
                   + self.rho_s*inner(self.f, v_)*dx

        self.dF_solid = derivative(self.F_solid, self.w)

        self.problem = NonlinearVariationalProblem(self.F_solid, self.w, bcs=self.bcs, J=self.dF_solid)
        self.solver  = NonlinearVariationalSolver(self.problem)

        # configure solver parameters
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        self.solver.parameters['newton_solver']['maximum_iterations'] = 15
        self.solver.parameters['newton_solver']['linear_solver']      = 'mumps'

        # create files for saving
        if not os.path.exists(result):
            os.makedirs(result)
        self.vfile = XDMFFile("%s/velocity.xdmf" % result)
        self.ufile = XDMFFile("%s/displacement.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result)
        self.vfile.parameters["flush_output"] = True
        self.ufile.parameters["flush_output"] = True
        self.sfile.parameters["flush_output"] = True
        self.data = open(result+'/data.csv', 'w')
        self.writer = csv.writer(self.data, delimiter=';', lineterminator='\n')
        self.writer.writerow(['time', 'x-coordinate of end of beam', 'y-coordinate of end of beam'])

    def solve(self, t, dt):
        self.t = t
        self.dt = Constant(dt)
        self.solver.solve()

        self.w0.assign(self.w)

    def save(self, t):
        (v, u) = self.w.split()

        v.rename("v", "velocity")
        u.rename("u", "displacement")
        self.vfile.write(v, t)
        self.ufile.write(u, t)

        self.w.set_allow_extrapolation(True)
        Ax_loc = self.u[0]((A.x(), A.y()))
        Ay_loc = self.u[1]((A.x(), A.y()))
        self.w.set_allow_extrapolation(False)
        
        pi = 0
        if self.bb.compute_first_collision(A) < 4294967295:
            pi = 1
        else:
            Ax_loc = 0.0
            Ay_loc = 0.0
        Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)
        Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)
        self.writer.writerow([t, Ax, Ay])


# time disretization
t_end = 10

parser = OptionParser()
parser.add_option("--mesh", dest="mesh_name", default='mesh_structure_L4')
parser.add_option("--benchmark", dest="benchmark", default='CSM3')
parser.add_option("--theta", dest="theta", default='0.5')
parser.add_option("--dt", dest="dt", default='0.01')

(options, args) = parser.parse_args()

mesh_name = 'meshes/' + options.mesh_name   # name of the mesh to load
benchmark = options.benchmark               # specifies benchmark specific material constants
theta = float(options.theta)                # vlaue for theta schema used for time discretization
dt = float(options.dt)                      # size of time step

# load mesh with boundary and domain markers
sys.path.append('.')
import utils
f, lambda_s, mu_s, rho_s, result = utils.get_benchmark_specification(benchmark)
result = result + '/' + mesh_name[-2:] + '/' + str(dt)

#(mesh, bndry, domains, interface, A, B) \
#        = marker.give_marked_mesh(mesh_coarseness = mesh_coarseness, refinement = True, ALE = True)
(mesh, bndry, domains, A) = utils.give_gmsh_mesh(mesh_name)

# domain (used while building mesh) - needed for inflow condition
gW = 0.41


structure = Structure(mesh, bndry, dt, theta, f, lambda_s, mu_s, rho_s, result)

t = 0.0

while  t < t_end:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    structure.solve(t, dt)
    structure.save(t)

    t += float(dt)

structure.data.close()
