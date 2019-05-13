from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD
from fenicsadapter import Adapter

if __version__[:4] == '2017':
    comm = mpi_comm_world()
else:
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

# marks for boundary facets 
_ZERO_DIRICHLET = 1
_COUPLING       = 2

class Solid(object):
    def __init__(self, mesh, bndry, dt, theta, f, lambda_s, mu_s, rho_s, result, *args, **kwargs):

        self.mesh      = mesh
        self.fenics_dt = Constant(dt)
        self.theta     = theta

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
        self.U = FunctionSpace(self.mesh, eU)           # function space for exchanging BCs

        bc_u = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _ZERO_DIRICHLET)
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

        # define coupling BCs
        self.u_D = Constant((0.0, 0.0))                     # Dirichlet BC (for Fluid)
        self.u_D_function = interpolate(self.u_D, self.U)   # Dirichlet BC as function
        self.f_N = Constant((0.0, 0.0))                     # Neumann BC (for solid)
        self.f_N_function = interpolate(self.f_N, self.U)   # Neumann BC as function

        # initial value of Dirichlet BC
        self.u_n = interpolate(self.u_D, self.U)

        # create self.displacement function for exchanging BCs
        self.displacement  = Function(self.U)
        self.displacement0 = Function(self.U)
        self.displacement.assign(project(self.u, self.U))
        self.displacement.rename("Displacement", "")

        # start preCICE adapter
        info('Initialize Adapter.')
        self.precice = Adapter()
        # read forces(Neumann condition for solid) and write displacement(Dirichlet condition for fluid)
        info('Call precice.initialize(...).')
        self.precise_dt = self.precice.initialize(coupling_subdomain=coupling_boundary, mesh=self.mesh, 
                read_field=self.f_N_function, write_field=self.u_D_function, u_n=self.u_n)

        info('Mesh ID = {}'.format(self.precice.getMeshID("Mesh-Solid")))

        self.dt = Constant(0.01)    # initial value that will be overwritten
        self.dt.assign(np.min(self.precice_dt, self.fenics_dt))

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
        self.F_solid = rho_s*inner(dv, v_)*dx \
                   + self.theta*inner(S_s , grad(v_))*dx \
                   + (1.0 - self.theta)*inner(S_s0, grad(v_))*dx \
                   + inner(du - (self.theta*self.v + (1.0 - self.theta)*self.v0), u_)*dx

        # apply Neumann boundary condition on coupling interface
        F += self.precice.create_coupling_neumann_condition(v_) 

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

    def solve(self, t, n):
        self.t = t
        self.dt.assign(np.min([self.fenics_dt, self.precice_dt]))

        # solve problem
        self.solver.solve()

        # update displacement
        self.displacement0.assign(self.displacement)
        self.displacement.assign(project(self.u, self.U))

        t, n, precice_timestep_complete, self.precice_dt \
                = self.precice.advance(self.displacement, self.displacement, self.displacement0, 
                        t, self.dt.values[0], n)

        self.w0.assign(self.w)

        return t, n, precice_timestep_complete

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
theta    = Constant(0.5)
fenics_dt = 0.01

if len(sys.argv) > 1:
    benchmark = str(sys.argv[1])
else:
    benchmark = 'CSM3'

# load mesh with boundary and domain markers
sys.path.append('.')
import utils_for_solid_solver as utils

#(mesh, bndry, domains, interface, A, B) \
#        = marker.give_marked_mesh(mesh_coarseness = mesh_coarseness, refinement = True, ALE = True)
(mesh, bndry, domains, A) = utils.give_gmsh_mesh('Solid/Solid-Mesh.h5')

# domain (used while building mesh) - needed for inflow condition
gW = 0.41


f, lambda_s, mu_s, rho_s, result = utils.get_benchmark_specification(benchmark)
result = result


solid = Solid(mesh, bndry, fenics_dt, theta, f, lambda_s, mu_s, rho_s, result)

#dt = np.min([solid.fenics_dt, solid.precice_dt])

t = 0.0

while solid.precice.is_coupling_ongoing():
    # compute solution
    t, n, precice_timestep_complete = solid.solve(t, n)

    if precice_time_step_complete:
        solid.save(t)


solid.precice.finalize()

solid.data.close()
