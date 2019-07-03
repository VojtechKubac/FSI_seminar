from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
from mpi4py.MPI import COMM_WORLD
from fenicsadapter import Adapter
import os.path

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

class Solid(object):
    def __init__(self, mesh, coupling_boundary, complementary_boundary, bndry, dt, theta,
            lambda_s, mu_s, rho_s, result, *args, **kwargs):

        self.mesh      = mesh
        self.coupling_boundary = coupling_boundary
        self.complementary_boundary = complementary_boundary

        self.fenics_dt = dt
        self.dt = Constant(dt)
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

        eW = MixedElement([eV, eU])			# function space
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
        self.U = FunctionSpace(self.mesh, eU)           # function space for exchanging BCs

        bc_u = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), self.complementary_boundary)
        bc_v = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), self.complementary_boundary)
        self.bcs = [bc_v, bc_u]

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
        self.f_N = Constant((0.0, 0.0))                     # Neumann BC (for Solid)
        self.f_N_function = interpolate(self.f_N, self.U)   # Neumann BC as function

        # initial value of Dirichlet BC
        self.u_n = interpolate(self.u_D, self.U)

        # create self.displacement function for exchanging BCs
        self.displacement  = Function(self.U)
        #self.displacement0 = Function(self.U)
        self.displacement.assign(project(self.u, self.U))
        self.displacement.rename("Displacement", "")

        # start preCICE adapter
        info('Initialize Adapter.')
        self.precice = Adapter()
        # read forces(Neumann condition for solid) and write displacement(Dirichlet condition for fluid)
        info('Call precice.initialize(...).')
        self.precice_dt = self.precice.initialize(
                coupling_subdomain=self.coupling_boundary, mesh=self.mesh, 
                read_field=self.f_N_function, write_field=self.u_D_function, u_n=self.w,#u_n,
                coupling_marker=self.bndry)
        #self.precice_dt = 0.01

        #info('Mesh ID = {}'.format(self.precice.getMeshID("Mesh-Solid")))

        #info("set dt")
        self.dt.assign(np.min([self.precice_dt, self.fenics_dt]))

        #info("define forms")
        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)

        # approximate time derivatives
        du = (1.0/self.dt)*(self.u - self.u0)
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute 1st Piola-Kirchhoff tensor for solid (St. Venant - Kirchhoff model)
        E_s  = 0.5*(self.FF.T *self.FF  - I)
        E_s0 = 0.5*(self.FF0.T*self.FF0 - I)
        #E_s  = 2*sym(grad(self.u))
        #E_s0 = 2*sym(grad(self.u0))

        S_s  = self.FF *(self.lambda_s*tr(E_s )*I + 2.0*self.mu_s*(E_s ))
        S_s0 = self.FF0*(self.lambda_s*tr(E_s0)*I + 2.0*self.mu_s*(E_s0))

        delta_W = 0.01
        alpha = Constant(1.0) # Constant(1.0/delta_W) # Constant(1000)
        self.f_surface = alpha*self.precice.create_coupling_neumann_boundary_condition(v_, 1, self.U)
        #self.f_surface = Function(self.U)

        # write equation for solid
        self.F_solid = rho_s*inner(dv, v_)*dx \
                   + self.theta*inner(S_s , grad(v_))*dx \
                   + (1.0 - self.theta)*inner(S_s0, grad(v_))*dx \
                   + inner(du - (self.theta*self.v + (1.0 - self.theta)*self.v0), u_)*dx \
                   + inner(self.f_surface, v_)*dss(1)

        # apply Neumann boundary condition on coupling interface
        #info("coupling Neumann")
        #self.F_neumann += self.precice.create_coupling_neumann_boundary_condition(v_, 1, self.U)
        #self.F_solid += self.precice.create_coupling_neumann_boundary_condition(v_, 1, self.U)

        #self.F_solid += self.F_neumann

        self.dF_solid = derivative(self.F_solid, self.w)

        #info("set Problem.")
        self.problem = NonlinearVariationalProblem(self.F_solid, self.w, bcs=self.bcs, J=self.dF_solid)
        self.solver  = NonlinearVariationalSolver(self.problem)

        # configure solver parameters
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        self.solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10
        self.solver.parameters['newton_solver']['maximum_iterations'] = 50
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
        with open(result+'/data.csv', 'w') as data_file:
            writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
            writer.writerow(['time', 'Ax_displacement', 'Ay_displacement', 'lift', 'drag'])
        info("Solid __init__ done")

    def solve(self, t, n):
        self.t = t
        self.dt.assign(np.min([self.fenics_dt, self.precice_dt]))

        # solve problem
        self.solver.solve()

        # extract velocity v and displacement u from mixed vector w
        (v, u) = self.w.split()

        # update displacement (Dirichlet BC for fluid)
        self.displacement.assign(project(u, self.U))
        #self.displacement0.assign(self.displacement)
        #self.displacement.assign(project(self.u, self.U))

        # precice coupling step
        t, n, precice_timestep_complete, self.precice_dt \
                = self.precice.advance(self.displacement, self.w, self.w0,#self.displacement, self.displacement0, 
                        t, self.dt.values()[0], n)

        #self.w0.assign(self.w)

        #print('precice_timestep_complete: ', precice_timestep_complete)

        return t, n, precice_timestep_complete

    def save(self, t):
        (v, u) = self.w.split()

        # save velocity and displacement
        v.rename("v", "velocity")
        u.rename("u", "displacement")
        self.vfile.write(v, t)
        self.ufile.write(u, t)

        # extract values of displacament in point A = (0.6, 0.2)
        self.w.set_allow_extrapolation(True)
        Ax_loc = self.u[0]((A.x(), A.y()))
        Ay_loc = self.u[1]((A.x(), A.y()))
        self.w.set_allow_extrapolation(False)
        
        # MPI stuff in case of more processes
        pi = 0
        if self.bb.compute_first_collision(A) < 4294967295:
            pi = 1
        else:
            Ax_loc = 0.0
            Ay_loc = 0.0
        Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)
        Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)

        # evaluate forces (lift and drag)
        drag = -assemble(self.f_surface[0]*dss(1))
        lift = -assemble(self.f_surface[1]*dss(1))

        # write displacement and forces to file
        with open(result+'/data.csv', 'a') as data_file:
            writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
            writer.writerow([t, Ax, Ay, lift, drag])

        info(' Ax: {}\n Ay: {} \n lift: {} \n drag: {}'.format(
            Ax, Ay, lift, drag))

        #self.w0.assign(self.w)
        #self.displacement0.assign(project(self.u0, self.U))

# time disretization
theta    = Constant(0.5)
fenics_dt = 0.001

if len(sys.argv) > 1:
    benchmark = str(sys.argv[1])
else:
    benchmark = 'FSI2'

# load mesh with boundary and domain markers
sys.path.append('.')
import utils_for_solid_solver as utils

(mesh, coupling_boundary, complementary_boundary, A) = utils.give_gmsh_mesh('Solid/Solid-Mesh.h5')
#(mesh, coupling_boundary, complementary_boundary, A) = utils.give_gmsh_mesh('../CSM_benchmark/meshes/mesh_structure_L3.h5')

# create MeshFunction for boundary integrals
bndry = MeshFunction('size_t', mesh, 1)
bndry.set_all(0)
coupling_boundary.mark(bndry, 1)

dss = Measure('ds', domain=mesh, subdomain_data=bndry)        # surface measure for integral over coupling boundary


f, lambda_s, mu_s, rho_s, result = utils.get_benchmark_specification(benchmark)
result = result


solid = Solid(mesh, coupling_boundary, complementary_boundary, bndry, fenics_dt, theta,
        lambda_s, mu_s, rho_s, result)

t = 0.0
n = 0

# start soling coupled problem, the final time is defined in precice-config.xml
while solid.precice.is_coupling_ongoing():
    # compute solution
    t, n, precice_timestep_complete = solid.solve(t, n)

    if precice_timestep_complete: 
        solid.save(t)

solid.precice.finalize()
