###############################################################
###
###     for more on mesh-motion techniques see Wick 
###                 https://doi.org/10.1016/j.compstruc.2011.02.019
###
###     Variational problem for projcting forces taken from question in FEniCS forum
###                 https://fenicsproject.discourse.group/t/project-gradient-on-boundarymesh/262
###
###############################################################

###############################################################
###
###     TODO:
###         * set coupling BC for displacement and velocities (straightforward approach fails)
###
###         * mapping of forces gives 'ERROR: RBF Polynomial linear system has not converged.'
###                 - that's something I don't understand - do I have a bad function type?
###                     (using boundary vector fuction for that)
###
###############################################################



from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD
from optparse import OptionParser
from fenicsadapter import Adapter


# MPI communication for parallel runs
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


class Fluid(object):
    def __init__(self, mesh, coupling_boundary, complementary_boundary, bndry, dt, theta, v_max, 
            mu_f, rho_f, result, *args, **kwargs):

        # initialize meshes and boundaries
        self.mesh  = mesh
        self.coupling_boundary = coupling_boundary
        self.complementary_boundary = complementary_boundary
        bnd_mesh = BoundaryMesh(mesh, 'exterior')
        self.bndry = bndry          # boundary-marking function for integration

        self.fenics_dt  = float(dt)
        self.dt = Constant(dt)
        self.theta = theta
        self.t     = 0.0
        self.v_max = v_max
        self.mu_f     = mu_f
        self.rho_f    = rho_f
        

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)	    # velocity element
        eU = VectorElement("CG", mesh.ufl_cell(), 2)	    # displacement element
        eP = FiniteElement("CG", mesh.ufl_cell(), 1)	    # pressure element

        eW = MixedElement([eV, eU, eP])                     # mixed element
        W  = FunctionSpace(self.mesh, eW)                   # function space for ALE fluid equation
        self.W = W
        self.U = FunctionSpace(self.mesh, eU)               # function space for projected functions 


        self.W_boundary = VectorFunctionSpace(bnd_mesh, 'CG', 2)    # boundary function space
        self.fun4forces = Function(self.W)                  # function for Variational formulation

        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
                      v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),
                      degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        #info("Expression set.")
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _FLUID_CYLINDER)
        bc_u_in     = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_walls  = DirichletBC(self.W.sub(1).sub(1), Constant(0.0), bndry, _WALLS)
        bc_u_circle = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _FLUID_CYLINDER)
        bc_u_out    = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_out]


        #info("Normal and Circumradius.")
        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define functions
        self.w  = Function(self.W)      
        self.w0 = Function(self.W)

        (v_, u_, p_) = TestFunctions(self.W)
        (self.v, self.u, self.p) = split(self.w)
        (self.v0, self.u0, self.p0) = split(self.w0)

        # define coupling BCs
        self.u_D = Constant((0.0, 0.0))                     # Dirichlet BC (for Fluid)
        self.u_D_function = interpolate(self.u_D, self.U)   # Dirichlet BC as function
        self.f_N = Constant((0.0, 0.0))                     # Neumann BC (for Solid)
        self.f_N_function = interpolate(self.f_N, self.U)   # Neumann BC as function

        # start preCICE adapter
        info('Initialize Adapter.')
        self.precice = Adapter()
        # read forces(Neumann condition for solid) and write displacement(Dirichlet condition for fluid)
        info('Call precice.initialize(...).')
        self.precice_dt = self.precice.initialize(
                coupling_subdomain=self.coupling_boundary, mesh=self.mesh, 
                read_field=self.u_D_function, write_field=self.f_N_function, u_n=self.w,#u_n,
                coupling_marker=self.bndry)


        # TODO: need to set DirichletBC for displacement AND velocity
        # functions for displacement BC
        self.u_bc  = self.precice.create_coupling_dirichlet_boundary_condition(self.U)  
        self.u_bc0 = self.u_bc

        dt = self.dt.values()[0]
        self.v_bc = self.u_bc                                   # wrong condition, but code runs
        #self.v_bc = (1.0/self.dt)*(self.u_bc - self.u_bc0)     # I need to do this, but raises error

        bc_u_FSI = DirichletBC(self.W.sub(1), self.u_bc, coupling_boundary)
        bc_v_FSI = DirichletBC(self.W.sub(0), self.v_bc, coupling_boundary)

        self.bcs.append(bc_u_FSI)
        self.bcs.append(bc_v_FSI)

        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        self.JJ  = det(self.FF)
        self.JJ0 = det(self.FF0)

        # mesh-moving eqaution (pseudoelasticity)
        self.gamma = 9.0/8.0
        h = CellVolume(self.mesh)**(self.gamma)     # makes mesh stiffer when mesh finer 
                                                    # (our mesh is finer by the FSI -moving- interface)
        E = Constant(1.0)

        E_mesh = E/h                                                    # Young Modulus
        nu_mesh = Constant(-0.02)                                       # Poisson ratio
        mu_mesh = E_mesh/(2*(1.0+nu_mesh))                              # Lame 1
        lambda_mesh = (nu_mesh*E_mesh)/((1+nu_mesh)*(1-2*nu_mesh))      # Lame 2

        # variational formulation for mesh motion
        F_mesh = inner(mu_mesh*2*sym(grad(self.u)), grad(u_))*dx(0) \
                + lambda_mesh*inner(div(self.u), div(u_))*dx(0)


        # define referential Grad and Div shortcuts
        def Grad(f, F): return dot( grad(f), inv(F) )
        def Div(f, F): return tr( Grad(f, F) )

        # approximate time derivatives
        du = (1.0/self.dt)*(self.u - self.u0)
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute velocuty part of Cauchy stress tensor for fluid
        self.T_f  = -self.p*I + 2*self.mu_f*sym(Grad(self.v,  self.FF))
        self.T_f0 = -self.p*I + 2*self.mu_f*sym(Grad(self.v0, self.FF0))
        # compute 1st Piola-Kirchhoff tensor for fluid
        self.S_f  = self.JJ *( -self.p*I + self.T_f )*inv(self.FF).T
        self.S_f0 = -self.JJ*self.p*I*inv(self.FF).T \
                     + self.JJ0*self.T_f0*inv(self.FF0).T

        # write equations for fluid
        a_fluid  = inner(self.S_f , Grad(v_, self.FF))*self.JJ*dx \
               + inner(self.rho_f*Grad(self.v, self.FF )*(self.v  - du), v_)*self.JJ*dx
        a_fluid0 = inner(self.S_f0, Grad(v_, self.FF0))*self.JJ0*dx \
               + inner(self.rho_f*Grad(self.v0, self.FF0)*(self.v0 - du), v_)*self.JJ0*dx

        b_fluid  = inner(Div( self.v, self.FF ), p_)*self.JJ*dx
        b_fluid0 = inner(Div( self.v, self.FF ), p_)*self.JJ*dx

        # final variationl formulation
        self.F_fluid  = (self.theta*self.JJ+(1.0 - self.theta)*self.JJ0)*self.rho_f*inner(dv, v_)*dx\
                   + self.theta*(a_fluid + b_fluid) + (1.0 - self.theta)*(a_fluid0 + b_fluid0) \
                   + F_mesh

        # differentiate w.r.t. unknown
        dF_fluid = derivative(self.F_fluid, self.w)

        # define problem and its solver
        self.problem = NonlinearVariationalProblem(self.F_fluid, self.w, bcs=self.bcs, J=dF_fluid)
        self.solver  = NonlinearVariationalSolver(self.problem)

        # Variational problem for extracting forces
        (v, u, p) = TrialFunctions(self.W)
        self.traction = self.F_fluid - inner(v, v_)*ds(_FSI)
        self.hBC = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), self.complementary_boundary)


        # configure solver parameters
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        self.solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10
        self.solver.parameters['newton_solver']['maximum_iterations'] = 50
        self.solver.parameters['newton_solver']['linear_solver']      = 'mumps'

        # create files for saving
        if my_rank == 0:
            if not os.path.exists(result):
                os.makedirs(result)
        self.vfile = XDMFFile("%s/velocity.xdmf" % result)
        self.ufile = XDMFFile("%s/displacement.xdmf" % result)
        self.pfile = XDMFFile("%s/pressure.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result)
        self.vfile.parameters["flush_output"] = True
        self.ufile.parameters["flush_output"] = True
        self.pfile.parameters["flush_output"] = True
        self.sfile.parameters["flush_output"] = True
        with open(result+'/data.csv', 'w') as data_file:
            writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
            writer.writerow(['time',
                              'x-coordinate of end of beam', 'y-coordinate of end of beam',
                              'drag', 'lift'])

        #info("Fluid __init__ done")

    def solve(self, t, n):
        self.t = t
        self.v_in.t = t
        self.dt.assign(np.min([self.fenics_dt, self.precice_dt]))
        # solve fluid equations
        self.solver.solve()

        # force-extracting procedure
        ABdry = assemble(lhs(self.traction),keep_diagonal=True)
        bBdry = assemble(rhs(self.traction))
        self.hBC.apply(ABdry, bBdry)
        solve(ABdry, self.fun4forces.vector(), bBdry)
        self.fun4forces.set_allow_extrapolation(True)
        (self.forces, _, _) = split(self.fun4forces)
        self.forces = project(self.forces, self.U)

        forces_for_solid = interpolate(self.forces, self.W_boundary) 

        # precice coupling step
        t, n, precice_timestep_complete, self.precice_dt \
                = self.precice.advance(forces_for_solid, self.w, self.w0, t, self.dt.values()[0], n)

        return t, n, precice_timestep_complete

    def save(self, t):
        (v, u, p) = self.w.split()

        # save functions to files
        v.rename("v", "velocity")
        u.rename("u", "displacement")
        p.rename("p", "pressure")
        self.vfile.write(v, t)
        self.ufile.write(u, t)
        self.pfile.write(p, t)

        # Compute drag and lift
        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.bndry, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.bndry, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        drag = -assemble(action(self.F_fluid,w_))
        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.bndry, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.bndry, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        lift = -assemble(action(self.F_fluid,w_))

        # MPI trick to extract beam displacement
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

        pi = 0
        if self.bb.compute_first_collision(B) < 4294967295:
            pi = 1
        else:
            pB_loc = 0.0
        pB = MPI.sum(comm, pB_loc) / MPI.sum(comm, pi)
        p_diff = pB - pA

        # write data to data file
        if my_rank == 0:
            with open(result+'/data.csv', 'a') as data_file:
                writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
                writer.writerow([t, Ax, Ay, drag, lift])



# set problem and its discretization
parser = OptionParser()
parser.add_option("--benchmark", dest="benchmark", default='FSI3')
parser.add_option("--mesh", dest="mesh_name", default='mesh_Fluid_L1')
parser.add_option("--theta", dest="theta", default='0.5')           # 0.5 -CN, 1.0 -implicit Euler
parser.add_option("--dt", dest="dt", default='0.001')

(options, args) = parser.parse_args()

# name of benchmark 
benchmark = options.benchmark

# name of mesh
mesh_name = options.mesh_name
relative_path_to_mesh = 'Fluid/'+mesh_name+'.h5'

# value of theta to theta scheme for temporal discretization
theta = Constant(options.theta)

# time step size
dt = options.dt


# load mesh with boundary and domain markers
sys.path.append('.')
import utils_for_fluid_solver as utils

v_max, mu_f, rho_f, t_end, result = utils.get_benchmark_specification(benchmark)
t_end = 10.0

(mesh, bndry, coupling_boundary, complementary_boundary, A, B) \
        = utils.give_gmsh_mesh(relative_path_to_mesh)

# domain (used while building mesh) - needed for inflow condition
gW = 0.41

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW         = 1
_WALLS          = 2
_FLUID_CYLINDER = 3
_FSI            = 4
_OUTFLOW        = 5


dx  = dx(domain=mesh)
ds  = ds(domain=mesh, subdomain_data = bndry)

################# export domains and bndry to xdmf for visualization in Paraview
# facet markers can be written directly
with XDMFFile("%s/mesh_bndry.xdmf" % result) as f:
    f.write(bndry)
############################################################################


fluid = Fluid(mesh, coupling_boundary, complementary_boundary, bndry, dt, theta, v_max, mu_f, rho_f, 
        result)

t = 0.0
n = 0

# start solving coupled problem, the final time is defined in precice-config.xml
while fluid.precice.is_coupling_ongoing():
    # compute solution
    t, n, precice_timestep_complete = fluid.solve(t, n)

    if precice_timestep_complete and t % 0.01 < 1e-05:
        fluid.save(t)

fluid.precice.finalize()
