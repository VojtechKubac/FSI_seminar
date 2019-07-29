###############################################################
###
###      implicit Euler for pressure and imcompressibility
###
###      artificial mesh movement done according to Wick 
###                     https://doi.org/10.1016/j.compstruc.2011.02.019
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

PETScOptions.set('mat_mumps_icntl_24', 1)		# detects null pivots
PETScOptions.set('mat_mumps_cntl_1', 0.01)		# set treshold for partial treshold pivoting, 0.01 is default value


class Fluid(object):
    def __init__(self, mesh, coupling_boundary, bndry, dt, theta, v_max, mu_f, rho_f, mesh_move, 
            result, *args, **kwargs):

        #info("Flow initialization.") 
        self.mesh  = mesh
        self.coupling_boundary = coupling_boundary

        self.fenics_dt  = float(dt)
        self.dt = Constant(dt)
        self.theta = theta
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        
        self.mesh_move = mesh_move
        self.bndry = bndry

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)		# velocity space
        eU = VectorElement("CG", mesh.ufl_cell(), 2)		# displacement space
        eP = FiniteElement("CG", mesh.ufl_cell(), 1)		# pressure space

        eW = MixedElement([eV, eU, eP])
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
        self.U = FunctionSpace(self.mesh, eU)

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


        # TODO
        #self.u_bc  = self.precice.create_coupling_dirichlet_boundary_condition(self.U)
        #self.bcs.append(DirichletBC(self.W.sub(1), self.u_bc, coupling_boundary))
        self.u_bc  = self.precice.create_coupling_dirichlet_boundary_condition(self.U)
        self.u_bc0 = self.u_bc

        dt = self.dt.values()[0]
        self.v_bc = self.u_bc
        #self.v_bc = (1.0/dt)*self.u_bc
        #self.v_bc = (1.0/self.dt)*(self.u_bc - self.u_bc0)

        bc_u_FSI = DirichletBC(self.W.sub(1), self.u_bc, coupling_boundary)
        bc_v_FSI = DirichletBC(self.W.sub(0), self.v_bc, coupling_boundary)

        self.bcs.append(bc_u_FSI)
        self.bcs.append(bc_v_FSI)

        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        self.JJ  = det(self.FF)
        self.JJ0 = det(self.FF0)

        # write ALE mesh movement 
        if self.mesh_move == 'laplace':
            dist = Expression("( pow(x[0] - 0.425, 2) < 0.04 ? sqrt(pow(x[1] - 0.2, 2)) : \
                        pow(x[1] - 0.2, 2) < 0.0001 ? sqrt(pow(x[0] - 0.425, 2)): \
                        sqrt(pow(x[0] - 0.425, 2) + pow(x[1] - 0.2, 2)) ) < 0.06 ? \
                        1e06 : 8e04", degree=1)
            E_mesh = 5.64*(dist - dist**3/sqrt(3) + 0.1*dist**5)
            F_mesh = self.theta*inner(E_mesh*grad(self.u), grad(u_))*dx \
                       + (1.0 - self.theta)*inner(E_mesh*grad(self.u0), grad(u_))*dx

        elif self.mesh_move == 'pseudoelasticity':
            E_mesh = Expression("( pow(x[0] - 0.425, 2) < 0.04 ? sqrt(pow(x[1] - 0.2, 2)) : \
                        pow(x[1] - 0.2, 2) < 0.0001 ? sqrt(pow(x[0] - 0.425, 2)): \
                        sqrt(pow(x[0] - 0.425, 2) + pow(x[1] - 0.2, 2)) ) < 0.06 ? \
                        1e06 : 8e04", degree=1)

            nu_mesh = Constant(-0.1)
            mu_mesh = E_mesh/(2*(1.0+nu_mesh))
            lambda_mesh = (nu_mesh*E_mesh)/((1+nu_mesh)*(1-2*nu_mesh))

            F_mesh = inner(mu_mesh*2*sym(grad(self.u)), grad(u_))*dx \
                              + lambda_mesh*inner(div(self.u), div(u_))*dx
            #F_mesh = self.theta*(inner(mu_mesh*2*sym(grad(self.u)), grad(u_))*dx \
            #                  + lambda_mesh*inner(div(self.u), div(u_))*dx ) \
            #         + (1.0 - self.theta)*(inner(mu_mesh*2*sym(grad(self.u0)), grad(u_))*dx \
            #                  + lambda_mesh*inner(div(self.u0), div(u_))*dx )

        elif self.mesh_move == 'biharmonic':
            """
            Continuity of normal forces not implemented (unknown direction of normal
            on Fluid-Structure Interface)
            """
            # !!!!! kam ukazuje noramala??!!!!
            F_mesh = self.theta*(inner(grad(self.z), grad(u_))*dx \
                          + inner(grad(self.u), grad(z_))*dx - inner(self.z, z_)*dx ) \
                     + (1.0 - self.theta)*(inner(grad(self.z0), grad(u_))*dx \
                          + inner(grad(self.u0), grad(z_))*dx - inner(self.z0, z_)*dx )
                # - self.elasticity_char_fun('+')*inner((grad(self.u)*self.n)('+'), z_('+'))*dS(1)  \
                # - self.elasticity_char_fun('-')*inner((grad(self.u)*self.n)('-'), z_('-'))*dS(1) ) \
                # - self.elasticity_char_fun('+')*inner((grad(self.u0)*self.n)('+'), z_('+'))*dS(1)  \
                # - self.elasticity_char_fun('-')*inner((grad(self.u0)*self.n)('-'), z_('-'))*dS(1) ) \

        else:
            raise ValueError('Invalid argument for "mesh_move"!')

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

        self.F_fluid  = (self.theta*self.JJ+(1.0 - self.theta)*self.JJ0)*self.rho_f*inner(dv, v_)*dx\
                   + self.theta*(a_fluid + b_fluid) + (1.0 - self.theta)*(a_fluid0 + b_fluid0) \
                   + F_mesh


        dF_fluid = derivative(self.F_fluid, self.w)

        #info("set Problem.")
        self.problem = NonlinearVariationalProblem(self.F_fluid, self.w, bcs=self.bcs, J=dF_fluid)
        self.solver  = NonlinearVariationalSolver(self.problem)

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
                              'pressure difference', 
                              'drag_circle', 'drag_fluid', 'drag_fullfluid',
                              'lift_circle', 'lift_fluid', 'lift_fullfluid'])

        info("Fluid __init__ done")

    def solve(self, t, n):
        self.t = t
        self.v_in.t = t
        self.dt.assign(np.min([self.fenics_dt, self.precice_dt]))
        #info("Solving...")
        self.solver.solve()
        #info("Solved.")

        # TODO
        #self.force = project(dot(self.T_f, self.n), self.U)
        #self.force = project(self.n, self.U)

        # precice coupling step
        t, n, precice_timestep_complete, self.precice_dt \
                = self.precice.advance(self.force, self.w, self.w0, t, self.dt.values()[0], n)

        return t, n, precice_timestep_complete

    def save(self, t):
        #info("Saving...")
        if self.mesh_move == 'biharmonic':
            (v, b1, u, b2, z, p) = self.w.split()
        else:
            (v, u, p) = self.w.split()

        v.rename("v", "velocity")
        u.rename("u", "displacement")
        p.rename("p", "pressure")
        self.vfile.write(v, t)
        self.ufile.write(u, t)
        self.pfile.write(p, t)

        # Compute drag and lift
        D_C = -assemble(self.force[0]*ds(_FLUID_CYLINDER))
        L_C = -assemble(self.force[1]*ds(_FLUID_CYLINDER))

        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.bndry, _FSI)
        Fbc.apply(w_.vector())
        D_F = -assemble(action(self.F_fluid,w_))
        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.bndry, _FSI)
        Fbc.apply(w_.vector())        
        L_F = -assemble(action(self.F_fluid,w_))

        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.bndry, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.bndry, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        D_FF = -assemble(action(self.F_fluid,w_))
        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.bndry, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.bndry, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        L_FF = -assemble(action(self.F_fluid,w_))


        #info("Extracting values")
        self.w.set_allow_extrapolation(True)
        pA_loc = self.p((A.x(), A.y()))
        pB_loc = self.p((B.x(), B.y()))
        pB_loc = self.p((B.x(), B.y()))
        Ax_loc = self.u[0]((A.x(), A.y()))
        Ay_loc = self.u[1]((A.x(), A.y()))
        self.w.set_allow_extrapolation(False)
        
        #info("collision for A.")
        pi = 0
        if self.bb.compute_first_collision(A) < 4294967295:
            #info("Collision found.")
            pi = 1
        else:
            #info("No collision on this process.")
            pA_loc = 0.0
            Ax_loc = 0.0
            Ay_loc = 0.0
        #info("MPI.Sum opperations.")
        pA = MPI.sum(comm, pA_loc) / MPI.sum(comm, pi)
        Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)
        Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)

        #info("Collision for B.")
        pi = 0
        if self.bb.compute_first_collision(B) < 4294967295:
            #info("Collision found.")
            pi = 1
        else:
            #info("No collision on thi process.")
            pB_loc = 0.0
        #info("MPI.Sum opperations for B.")
        pB = MPI.sum(comm, pB_loc) / MPI.sum(comm, pi)
        p_diff = pB - pA

        #info("{}, {}, {}, {}, {}, {}, {}, {}".format\
        if my_rank == 0:
            with open(result+'/data.csv', 'a') as data_file:
                writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
                writer.writerow([t, Ax, Ay, p_diff, D_C, D_F, D_FF, L_C, L_F, L_FF])
        #       (t, P, PI, Ax, Ay, p_diff, drag, lift))



# set problem and its discretization
parser = OptionParser()
parser.add_option("--benchmark", dest="benchmark", default='FSI3')
parser.add_option("--mesh", dest="mesh_name", default='mesh_Fluid_L1')
parser.add_option("--mesh_move", dest="mesh_move", default='pseudoelasticity')
#parser.add_option("--theta", dest="theta", default='0.5')
parser.add_option("--dt", dest="dt", default='0.001')
parser.add_option("--dt_scheme", dest="dt_scheme", default='CN')	# BE BE_CN

(options, args) = parser.parse_args()

# name of benchmark 
benchmark = options.benchmark

# name of mesh
mesh_name = options.mesh_name
relative_path_to_mesh = 'Fluid/'+mesh_name+'.h5'

# approah to mesh moving in fluid region
mesh_move = options.mesh_move

# value of theta to theta scheme for temporal discretization
#theta = Constant(options.theta)

# time step size
dt = options.dt

# time stepping scheme
dt_scheme = options.dt_scheme

# choose theta according to dt_scheme
if dt_scheme in ['BE', 'BE_CN']:
    theta = Constant(1.0)
elif dt_scheme == 'CN':
    theta = Constant(0.5)
else:
    raise ValueError('Invalid argument for dt_scheme')

# load mesh with boundary and domain markers
sys.path.append('.')
import utils_for_fluid_solver as utils

v_max, mu_f, rho_f, t_end, result = utils.get_benchmark_specification(benchmark)
t_end = 10.0

#(mesh, bndry, domains, interface, A, B) \
#        = marker.give_marked_mesh(mesh_coarseness = mesh_coarseness, refinement = True, ALE = True)
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


dx  = dx(domain=mesh)#, subdomain_data = domains)
ds  = ds(domain=mesh, subdomain_data = bndry)

################# export domains and bndry to xdmf for visualization in Paraview
# facet markers can be written directly
with XDMFFile("%s/mesh_bndry.xdmf" % result) as f:
    f.write(bndry)
############################################################################


fluid = Fluid(mesh, coupling_boundary, bndry, dt, theta, v_max, mu_f, rho_f, mesh_move, result)

t = 0.0
n = 0

# start solving coupled problem, the final time is defined in precice-config.xml
while fluid.precice.is_coupling_ongoing():
    # compute solution
    t, n, precice_timestep_complete = fluid.solve(t, n)

    if precice_timestep_complete and t % 0.01 < 1e-05:
        fluid.save(t)

fluid.precice.finalize()
