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
        raise ValueError('{} is an invalid name for gmsh mesh.'.format(name))

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)
    domains = MeshFunction('size_t', mesh, 2, mesh.domains())
    #domains.set_all(0)
    hdf.read(domains, '/domains')
    #hdf.read(domains, '/subdomains')
    bndry = MeshFunction('size_t', mesh, 1)
    #bndry.set_all(0)
    hdf.read(bndry, '/bndry')    
    #hdf.read(bndry, '/boundaries')    
    interface = MeshFunction('size_t', mesh, 1)
    interface.set_all(0)
    
    for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            if bndry[f] == _CIRCLE and not(mp[0] > gX and mp[1] < gY + 0.5*gEH + DOLFIN_EPS \
                    and mp[1] > gY - 0.5*gEH - DOLFIN_EPS):
                interface[f] = _FLUID_CYLINDER
        else:
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag |= 1
                if domains[c] == 1:
                    flag |= 2
            if flag == 3:
                interface[f] = _FSI

    return(mesh, bndry, domains, interface, A, B)


def generate_mesh(name, par=40, refinement=False, ALE=True):

    info('Generating mesh {} ...'.format(name))

    # construct mesh
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - mshr.Circle(Point(gX, gY), g_radius, 20)
    if ALE:
        geometry.set_subdomain(1, mshr.Rectangle(Point(gX, gY - 0.5*gEH), 
            Point(gX + g_radius + gEL, gY + 0.5*gEH)))
    mesh = mshr.generate_mesh(geometry, par)

    if refinement:
        parameters['refinement_algorithm'] = 'plaza_with_parent_facets'
        for k in range(2):		# 2
            cf = MeshFunction('bool', mesh, 2)
            cf.set_all(False)
            z = 0.08
            for c in cells(mesh):
                if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
                elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
                 and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                    cf[c] = True
            mesh = refine(mesh, cf)

    mesh.init()

    # Save mesh
    if __version__[:4] == '2017':
        hdf5 = HDF5File(mpi_comm_world(), name, 'w')
    else:
        hdf5 = HDF5File(MPI.comm_world, name, 'w')
    hdf5.write(mesh, '/mesh')

    return(mesh)

def give_marked_mesh(mesh_coarseness = 40, refinement = False, ALE = True):
    '''
    Loads/Generates mesh and defines boundary and domain classification functions.
    If ALE == True, then the mesh fits the initial position of elastic beam,
    otherwise it is ignored.
    '''

    # Generate name of the mesh
    name = '../meshes/mesh{}'.format(mesh_coarseness)	# assume calling from code computing FSI
    if refinement:
        name += '_refined'
    if ALE:
        name += '_ALE'
    name += '.h5'

    # Load existing mesh or generate new one
    if os.path.isfile(name):
        mesh = Mesh()
        if __version__[:4] == '2017':
            hdf5 = HDF5File(mpi_comm_world(), name, 'r')
        else:
            hdf5 = HDF5File(MPI.comm_world, name, 'r')
        hdf5.read(mesh, '/mesh', False)
    else:
        mesh = generate_mesh(name, mesh_coarseness, refine, ALE)

    # mark subdomains
    class Cylinder(SubDomain):
        def snap(self, x):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            if r <= g_radius:
                x[0] = gX + (g_radius/r) * (x[0] - gX)
                x[1] = gY + (g_radius/r) * (x[1] - gY)
        def inside(self, x, on_boundary):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            return( (r <= g_radius + DOLFIN_EPS) and on_boundary)

    cylinder = Cylinder()

    class Elasticity(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
             and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS)

    elasticity = Elasticity()

    # construct facet and domain markers
    bndry     = MeshFunction('size_t', mesh, 1)		# boundary marker 
    interface = MeshFunction('size_t', mesh, 1)		# boundary marker 
    domains   = MeshFunction('size_t', mesh, 2, mesh.domains())
    bndry.set_all(0)
    interface.set_all(0)
    domains.set_all(0)
    elasticity.mark(domains, 1)

    cylinder.mark(bndry, _CIRCLE)
    for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            if near(mp[0], 0.0):
                bndry[f] = _INFLOW 				
            elif near(mp[1], 0.0) or near(mp[1], gW):
                bndry[f] = _WALLS			
            elif near(mp[0], gL):
                bndry[f] = _OUTFLOW		
            elif bndry[f] == _CIRCLE and not(mp[0] > gX and mp[1] < gY + 0.5*gEH + DOLFIN_EPS \
                    and mp[1] > gY - 0.5*gEH - DOLFIN_EPS):
                interface[f] = _FLUID_CYLINDER
            else:
                if bndry[f] != _CIRCLE:
                    raise RuntimeError('Unclassified exterior facet with midpoint [%.3f, %.3f].' \
                        % (mp.x(), mp.y()))
        else:
        #if 1:
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag |= 1
                if domains[c] == 1:
                    flag |= 2
            if flag == 3:
                interface[f] = _FSI

    #for f in facets(elasticity):
    #    flag = 0
    #    for c in cells(f):
    #        if domain[c] == 0:
    #            flag = 1
    #    if flag == 1:
    #        interface[f] = 1

    #info('\t -done') 
    return(mesh, bndry, domains, interface, A, B)

def give_marked_multimesh(background_coarseness = 40, elasticity_coarseness = 40, refine = False):
    # generate multimesh
    approx_circle_with_edges = 20
    beam_mesh_length = g_radius + 1.5*gEL
    beam_mesh_width  = 2.0*g_radius
    beam_mesh_X = gX
    beam_mesh_Y = gY - 2.0*g_radius


    multimesh = MultiMesh()
    bg_geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - \
                     mshr.Circle(Point(gX, gY), g_radius, approx_circle_with_edges)
    bg_mesh = mshr.generate_mesh(bg_geometry, background_coarseness)
    if refine:
        parameters['refinement_algorithm'] = 'plaza_with_parent_facets'
        for k in range(2):		# 2
            cf = MeshFunction('bool', bg_mesh, 2)
            cf.set_all(False)
            z = 0.08
            for c in cells(bg_mesh):
                if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
                elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
                 and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                    cf[c] = True
            bg_mesh = refine(bg_mesh, cf)

    multimesh.add(bg_mesh)

    beam_geometry = mshr.Rectangle(Point(beam_mesh_X,beam_mesh_Y ), \
                Point(beam_mesh_X + beam_mesh_length, beam_mesh_Y + beam_mesh_width)) \
                - mshr.Circle(Point(gX, gY), g_radius, approx_circle_with_edges)
    beam_mesh = mshr.generate_mesh(beam_geometry, elasticity_coarseness)
    multimesh.add(beam_mesh)
    multimesh.build()

    # define boundary and subdomain functions
    class Cylinder(SubDomain):
        def snap(self, x):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            if r <= g_radius:
                x[0] = gX + (g_radius/r) * (x[0] - gX)
                x[1] = gY + (g_radius/r) * (x[1] - gY)
        def inside(self, x, on_boundary):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            return( (r <= g_radius + DOLFIN_EPS) and on_boundary)

    cylinder = Cylinder()

    class Inflow(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and near(x[0], 0.0))

    inflow_bndry = Inflow()

    class Outflow(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and near(x[0], gL))

    outflow_bndry = Outflow()

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and (near(x[1], 0.0) or near(x[1], gW)))

    walls = Walls()

    class Elasticity(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
             and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS)

    elasticity = Elasticity()

    class ALE_Fluid(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] >= beam_mesh_X - DOLFIN_EPS \
                    and x[0] <= beam_mesh_X + beam_mesh_length + DOLFIN_EPS\
                    and x[1] >= beam_mesh_Y - DOLFIN_EPS \
                    and x[1] <= beam_mesh_Y + beam_mesh_width + DOLFIN_EPS \
                    and not (x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
                      and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS
                    ))

    ale_fluid = ALE_Fluid()

    ALE_domains = MeshFunction('size_t', multimesh.part(1), 1)
    ALE_domains.set_all(0)
    elasticity.mark(ALE_domains, 1)

    Eulerian_fluid = MeshFunction('size_t', multimesh.part(0), 1)
    Eulerian_fluid.set_all(0)

    FS_interface = MeshFunction('size_t', multimesh.part(1), 1)
    FS_interface.set_all(0)
    for f in facets(multimesh.part(1)):
        if not f.exterior():
            flag = 0
            for c in cells(f):
                if ALE_domains[c] == 0:
                    flag |= 1  
                if ALE_domains[c] == 1:
                    flag |= 2
            if flag == 3:
                FS_interface[f] = 1
    

    return(multimesh, 
            inflow_bndry, outflow_bndry, walls, cylinder, 
            ALE_domains, Eulerian_fluid, FS_interface, A, B)

if __name__ == '__main__':
    
    for par in [30,40,50,60,70,80]:
        for refinement in [True, False]:
            for ALE in [True, False]:
                name = 'mesh{}'.format(par)
                if refinement:
       	            name += '_refined'
                if ALE:
                    name += '_ALE'
                name += '.h5'
                
                generate_mesh(name, par, refinement, ALE)
