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
_INFLOW     = 1
_WALLS      = 2
_INTERFACE  = 3
_OUTFLOW    = 4

def give_gmsh_mesh(name):
    if not os.path.isfile(name):
        raise ValueError('Invalid name for gmsh mesh.')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)
    domains = MeshFunction('size_t', mesh, 2, mesh.domains())
    hdf.read(domains, '/domains')
    bndry = MeshFunction('size_t', mesh, 1)
    hdf.read(bndry, '/bndry')    

    return(mesh, bndry, A, B)

def generate_mesh(name, par=40, refinement=False):
    info("Generating mesh {} ...".format(name))

    # construct mesh
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) \
           - mshr.Circle(Point(gX, gY), g_radius, 20) \
           - mshr.Rectangle(Point(gX, gY - 0.5*gEH), Point(gX + g_radius + gEL, gY + 0.5*gEH)) 
    mesh = mshr.generate_mesh(geometry, par)

    if refinement:
        parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        for k in range(2):		# 2
            cf = MeshFunction('bool', mesh, 2)
            cf.set_all(False)
            z = 0.2 #0.15 # 0.08
            for c in cells(mesh):
                if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
                elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
                 and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                    cf[c] = True
            mesh = refine(mesh, cf)

    mesh.init()

    # Save mesh
    if __version__[:4] == '2018':
        hdf5 = HDF5File(MPI.comm_world, name, 'w')
    else:
        hdf5 = HDF5File(mpi_comm_world(), name, 'w')
    hdf5.write(mesh, "/mesh")

    return mesh

def give_marked_mesh(mesh_coarseness = 40, refinement = False):
    '''
    Loads/Generates mesh and defines boundary and domain classification functions.
    '''

    # Generate name of the mesh
    name = '../meshes/CFD_mesh{}'.format(mesh_coarseness)	# assume calling from code computing FSI
    if refinement:
        name += '_refined'
    name += '.h5'

    # Load existing mesh or generate new one
    if os.path.isfile(name):
        mesh = Mesh()
        if __version__[:4] == '2018':
            hdf5 = HDF5File(MPI.comm_world, name, 'r')
        else:
            hdf5 = HDF5File(mpi_comm_world(), name, 'r')
        hdf5.read(mesh, '/mesh', False)
    else:
        mesh = generate_mesh(name, mesh_coarseness, refine)

    # mark subdomains
    class Interface(SubDomain):
        def snap(self, x):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            if r <= g_radius:
                x[0] = gX + (g_radius/r) * (x[0] - gX)
                x[1] = gY + (g_radius/r) * (x[1] - gY)
        def inside(self, x, on_boundary):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            return( (r <= g_radius + DOLFIN_EPS 
                    or (near(x[1], gX - 0.5*gEH) and x[0]<0.600001 and x[0] > 0.2)
                    or (near(x[1], gX + 0.5*gEH) and x[0]<0.600001 and x[0] > 0.2)
                    or (near(x[0], 0.6) and x[1] < 0.210001 and x[1] > 0.1899999)
                    ) and on_boundary)

    interface = Interface() 

    # construct facet and domain markers
    bndry = MeshFunction('size_t', mesh, 1)		# boundary marker
    bndry.set_all(0)
    interface.mark(bndry, _INTERFACE)

    for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            if near(mp[0], 0.0):
                bndry[f] = _INFLOW 				
            elif near(mp[1], 0.0) or near(mp[1], gW):
                bndry[f] = _WALLS			
            elif near(mp[0], gL):
                bndry[f] = _OUTFLOW		
            elif bndry[f] != _INTERFACE:
                raise ValueError('Unclassified exterior facet with midpoint [%.3f, %.3f].' \
                    % (mp.x(), mp.y()))

    #info("\t -done") 
    return(mesh, bndry, A, B)

if __name__ == "__main__":
    
    for par in [30,40,50,60,70,80]:
        for refinement in [True, False]:
            name = 'CFD_mesh{}'.format(par)
            if refinement:
                name += '_refined'
            name += '.h5'
            
            generate_mesh(name, par, refinement)
