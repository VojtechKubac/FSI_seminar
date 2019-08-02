from dolfin import *
import mshr

# define domain
gW = 0.41
gL = 2.5

# define incomressible part
gX = 0.2		# x coordinate of the centre of the circle
gY = 0.2		# y coordinate of the centre of the circle
g_radius  = 0.05	# radius of the circle

# define compressible part
gEL = 0.35		# length of the elastic part (left end fully attached to the circle)
gEH = 0.02		# hight of the elastic part

_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

for par in [20, 30, 40, 50, 60, 70, 80, 90]:
    info("constructing mesh without refinement, with parameter %d." % par)
    # construct mesh
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - mshr.Circle(Point(gX, gY), g_radius, 20)
    geometry.set_subdomain(1, mshr.Rectangle(Point(gX, gY - 0.5*gEH), Point(gX + g_radius + gEL, gY + 0.5*gEH)))
    mesh = mshr.generate_mesh(geometry, par)
    mesh.init()

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

    hdf = HDF5File(mesh.mpi_comm(), "mesh{}_unrefined.h5".format(par), "w")
    hdf.write(mesh, "/mesh")

for par in [20, 30, 40, 50, 60, 70, 80, 90]:
    info("constructing mesh with refinement, with parameter %d." % par)
    # construct mesh
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - mshr.Circle(Point(gX, gY), g_radius, 20)
    geometry.set_subdomain(1, mshr.Rectangle(Point(gX, gY - 0.5*gEH), Point(gX + g_radius + gEL, gY + 0.5*gEH)))
    mesh = mshr.generate_mesh(geometry, par)
    mesh.init()

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

    # mesh refinement
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    for k in range(2):		# 2
        info("\trefinement level {}".format(k))
        cf = MeshFunction('bool', mesh, 2)
        cf.set_all(False)
        z = 0.08
        for c in cells(mesh):
            if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
            elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
             and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                cf[c] = True
        #cf[c]=True
        mesh = refine(mesh, cf)
        mesh.snap_boundary(cylinder, False)

    hdf = HDF5File(mesh.mpi_comm(), "mesh{}_refined.h5".format(par), "w")
    hdf.write(mesh, "/mesh")
