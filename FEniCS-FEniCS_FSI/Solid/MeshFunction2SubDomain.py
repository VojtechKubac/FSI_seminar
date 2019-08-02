from dolfin import *

mesh = UnitSquareMesh(10, 10, 'left')

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

upper = Upper()

mesh_fun = MeshFunction('size_t', mesh, 1)
mesh_mark = MeshFunction('size_t', mesh, 1)

mesh_mark.set_all(0)
mesh_fun.set_all(0)

upper.mark(mesh_mark, 1)

for f in facets(mesh):
    if f.exterior():
        mp = f.midpoint()
        if near(mp[1], 1.0):
            mesh_fun[f] = 1

for f in facets(mesh):
    if mesh_fun[f] != mesh_mark[f]:
        print('mesh_fun[f] = {}, mesh_mark[f] = {} for facet f with midpoint ({}, {}).'.format(
            mesh_fun[f], mesh_mark[f], f.midpoint().x(), f.midpoint().y()))


class Upper2(SubDomain):
    def __init__(self, mesh_fun, mesh):
        SubDomain.__init__(self)
        self.SubDomain_facets = []
        for f in facets(mesh):
            if mesh_fun[f] == 1:
                self.SubDomain_facets.append(f)

    def inside(self, x, on_boundary):
        if on_boundary:
            for f in self.SubDomain_facets:
                mp = f.midpoint()
                if near(x[0], mp.x(), 5e-02) and near(x[1], mp.y(), 5e-02):
                    return True
        return False

mesh_mark2 = MeshFunction('size_t', mesh, 1)
mesh_mark2.set_all(0)

upper2 = Upper2(mesh_fun, mesh)
upper2.mark(mesh_mark2, 1)

for f in facets(mesh):
    if mesh_mark2[f] != mesh_mark[f]:
        print('mesh_mark2[f] = {}, mesh_mark[f] = {} for facet f with midpoint ({}, {}).'.format(
            mesh_mark2[f], mesh_mark[f], f.midpoint().x(), f.midpoint().y()))
