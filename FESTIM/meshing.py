from fenics import *


def create_mesh(mesh_parameters):
    if "cells_file" in mesh_parameters.keys():
        # Read volumetric mesh
        mesh = Mesh()
        XDMFFile(volumetric_file).read(mesh)
    else:
        mesh = mesh_and_refine(mesh_parameters)
    return mesh


def subdomains(mesh, parameters):
    mesh_parameters = parameters["mesh_parameters"]
    if "cells_file" in mesh_parameters.keys():
        volume_markers, surface_markers = \
            read_subdomains_from_xdmf(
                mesh_parameters["cells_file"], mesh_parameters["facets_file"])
    else:
        size = parameters["mesh_parameters"]["size"]
        volume_markers, surface_markers = \
            subdomains_1D(mesh, parameters["materials"], size)
    return volume_markers, surface_markers


def read_subdomains_from_xdmf(volumetric_file, boundary_file):

    # Read tags for volume elements
    cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    XDMFFile(volumetric_file).read(cell_markers, "cell_tags")

    # Read tags for surface elements
    # (can also be used for applying DirichletBC)
    boundaries = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    XDMFFile(boundary_file).read(boundaries, "cell_tags")
    boundaries = MeshFunction("size_t", mesh, boundaries)

    print(len(cell_markers))
    return cell_markers, boundaries


def mesh_and_refine(mesh_parameters):
    '''
    Mesh and refine iteratively until meeting the refinement
    conditions.
    Arguments:
    - mesh_parameters : dict, contains initial number of cells, size,
    and refinements (number of cells and position)
    Returns:
    - mesh : the refined mesh.
    '''
    print('Meshing ...')
    initial_number_of_cells = mesh_parameters["initial_number_of_cells"]
    size = mesh_parameters["size"]
    mesh = IntervalMesh(initial_number_of_cells, 0, size)
    if "refinements" in mesh_parameters:
        for refinement in mesh_parameters["refinements"]:
            nb_cells_ref = refinement["cells"]
            refinement_point = refinement["x"]
            print("Mesh size before local refinement is " +
                  str(len(mesh.cells())))
            while len(mesh.cells()) < \
                    initial_number_of_cells + nb_cells_ref:
                cell_markers = MeshFunction(
                    "bool", mesh, mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in cells(mesh):
                    if cell.midpoint().x() < refinement_point:
                        cell_markers[cell] = True
                mesh = refine(mesh, cell_markers)
            print("Mesh size after local refinement is " +
                  str(len(mesh.cells())))
            initial_number_of_cells = len(mesh.cells())
    else:
        print('No refinement parameters found')
    return mesh


def subdomains_1D(mesh, materials, size):
    '''
    Iterates through the mesh and mark them
    based on their position in the domain
    Arguments:
    - mesh : the mesh
    - materials : list, contains the dictionaries of the materials
    Returns :
    - volume_markers : MeshFunction that contains the subdomains
        (0 if no domain was found)
    - measurement_dx : the measurement dx based on volume_markers
    - surface_markers : MeshFunction that contains the surfaces
    - measurement_ds : the measurement ds based on surface_markers
    '''
    volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in cells(mesh):
        for material in materials:
            if cell.midpoint().x() >= material['borders'][0] \
             and cell.midpoint().x() <= material['borders'][1]:
                volume_markers[cell] = material['id']
    surface_markers = MeshFunction(
        "size_t", mesh, mesh.topology().dim()-1, 0)
    surface_markers.set_all(0)
    i = 0
    for f in facets(mesh):
        i += 1
        x0 = f.midpoint()
        surface_markers[f] = 0
        if near(x0.x(), 0):
            surface_markers[f] = 1
        if near(x0.x(), size):
            surface_markers[f] = 2
    return volume_markers, surface_markers
