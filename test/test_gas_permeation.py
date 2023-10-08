from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    FunctionSpace,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
    assemble_scalar,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
)
from dolfinx.nls.petsc import NewtonSolver
from ufl import (
    dot,
    FiniteElement,
    grad,
    TestFunction,
    exp,
    FacetNormal,
    dx,
    Cell,
    Mesh,
    VectorElement,
    Measure,
)
from dolfinx.mesh import (
    create_mesh,
    meshtags,
)
from dolfinx import log
import numpy as np


def test_example():
    # mesh nodes
    indices = np.linspace(0, 3e-4, num=1001)

    gdim, shape, degree = 1, "interval", 1
    cell = Cell(shape, geometric_dimension=gdim)
    domain = Mesh(VectorElement("Lagrange", cell, degree))
    mesh_points = np.reshape(indices, (len(indices), 1))
    indexes = np.arange(mesh_points.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
    my_mesh = create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)

    elements = FiniteElement("CG", my_mesh.ufl_cell(), 1)
    V = FunctionSpace(my_mesh, elements)
    u = Function(V)
    u_n = Function(V)
    v = TestFunction(V)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], indices[-1]))

    dofs_facets = np.array([dofs_L[0], dofs_R[0]], dtype=np.int32)
    tags_facets = np.array([1, 2], dtype=np.int32)

    facet_dimension = my_mesh.topology.dim - 1

    mesh_tags_facets = meshtags(my_mesh, facet_dimension, dofs_facets, tags_facets)
    ds = Measure("ds", domain=my_mesh, subdomain_data=mesh_tags_facets)

    temperature = 500
    k_B = 8.6173303e-5
    n = FacetNormal(my_mesh)

    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / k_B / T)
        return S * pressure**0.5

    fdim = my_mesh.topology.dim - 1
    left_facets = mesh_tags_facets.find(1)
    left_dofs = locate_dofs_topological(V, fdim, left_facets)
    right_facets = mesh_tags_facets.find(2)
    right_dofs = locate_dofs_topological(V, fdim, right_facets)

    surface_conc = siverts_law(T=temperature, S_0=4.02e21, E_S=1.04, pressure=100)
    bc_sieverts = dirichletbc(
        Constant(my_mesh, PETSc.ScalarType(surface_conc)), left_dofs, V
    )
    bc_outgas = dirichletbc(Constant(my_mesh, PETSc.ScalarType(0)), right_dofs, V)
    bcs = [bc_sieverts, bc_outgas]

    D_0 = 1.9e-7
    E_D = 0.2

    D = D_0 * exp(-E_D / k_B / temperature)

    dt = 1 / 20
    final_time = 50
    num_steps = int(final_time / dt)

    F = dot(D * grad(u), grad(v)) * dx
    F += ((u - u_n) / dt) * v * dx

    problem = NonlinearProblem(F, u, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e10

    flux_values = []
    times = []
    t = 0

    for i in range(num_steps):
        t += dt

        solver.solve(u)

        surface_flux = form(D * dot(grad(u), n) * ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)
        np.savetxt("outgassing_flux.txt", np.array(flux_values))
        np.savetxt("times.txt", np.array(times))

        u_n.x.array[:] = u.x.array[:]
