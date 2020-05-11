import FESTIM
from fenics import *
import sympy as sp


def run(parameters, log_level=40):
    # Export parameters
    if "parameters" in parameters["exports"].keys():
        try:
            FESTIM.export.export_parameters(parameters)
        except TypeError:
            pass

    set_log_level(log_level)

    # Check if transient
    transient = True
    if "type" in parameters["solving_parameters"].keys():
        if parameters["solving_parameters"]["type"] == "solve_transient":
            transient = True
        elif parameters["solving_parameters"]["type"] == "solve_stationary":
            transient = False
        elif "type" in parameters["solving_parameters"].keys():
            raise ValueError(
                str(parameters["solving_parameters"]["type"]) + ' unkown')

    # Declaration of variables
    dt = 0
    if transient:
        final_time = parameters["solving_parameters"]["final_time"]
        initial_stepsize = parameters["solving_parameters"]["initial_stepsize"]
        dt = Constant(initial_stepsize, name="dt")  # time step size

    # Mesh and refinement
    mesh = FESTIM.meshing.create_mesh(parameters["mesh_parameters"])

    # Define and mark subdomains
    volume_markers, surface_markers = \
        FESTIM.meshing.subdomains(mesh, parameters)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)

    # Define function space for system of concentrations and properties
    if "traps_element_type" in parameters["solving_parameters"].keys():
        trap_element = parameters["solving_parameters"]["traps_element_type"]
    else:
        trap_element = "CG"  # Default is CG
    V = FESTIM.functionspaces_and_functions.create_function_space(
        mesh, len(parameters["traps"]), element_trap=trap_element)
    W = FunctionSpace(mesh, 'CG', 1)  # function space for T and ext trap dens
    V_DG1 = FunctionSpace(mesh, 'DG', 1)

    # Define temperature
    T = Function(W, name="T")
    T_n = Function(W, name="T_n")
    if parameters["temperature"]["type"] == "expression":
        T_expr = Expression(
            sp.printing.ccode(
                parameters["temperature"]['value']), t=0, degree=2)
        T.assign(interpolate(T_expr, W))
        T_n.assign(T)
    else:
        # Define variational problem for heat transfers

        vT = TestFunction(W)
        if parameters["temperature"]["type"] == "solve_transient":
            T_ini = sp.printing.ccode(
                parameters["temperature"]["initial_condition"])
            T_ini = Expression(T_ini, degree=2, t=0)
            T_n.assign(interpolate(T_ini, W))
        bcs_T, expressions_bcs_T = \
            FESTIM.boundary_conditions.define_dirichlet_bcs_T(
                parameters, W, surface_markers)
        FT, expressions_FT = \
            FESTIM.formulations.define_variational_problem_heat_transfers(
                parameters, [T, vT, T_n], [dx, ds], dt)
        if parameters["temperature"]["type"] == "solve_stationary":
            print("Solving stationary heat equation")
            solve(FT == 0, T, bcs_T)

    # Create functions for properties
    D, thermal_cond, cp, rho, H, S =\
        FESTIM.post_processing.create_properties(
            mesh, parameters["materials"], volume_markers, T)

    # Define functions
    u = Function(V)
    concentrations = list(split(u))
    extrinsic_traps = [Function(W) for d in parameters["traps"]
                       if "type" in d.keys() if d["type"] == "extrinsic"]
    testfunctions_traps = [TestFunction(W) for d in parameters["traps"]
                           if "type" in d.keys() if d["type"] == "extrinsic"]

    v = TestFunction(V)
    testfunctions_concentrations = list(split(v))

    # Initialising the solutions
    if "initial_conditions" in parameters.keys():
        initial_conditions = parameters["initial_conditions"]
    else:
        initial_conditions = []
    u_n, concentrations_n = \
        FESTIM.initialise_solutions.initialising_solutions(
            parameters, V, S)
    previous_solutions_traps = \
        FESTIM.initialise_solutions.initialising_extrinsic_traps(
            W, len(extrinsic_traps))

    # Boundary conditions
    print('Defining boundary conditions')
    bcs, expressions = FESTIM.boundary_conditions.apply_boundary_conditions(
        parameters, V, [volume_markers, surface_markers], T)
    fluxes, expressions_fluxes = FESTIM.boundary_conditions.apply_fluxes(
        parameters, concentrations, testfunctions_concentrations, ds, T, S)

    # Define variational problem H transport
    print('Defining variational problem')
    F, expressions_F = FESTIM.formulations.formulation(
        parameters, extrinsic_traps,
        concentrations, testfunctions_concentrations,
        concentrations_n, dt, dx, T, T_n, transient=transient)
    F += fluxes

    du = TrialFunction(u.function_space())
    J = derivative(F, u, du)  # Define the Jacobian

    # Define variational problem for extrinsic traps
    if transient:
        extrinsic_formulations, expressions_form = \
            FESTIM.formulations.formulation_extrinsic_traps(
                parameters["traps"], extrinsic_traps, testfunctions_traps,
                previous_solutions_traps, dt)

    # Solution files
    files = []
    append = False
    if "xdmf" in parameters["exports"].keys():
        files = FESTIM.export.define_xdmf_files(parameters["exports"])

    derived_quantities_global = []
    if "derived_quantities" in parameters["exports"].keys():
        derived_quantities_global = \
            [FESTIM.post_processing.header_derived_quantities(parameters)]

    t = 0  # Initialising time to 0s
    timer = Timer()  # start timer

    if transient:
        #  Time-stepping
        print('Time stepping...')
        while t < final_time:
            # Update current time
            t += float(dt)
            FESTIM.helpers.update_expressions(
                expressions, t)
            FESTIM.helpers.update_expressions(
                expressions_form, t)
            FESTIM.helpers.update_expressions(
                expressions_F, t)
            FESTIM.helpers.update_expressions(
                expressions_fluxes, t)
            if parameters["temperature"]["type"] != "expression":
                FESTIM.helpers.update_expressions(
                    expressions_FT, t)
                FESTIM.helpers.update_expressions(
                    expressions_bcs_T, t)

            else:
                T_n.assign(T)
                T_expr.t = t
                T.assign(interpolate(T_expr, W))
            D._T = T
            if H is not None:
                H._T = T
            if thermal_cond is not None:
                thermal_cond._T = T
            if S is not None:
                S._T = T
                for expr in expressions:
                    if "_bci" in expr.__dict__.keys():
                        expr._bci.t = t
            # Display time
            print(str(round(t/final_time*100, 2)) + ' %        ' +
                  str(round(t, 1)) + ' s' +
                  "    Ellapsed time so far: %s s" %
                  round(timer.elapsed()[0], 1),
                  end="\r")

            # Solve heat transfers
            if parameters["temperature"]["type"] == "solve_transient":
                dT = TrialFunction(T.function_space())
                JT = derivative(FT, T, dT)  # Define the Jacobian
                problem = NonlinearVariationalProblem(FT, T, bcs_T, JT)
                solver = NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]["absolute_tolerance"] = \
                    1e-3
                solver.parameters["newton_solver"]["relative_tolerance"] = \
                    1e-10
                solver.solve()
                T_n.assign(T)

            # Solve main problem
            FESTIM.solving.solve_it(
                F, u, J, bcs, t, dt, parameters["solving_parameters"])

            # Solve extrinsic traps formulation
            for j in range(len(extrinsic_formulations)):
                solve(extrinsic_formulations[j] == 0, extrinsic_traps[j], [])

            # Post processing
            FESTIM.post_processing.run_post_processing(
                parameters,
                transient,
                u, T,
                [volume_markers, surface_markers],
                W, V_DG1,
                t,
                dt,
                files,
                append,
                [D, thermal_cond, cp, rho, H, S],
                derived_quantities_global)
            append = True

            # Update previous solutions
            u_n.assign(u)
            for j in range(len(previous_solutions_traps)):
                previous_solutions_traps[j].assign(extrinsic_traps[j])
    else:
        # Solve steady state
        print('Solving steady state problem...')

        du = TrialFunction(u.function_space())
        FESTIM.solving.solve_once(
            F, u, J, bcs, parameters["solving_parameters"])

        # Post processing
        FESTIM.post_processing.run_post_processing(
            parameters,
            transient,
            u, T,
            [volume_markers, surface_markers],
            W, V_DG1,
            t,
            dt,
            files,
            append,
            [D, thermal_cond, cp, rho, H, S],
            derived_quantities_global)

    # Store data in output
    output = dict()  # Final output

    # Compute error
    if u.function_space().num_sub_spaces() == 0:
        res = [u]
    else:
        res = list(u.split())
    if "error" in parameters["exports"].keys():
        if S is not None:
            solute = project(res[0]*S, V_DG1)
            res[0] = solute
        error = FESTIM.post_processing.compute_error(
            parameters["exports"]["error"], t, [*res, T], mesh)
        output["error"] = error
    output["parameters"] = parameters
    output["mesh"] = mesh
    if "derived_quantities" in parameters["exports"].keys():
        output["derived_quantities"] = derived_quantities_global
        FESTIM.export.write_to_csv(parameters["exports"]["derived_quantities"],
                                   derived_quantities_global)

    # End
    print('\007s')
    return output
