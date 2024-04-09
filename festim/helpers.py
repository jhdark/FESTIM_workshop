from dolfinx import fem
import ufl


def as_fenics_constant(value, mesh):
    """Converts a value to a dolfinx.Constant

    Args:
        value (float, int or dolfinx.Constant): the value to convert
        mesh (dolfinx.mesh.Mesh): the mesh of the domiain

    Returns:
        dolfinx.Constant: the converted value

    Raises:
        TypeError: if the value is not a float, an int or a dolfinx.Constant
    """
    if isinstance(value, (float, int)):
        return fem.Constant(mesh, float(value))
    elif isinstance(value, fem.Constant):
        return value
    else:
        raise TypeError(
            f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
        )


def as_fenics_expr(value, mesh, function_space, temperature=None, t=None):
    x = ufl.SpatialCoordinate(mesh)

    arguments = value.__code__.co_varnames

    if "t" in arguments and "x" not in arguments and "T" not in arguments:
        # only t is an argument
        if not isinstance(value(t=float(t)), (float, int)):
            raise ValueError(
                f"value should return a float or an int, not {type(value(t=float(t)))} "
            )
        return as_fenics_constant(mesh=mesh, value=value(t=float(t))), None
    else:
        kwargs = {}
        if "t" in arguments:
            kwargs["t"] = t
        if "x" in arguments:
            kwargs["x"] = x
        if "T" in arguments:
            kwargs["T"] = temperature

        for arg in arguments:
            if arg not in "txT":
                raise ValueError("unacceptable argument in value")

        return fem.Expression(
            value(**kwargs),
            function_space.element.interpolation_points(),
        )


def as_fenics_function(value, mesh, function_space, temperature=None, t=None):
    """Creates the value of the boundary condition as a fenics object and sets it to
    self.value_fenics.
    If the value is a constant, it is converted to a fenics.Constant.
    If the value is a function of t, it is converted to a fenics.Constant.
    Otherwise, it is converted to a fenics.Function and the
    expression of the function is stored in self.bc_expr.

    Args:
        mesh (dolfinx.mesh.Mesh) : the mesh
        function_space (dolfinx.fem.FunctionSpaceBase): the function space
        temperature (float): the temperature
        t (dolfinx.fem.Constant): the time

    Returns:
        dolfinx.Constant: the converted value

    """
    value_fenics_func = fem.Function(function_space)

    value_fenics_expr = as_fenics_expr(
        value=value,
        mesh=mesh,
        function_space=function_space,
        temperature=temperature,
        t=t,
    )

    value_fenics_func.interpolate(value_fenics_expr)

    return value_fenics_func, value_fenics_expr


def create_value_fenics(value, mesh, function_space, t=None, temperature=None):
    """Creates the value of the boundary condition as a fenics object and sets it to
    self.value_fenics.
    If the value is a constant, it is converted to a fenics.Constant.
    If the value is a function of t, it is converted to a fenics.Constant.
    Otherwise, it is converted to a fenics.Function and the
    expression of the function is stored in self.bc_expr.

    Args:
        mesh (dolfinx.mesh.Mesh) : the mesh
        function_space (dolfinx.fem.FunctionSpaceBase): the function space
        temperature (float): the temperature
        t (dolfinx.fem.Constant): the time
    """
    if isinstance(value, (int, float)):
        return as_fenics_constant(mesh=mesh, value=value), None

    elif isinstance(value, (fem.Constant, fem.Function, ufl.core.expr.Expr)):
        return value, None

    elif callable(value):
        return as_fenics_function(
            value=value,
            mesh=mesh,
            function_space=function_space,
            temperature=temperature,
            t=t,
        )
