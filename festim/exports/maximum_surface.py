import numpy as np

import festim as F


class MaximumSurface(F.SurfaceQuantity):
    """Computes the maximum value of a field on a given surface

    Args:
        field (festim.Species): species for which the maximum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the maximum surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Maximum {self.field.name} surface {self.surface.id}"

    def compute(self):
        """
        Computes the maximum value of the field on the defined surface
        subdomain, and appends it to the data list
        """
        self.value = np.max(self.field.solution.x.array[self.surface.indices])
        self.data.append(self.value)
