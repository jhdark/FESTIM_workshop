import festim as F
import numpy as np


class MaximumVolume(F.VolumeQuantity):
    """Computes the maximum value of a field in a given volume

    Args:
        field (festim.Species): species for which the maximum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the maximum volume is exported

    Attributes:
        see `festim.VolumeQuantity`

    """

    def __init__(
        self,
        field: F.Species,
        volume: F.VolumeSubdomain,
        filename: str = None,
    ) -> None:
        super().__init__(field=field, volume=volume, filename=filename)

    @property
    def title(self):
        return f"Maximum {self.field.name} volume {self.volume.id}"

    def compute(self):
        """
        Computes the maximum value of solution function within the defined volume
        subdomain, and appends it to the data list
        """
        self.value = np.max(self.field.solution.x.array[self.volume.entities])
        self.data.append(self.value)
