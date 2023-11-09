from dolfinx.io import XDMFFile
from mpi4py import MPI
import festim as F


class MeshXDMF(F.Mesh):
    """
    Mesh read from the XDMF files

    Args:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file

    Attributes:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file
        mesh (fenics.mesh.Mesh): the mesh
    """

    def __init__(self, volume_file, facet_file) -> None:
        self.volume_file = volume_file
        self.facet_file = facet_file

        volumes_file = XDMFFile(MPI.COMM_WORLD, self.volume_file, "r")
        mesh = volumes_file.read_mesh(name="Grid")

        super().__init__(mesh=mesh)

    def define_surface_markers(self):
        """Creates the surface markers

        Returns:
            dolfinx.MeshTags: the tags containing the surfacemarkers
        """
        facets_file = XDMFFile(MPI.COMM_WORLD, self.facet_file, "r")
        facet_meshtags = facets_file.read_meshtags(self.mesh, name="Grid")

        return facet_meshtags

    def define_volume_markers(self):
        """Creates the volume markers

        Returns:
            dolfinx.MeshTags: the tags containing the volume markers
        """
        volume_file = XDMFFile(MPI.COMM_WORLD, self.cell_file, "r")
        volume_meshtags = volume_file.read_meshtags(self.mesh, name="Grid")

        return volume_meshtags
