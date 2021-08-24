import numpy as np
import torch

from .material_collection import MaterialCollection


class LayerBuilder:
    def __init__(self, run_cfg) -> None:
        """Initializes the builder

        Args:
            run_cfg (DotMap): Run configuration
        """
        self.run_cfg = run_cfg

    def _setup_grid_and_materials(self,):
        """Utility function to setup essentials
        """
        # Get the grid ticks
        x0 = torch.linspace(0, 1, self.run_cfg.Nx)
        y0 = torch.linspace(0, 1, self.run_cfg.Ny)
        # Create a meshgrid from the grid ticks
        x, y = torch.meshgrid(x0, y0)

        eps = torch.zeros(
            self.run_cfg.Nx, self.run_cfg.Ny, self.run_cfg.N_freq, dtype=torch.cdouble
        )

        material_collection = MaterialCollection(self.run_cfg.target_frequencies)

        return eps, material_collection, x, y

    def build_uniform_layer(self, grid_material):
        """Generates a uniform layer.

        Args:
            grid_material (str): name of first material

        Returns:
            torch.tensor: Tensor with epsilon values for the layer.
        """
        eps, material_collection, _, _ = self._setup_grid_and_materials()

        eps[:, :] = material_collection[grid_material]

        return eps

    def build_squared_layer(self, grid_material, square_material, a=0.33):
        """Generates a square layer with a square hole in the middle.

        Args:
            grid_material (str): name of first material
            square_material (str): name of second material
            a (float, optional): Size of the square (0 to 1). Defaults to 0.33.

        Returns:
            torch.tensor: Tensor with epsilon values for the layer.
        """
        eps, material_collection, grid_x, grid_y = self._setup_grid_and_materials()

        eps[:, :] = material_collection[grid_material]

        # Define the square
        ind = np.logical_and(
            np.logical_and((grid_x > a), (grid_x < 1 - a)),
            np.logical_and((grid_y > a), (grid_y < 1 - a)),
        )

        eps[ind] = material_collection[square_material]
        return eps

    def build_circle_layer(
        self, grid_material, circle_material, radius=0.33, center=0.5
    ):
        """Generates a square layer with a square hole in the middle.

        Args:
            grid_material (str): name of first material
            circle_material (str): name of second material
            a (float, optional): Size of the square (0 to 1). Defaults to 0.33.

        Returns:
            torch.tensor: Tensor with epsilon values for the layer.
        """
        eps, material_collection, grid_x, grid_y = self._setup_grid_and_materials()

        eps[:, :] = material_collection[grid_material]

        # Define the circle
        ind = (grid_x - center) ** 2 + (grid_y - center) ** 2 < radius ** 2

        eps[ind] = material_collection[circle_material]

        return eps
