import pandas
import torch
import os, sys
from glob import glob

from loguru import logger


class MaterialCollection:
    """Database of materials included with NIDN"""

    epsilon_matrix = None
    material_names = []
    N_materials = 0

    def __init__(self, target_frequencies, to_skip=None):
        """Initalizes the MaterialCollection. Loads all materials from the data folder.

        Args:
            target_frequencies (list): Frequencies we are targeting. Closest ones in the data will be used.
            to_skip (list): List of materials to skip.
        """
        logger.trace("Initializing material collection")
        self.epsilon_matrix = None
        self.material_names = []

        if len(target_frequencies) == 0:
            raise ValueError("No target frequencies specified.")

        self.target_frequencies = target_frequencies
        self.materials_folder = os.path.dirname(__file__) + "/data/"
        self._load_materials_folder(to_skip)

    def __getitem__(self, key):
        """Given the name of a material will return the corresponding epsilon tensor.

        Args:
            key (str): Name of the material.

        Returns:
            torch.tensor: The corresponding epsilon tensor.
        """
        if key in self.material_names:
            return self.epsilon_matrix[self.material_names.index(key)]
        else:
            raise KeyError(
                f"Material '{key}' not found in the material collection. Available are: {self.material_names}"
            )

    def _load_materials_folder(self, to_skip=None):
        """Loads all csv files from folder "data"
        and sets up the EPS_MATRIX and MATERIAL_NAMES

        Args:
            to_skip (list): List of materials to skip.
        """
        logger.trace(f"Loading materials from folder: {self.materials_folder}")

        # Load materials data
        files = glob(self.materials_folder + "/*.csv")
        logger.debug("Found the following files")
        logger.debug(files)

        eps_list = []
        for file in files:
            file = file.replace("\\", "/")
            name = file.split("/")[-1].split(".csv")[0]

            # Skip materials we don't want
            if to_skip is not None and name in to_skip:
                logger.debug(f"Skipping material {name}")
                continue

            # Load material epsilon (permittivity)
            eps_list.append(self._load_material_data(file))

            # Remember file name as material name
            self.material_names.append(name)

        # Create a single tensor of all materials
        logger.trace("Creating material tensor")
        self.epsilon_matrix = torch.stack(eps_list).squeeze()  # .cuda()
        self.N_materials = len(self.material_names)

    def _load_material_data(self, name):
        """Loads data (wavelength, n, and k) from the passed csv file for the closest frequencies and returns epsilon (permittivity).

        Args:
            name (str): Path to csv.

        Returns:
            torch.tensor: Epsilon for the material (permittivity)
        """
        logger.debug(f"Loading material {name}")
        csv_data = pandas.read_csv(name, delimiter="\t")

        eps = []
        # Compute epsilon from n and k
        # see e.g. https://www.tf.uni-kiel.de/matwis/amat/elmat_en/kap_3/backbone/r3_7_2.html
        logger.trace("Computing epsilon from n and k for each target frequency")
        for freq in self.target_frequencies:
            wl = 1.0 / freq
            result_index = csv_data["Wavelength"].sub(wl).abs().idxmin()
            entry = csv_data.iloc[result_index]
            real = entry.n * entry.n - entry.k * entry.k
            imag = 2 * entry.n * entry.k
            eps.append([real + imag * 1.0j])

        eps = torch.tensor(eps)
        logger.trace(f"Epsilon for material {name} is: {eps}")
        return eps
