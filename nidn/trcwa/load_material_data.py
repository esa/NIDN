import pandas
import torch


def _load_material_data(name, target_frequencies=[1.0]):
    """Loads the passed wavelength, n, k data from the passed csv file for the closest frequencies and returns epsilon (permittivity).

    Args:
        name (str): Path to csv.
        target_frequencies (list, optional): Target frequencies to compute epsilon for. Defaults to [1.0].

    Returns:
        torch.tensor: Epsilon (complex permittivity) for the material
    """
    csv_data = pandas.read_csv(name, delimiter="\t")

    eps = []
    for freq in target_frequencies:
        wl = 1.0 / freq
        result_index = csv_data["Wavelength"].sub(wl).abs().idxmin()
        entry = csv_data.iloc[result_index]
        real = entry.n * entry.n - entry.k * entry.k
        imag = 2 * entry.n * entry.k
        eps.append([real + imag * 1.0j])

    eps = torch.tensor(eps)
    return eps
