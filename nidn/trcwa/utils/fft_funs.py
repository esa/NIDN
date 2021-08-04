import torch


def Epsilon_fft(dN, eps_grid, G):
    """dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2

    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    """

    if len(eps_grid) == 3 and eps_grid[0].ndim == 2:
        epsx_fft = get_conv(dN, eps_grid[0], G)
        epsy_fft = get_conv(dN, eps_grid[1], G)
        epsz_fft = get_conv(dN, eps_grid[2], G)
        epsinv = torch.inverse(epsz_fft)

        tmp1 = torch.vstack((epsx_fft, torch.zeros_like(epsx_fft)))
        tmp2 = torch.vstack((torch.zeros_like(epsx_fft), epsy_fft))
        eps2 = torch.hstack((tmp1, tmp2))

    elif eps_grid[0].ndim == 1:
        eps_fft = get_conv(dN, eps_grid, G)
        epsinv = torch.inverse(eps_fft)

        tmp1 = torch.vstack((eps_fft, torch.zeros_like(eps_fft)))
        tmp2 = torch.vstack((torch.zeros_like(eps_fft), eps_fft))
        eps2 = torch.hstack((tmp1, tmp2))
    else:
        raise ValueError("Wrong eps_grid type")

    # Set very small components to zero for torch
    for i in range(eps2.shape[0]):
        for j in range(eps2.shape[1]):
            if eps2[i][j].real < 1e-14 and eps2[i][j].imag < 1e-14:
                eps2[i][j] = 0 + 0j

    return epsinv, eps2


def get_conv(dN, s_in, G):
    """Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    """
    nG, _ = G.shape
    sfft = torch.fft.fft2(s_in) * dN

    ix = range(nG)
    ii, jj = torch.meshgrid(torch.tensor(ix), torch.tensor(ix))
    s_out = sfft[G[ii, 0] - G[jj, 0], G[ii, 1] - G[jj, 1]]
    return s_out


def get_fft(dN, s_in, G):
    """
    FFT to get Fourier components

    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    """

    sfft = torch.fft.fft2(s_in) * dN
    return sfft[G[:, 0], G[:, 1]]


def get_ifft(Nx, Ny, s_in, G):
    """
    Reconstruct real-space fields
    """
    dN = 1.0 / Nx / Ny
    nG, _ = G.shape

    s0 = torch.zeros((Nx, Ny), dtype=complex)
    for i in range(nG):
        x = G[i, 0]
        y = G[i, 1]

        stmp = torch.zeros((Nx, Ny), dtype=complex)
        stmp[x, y] = 1.0
        s0 = s0 + s_in[i] * stmp

    s_out = torch.ifft2(s0) / dN
    return s_out
