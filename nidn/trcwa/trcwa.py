import torch

from nidn.utils.global_constants import PI

from .constants import *

from .utils.fft_funs import Epsilon_fft, get_ifft
from .utils.kbloch import Lattice_Reciprocate, Lattice_getG, Lattice_SetKs
from .utils.torch_functions import torch_transpose, torch_dot, torch_eye, torch_zeros


class TRCWA:
    def __init__(self, nG, L1, L2, freq, theta, phi, verbose=1):
        """Initializes the TRCWA class.

        The time harmonic convention is exp(-i omega t), speed of light = 1.

        Two kinds of layers are currently supported: uniform layer,
        patterned layer from grids. Interface for patterned layer by
        direct analytic expression of Fourier series is included, but
        no examples inclded so far.

        Args:
            nG (int): Truncation order. Note: the actual truncation order might not be nG but smaller.
            L1, L2 (float): Lattice vectors in the x and y direction, respectively. Defines the base unit of the whole system.
            freq (float): The "unitless" frequency, given in relation to the lattice vectors. For a better understanding, read https://web.stanford.edu/group/fan/S4/units.html#length-time-and-frequency.
            theta (float): The polar angle in radians (0 <= theta <= pi).
            phi (float): The azimuthal angle in radians (0 <= theta < 2pi).
            verbose (int, optional): If verbose > 0, the actual nG is printed. Defaults to 1.
        """
        self.freq = freq
        self.omega = 2 * PI * freq + 0.0j
        self.L1 = L1
        self.L2 = L2
        self.phi = torch.tensor(phi)
        self.theta = torch.tensor(theta)
        self.nG = nG
        self.verbose = verbose
        self.Layer_N = 0  # total number of layers

        # the length of the following variables = number of total layers
        self.thickness_list = []
        self.id_list = (
            []
        )  # [type, No., No. in patterned/uniform, No. in its family] starting from 0
        # type:0 for uniform, 1 for Grids, 2 for Fourier

        self.kp_list = []
        self.q_list = []  # eigenvalues
        self.phi_list = []  # eigenvectors

        # Uniform layer
        self.Uniform_ep_list = []
        self.Uniform_N = 0

        # Patterned layer
        self.Patterned_N = 0  # total number of patterned layers
        self.Patterned_epinv_list = []
        self.Patterned_ep2_list = []

        # patterned layer from Grids
        self.GridLayer_N = 0
        self.GridLayer_Nxy_list = []

        # layers of analytic Fourier series (e.g. circles)
        self.FourierLayer_N = 0
        self.FourierLayer_params = []

    def Add_LayerUniform(self, thickness, epsilon):
        # assert type(thickness) == float, 'thickness should be a float'

        self.id_list.append([0, self.Layer_N, self.Uniform_N])
        self.Uniform_ep_list.append(epsilon)
        self.thickness_list.append(torch.tensor(thickness))

        self.Layer_N += 1
        self.Uniform_N += 1

    def Add_LayerGrid(self, thickness, Nx, Ny):
        self.thickness_list.append(thickness)
        self.GridLayer_Nxy_list.append([Nx, Ny])
        self.id_list.append([1, self.Layer_N, self.Patterned_N, self.GridLayer_N])

        self.Layer_N += 1
        self.GridLayer_N += 1
        self.Patterned_N += 1

    def Add_LayerFourier(self, thickness, params):
        self.thickness_list.append(thickness)
        self.FourierLayer_params.append(params)
        self.id_list.append([2, self.Layer_N, self.Patterned_N, self.FourierLayer_N])

        self.Layer_N += 1
        self.Patterned_N += 1
        self.FourierLayer_N += 1

    def Init_Setup(self, Pscale=1.0, Gmethod=0):
        """Set up the reciprocal lattice, compute eigenvalues for uniform layers, and initialize vectors for patterned layers.

        Args:
            Pscale (float, optional): To scale the periodicity in both lateral directions simultaneously (as an autogradable parameter). Period will be Pscale*Lx and Pscale*Ly.
            Gmethod (int, optional): Fourier space truncation scheme; 0 for circular, 1 for rectangular.
        """
        kx0 = (
            self.omega
            * torch.sin(self.theta)
            * torch.cos(self.phi)
            * torch.sqrt(self.Uniform_ep_list[0])
        )
        ky0 = (
            self.omega
            * torch.sin(self.theta)
            * torch.sin(self.phi)
            * torch.sqrt(self.Uniform_ep_list[0])
        )

        # set up reciprocal lattice
        self.Lk1, self.Lk2 = Lattice_Reciprocate(self.L1, self.L2)
        self.G, self.nG = Lattice_getG(self.nG, self.Lk1, self.Lk2, method=Gmethod)

        self.Lk1 = self.Lk1 / Pscale
        self.Lk2 = self.Lk2 / Pscale
        # self.kx = kx0 + 2*torch.pi*(self.Lk1[0]*self.G[:,0]+self.Lk2[0]*self.G[:,1])
        # self.ky = ky0 + 2*torch.pi*(self.Lk1[1]*self.G[:,0]+self.Lk2[1]*self.G[:,1])
        self.kx, self.ky = Lattice_SetKs(self.G, kx0, ky0, self.Lk1, self.Lk2)

        # normalization factor for energies off normal incidence
        self.normalization = torch.sqrt(self.Uniform_ep_list[0]) / torch.cos(self.theta)

        # if comm.rank == 0 and verbose>0:
        if self.verbose > 0:
            print("Total nG = ", self.nG)

        self.Patterned_ep2_list = [None] * self.Patterned_N
        self.Patterned_epinv_list = [None] * self.Patterned_N
        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega, 0, 1.0 / ep, self.kx, self.ky)
                self.kp_list.append(kp)

                q, phi = SolveLayerEigensystem_uniform(self.omega, self.kx, self.ky, ep)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)

    def MakeExcitationPlanewave(
        self, p_amp, p_phase, s_amp, s_phase, order=0, direction="forward"
    ):
        """Sets the excitation to be a planewave incident upon the front (first layer specified) or back of the structure.
        If both tilt angles are specified to be zero, then the planewave is normally incident with the electric field
        polarized along the x-axis for the p-polarization. The phase of each polarization is defined at the origin (z = 0).

        Args:
            p_amp (complex float): The electric field amplitude of the p-polarizations of the planewave. Is a complex number with absolute value between 0 and 1.
            p_phase (float): Phase of the p-polarized part of the light.
            s_amp (complex float): The electric field amplitude of the s-polarizations of the planewave. Is a complex number with absolute value between 0 and 1.
            s_phase (float): Phase of the s-polarized part of the light.
            order (int, optional): A positive integer specifying which order (mode index) to excite. Defaults to 0.
            direction (string, optional): The direction of the planewave (forward is incident upon the front). Defaults to "forward".
        """
        self.direction = direction
        theta = self.theta
        phi = self.phi
        a0 = torch_zeros(2 * self.nG, dtype=complex)
        bN = torch_zeros(2 * self.nG, dtype=complex)
        if direction == "forward":
            tmp1 = torch_zeros(2 * self.nG, dtype=complex)
            tmp1[order] = 1.0
            a0 = a0 + tmp1 * (
                -s_amp
                * torch.cos(theta)
                * torch.cos(phi)
                * torch.exp(torch.tensor(1j * s_phase))
                - p_amp * torch.sin(phi) * torch.exp(torch.tensor(1j * p_phase))
            )

            tmp2 = torch_zeros(2 * self.nG, dtype=complex)
            tmp2[order + self.nG] = 1.0
            a0 = a0 + tmp2 * (
                -s_amp
                * torch.cos(theta)
                * torch.sin(phi)
                * torch.exp(torch.tensor(1j * s_phase))
                + p_amp * torch.cos(phi) * torch.exp(torch.tensor(1j * p_phase))
            )
        elif direction == "backward":
            tmp1 = torch_zeros(2 * self.nG, dtype=complex)
            tmp1[order] = 1.0
            bN = bN + tmp1 * (
                -s_amp
                * torch.cos(theta)
                * torch.cos(phi)
                * torch.exp(torch.tensor(1j * s_phase))
                - p_amp * torch.sin(phi) * torch.exp(torch.tensor(1j * p_phase))
            )

            tmp2 = torch_zeros(2 * self.nG, dtype=complex)
            tmp2[order + self.nG] = 1.0
            bN = bN + tmp2 * (
                -s_amp
                * torch.cos(theta)
                * torch.sin(phi)
                * torch.exp(torch.tensor(1j * s_phase))
                + p_amp * torch.cos(phi) * torch.exp(torch.tensor(1j * p_phase))
            )

        self.a0 = a0
        self.bN = bN

    def GridLayer_geteps(self, ep_all):
        """Fourier transform + eigenvalue for grid layer.

        Args:
            ep_all (torch.tensor): A tensor containing all epsilon values.
        """
        ptri = 0
        ptr = 0
        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue

            Nx = self.GridLayer_Nxy_list[ptri][0]
            Ny = self.GridLayer_Nxy_list[ptri][1]
            dN = 1.0 / Nx / Ny

            if len(ep_all) == 3 and ep_all[0].ndim > 0:
                ep_grid = [
                    torch.reshape(ep_all[0][ptr : ptr + Nx * Ny], [Nx, Ny]),
                    torch.reshape(ep_all[1][ptr : ptr + Nx * Ny], [Nx, Ny]),
                    torch.reshape(ep_all[2][ptr : ptr + Nx * Ny], [Nx, Ny]),
                ]
            elif len(ep_all.shape) == 3:
                ep_grid = ep_all[:, :, i - 1]
            else:
                ep_grid = torch.reshape(
                    torch.tensor(ep_all[ptr : ptr + Nx * Ny]), [Nx, Ny]
                )

            epinv, ep2 = Epsilon_fft(dN, ep_grid, self.G)

            self.Patterned_epinv_list[self.id_list[i][2]] = epinv
            self.Patterned_ep2_list[self.id_list[i][2]] = ep2

            kp = MakeKPMatrix(self.omega, 1, epinv, self.kx, self.ky)
            self.kp_list[self.id_list[i][1]] = kp

            q, phi = SolveLayerEigensystem(self.omega, self.kx, self.ky, kp, ep2)
            self.q_list[self.id_list[i][1]] = q
            self.phi_list[self.id_list[i][1]] = phi

            ptr += Nx * Ny
            ptri += 1

    def Return_eps(self, which_layer, Nx, Ny, component="xx"):
        """Used to get real-space epsilon profile reconstructured from the truncated Fourier orders.

        Args:
            which_layer (int): Which layer to return the epsilon values from.
            Nx (int): Grid points within the unit cell in the x direction.
            Ny (int): Grid points within the unit cell in the y direction.
            component (string, optional): For patterned layer, component = 'xx','xy','yx','yy','zz'. For uniform layer, currently it's assumed to be isotropic. Defaults to "xx".

        Returns:
            torch.tensor: Real-space epsilon profile
        """
        i = which_layer
        # uniform layer
        if self.id_list[i][0] == 0:
            ep = self.Uniform_ep_list[self.id_list[i][2]]
            return torch.ones((Nx, Ny)) * ep

        # patterned layer
        elif self.id_list[i][0] == 1:
            if component == "zz":
                epk = torch.inv(self.Patterned_epinv_list[self.id_list[i][2]])
            elif component == "xx":
                epk = self.Patterned_ep2_list[self.id_list[i][2]][: self.nG, : self.nG]
            elif component == "xy":
                epk = self.Patterned_ep2_list[self.id_list[i][2]][: self.nG, self.nG :]
            elif component == "yx":
                epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG :, : self.nG]
            elif component == "yy":
                epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG :, self.nG :]

            return get_ifft(Nx, Ny, epk[0, :], self.G)

    def RT_Solve(self, normalize=0, byorder=0):
        """
        Reflection and transmission power computation.
        Returns 2R and 2T, following Victor Liu's notation (https://web.stanford.edu/group/fan/S4/python_api.html?highlight=power#S4.Simulation.GetPowerFlux).

        Args:
            normalize (int, optional): To normalize the output when the 0-th media is not vacuum, or for oblique incidence. If normalize = 1, the output will be divided by n[0]*cos(theta). Defaults to 0.
            byorder (int, optional): To get Poynting flux by order. If 0, the total power is computed. Defaults to 0.

        Returns:
            torch.tensor: Reflection power, Transmission power
        """
        aN, b0 = SolveExterior(
            self.a0,
            self.bN,
            self.q_list,
            self.phi_list,
            self.kp_list,
            self.thickness_list,
        )

        fi, bi = GetZPoyntingFlux(
            self.a0,
            b0,
            self.omega,
            self.kp_list[0],
            self.phi_list[0],
            self.q_list[0],
            byorder=byorder,
        )

        fe, be = GetZPoyntingFlux(
            aN,
            self.bN,
            self.omega,
            self.kp_list[-1],
            self.phi_list[-1],
            self.q_list[-1],
            byorder=byorder,
        )

        if self.direction == "forward":
            R = torch.real(-bi)
            T = torch.real(fe)
        elif self.direction == "backward":
            R = torch.real(fe)
            T = torch.real(-bi)

        if normalize == 1:
            R = R * self.normalization
            T = T * self.normalization
        return R, T

    def GetAmplitudes_noTranslate(self, which_layer):
        """To get the amplitude of the eigenvectors at some layer.

        Args:
            which_layer (int): Which layer to return the Fourier amplitude from.

        Returns:
            torch.tensor: ai, bi; each containing the complex amplitudes of each forward and backward mode
        """
        if which_layer == 0:
            aN, b0 = SolveExterior(
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N - 1:
            aN, b0 = SolveExterior(
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(
                which_layer,
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )
        return ai, bi

    def GetAmplitudes(self, which_layer, z_offset):
        """To get the Fourier amplitude of the eigenvectors at some layer and at some zoffset.
        Returns the raw mode amplitudes within a particular layer. For uniform (unpatterned) layers,
        the modes are simply the diffracted orders The first value is guaranteed to be the straight transmitted or specularly reflected diffraction order. For patterned layers, there is typically no meaningful information in these amplitudes.

        Args:
            which_layer (int): Which layer to return the Fourier amplitude from.
            z_offset (float): The z-offset at which to obtain the mode amplitudes. Must be 0 < z_offset < layer thickness.

        Returns:
            torch.tensor: ai, bi; containing the complex amplitudes of each forward and backward mode
        """
        if which_layer == 0:
            aN, b0 = SolveExterior(
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N - 1:
            aN, b0 = SolveExterior(
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(
                which_layer,
                self.a0,
                self.bN,
                self.q_list,
                self.phi_list,
                self.kp_list,
                self.thickness_list,
            )

        ai, bi = TranslateAmplitudes(
            self.q_list[which_layer], self.thickness_list[which_layer], z_offset, ai, bi
        )

        return ai, bi

    def Solve_FieldFourier(self, which_layer, z_offset):
        """Returns the field amplitude in Fourier space: [Ex,Ey,Ez], [Hx,Hy,Hz]
        Args:
            which_layer (int): Which layer to return the field amplitude from.
            z_offset (torch.tensor): The z-offset at which to obtain the field amplitude. Must be 0 < z_offset < layer thickness. Can be a single number or a list of numbers.

        Returns:
            torch.tensor: Tensors containing the field amplitudes [Ex,Ey,Ez], [Hx,Hy,Hz] in Fourier space
        """
        ai0, bi0 = self.GetAmplitudes_noTranslate(which_layer)
        # ai, bi = self.GetAmplitudes(which_layer,z_offset)

        if torch.isinstance(z_offset, float) or torch.isinstance(z_offset, int):
            zl = [z_offset]
        else:
            zl = z_offset

        eh = []
        for zoff in zl:
            ai, bi = TranslateAmplitudes(
                self.q_list[which_layer],
                self.thickness_list[which_layer],
                zoff,
                ai0,
                bi0,
            )
            # hx, hy in Fourier space
            fhxy = torch_dot(self.phi_list[which_layer], ai + bi)
            fhx = fhxy[: self.nG]
            fhy = fhxy[self.nG :]

            # ex,ey in Fourier space
            tmp1 = (ai - bi) / self.omega / self.q_list[which_layer]
            tmp2 = torch_dot(self.phi_list[which_layer], tmp1)
            fexy = torch_dot(self.kp_list[which_layer], tmp2)
            fey = -fexy[: self.nG]
            fex = fexy[self.nG :]

            # hz in Fourier space
            fhz = (self.kx * fey - self.ky * fex) / self.omega

            # ez in Fourier space
            fez = (self.ky * fhx - self.kx * fhy) / self.omega
            if self.id_list[which_layer][0] == 0:
                fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
            else:
                fez = torch_dot(
                    self.Patterned_epinv_list[self.id_list[which_layer][2]], fez
                )
            eh.append([[fex, fey, fez], [fhx, fhy, fhz]])
        return eh

    def Solve_FieldOnGrid(self, which_layer, z_offset, Nxy=None):
        """To get fields in real space on grid points. If single z_offset, the output is [[Ex,Ey,Ez], [Hx,Hy,Hz]].
        If z_offset is a list, the output os [[[Ex1,Ey1,Ez1],[Hx1,Hy1,Hz1]],  [[Ex2,Ey2,Ez2],[Hx2,Hy2,Hz2]],  ...].

        Args:
            which_layer (int): Which layer to return the field amplitude from.
            z_offset (torch.tensor): The z-offset at which to obtain the field amplitude. Must be 0 < z_offset < layer thickness. Can be a single number or a list of numbers.
            Nxy (torch.tensor, optional): Nxy = [Nx,Ny], if not supplied, will use the number in patterned layer.

        Returns:
            torch.tensor: Tensor containing the field amplitudes [[Ex,Ey,Ez], [Hx,Hy,Hz]] in Fourier space for each z-offset
        """

        if torch.isinstance(Nxy, type(None)):
            Nxy = self.GridLayer_Nxy_list[self.id_list[which_layer][3]]
        Nx = Nxy[0]
        Ny = Nxy[1]

        # e,h in Fourier space
        fehl = self.Solve_FieldFourier(which_layer, z_offset)

        eh = []
        for feh in fehl:
            fe = feh[0]
            fh = feh[1]
            ex = get_ifft(Nx, Ny, fe[0], self.G)
            ey = get_ifft(Nx, Ny, fe[1], self.G)
            ez = get_ifft(Nx, Ny, fe[2], self.G)

            hx = get_ifft(Nx, Ny, fh[0], self.G)
            hy = get_ifft(Nx, Ny, fh[1], self.G)
            hz = get_ifft(Nx, Ny, fh[2], self.G)
            eh.append([[ex, ey, ez], [hx, hy, hz]])
        if torch.isinstance(z_offset, float) or torch.isinstance(z_offset, int):
            eh = eh[0]
        return eh

    def Volume_integral(self, which_layer, Mx, My, Mz, normalize=0):
        """To get volume integration with respect to some convolution matrix M defined for 3 directions, respectively.
        Returns the volume integral of a particular density over a unit cell throughout the entire thickness of a layer.
        Mxyz is the convolution matrix. This function computes 1/A\int_V Mx|Ex|^2+My|Ey|^2+Mz|Ez|^2.
        To be consistent with Poynting vector defintion here, the absorbed power will be just omega*output.

        Args:
            which_layer (int): Which layer to get the volume integral from.
            Mx (): Some convolution matrix M defined for the x-direction.
            My (): Some convolution matrix M defined for the y-direction.
            Mz (): Some convolution matrix M defined for the z-direction.
            normalize (int, optional): If 1, the return value is normalized according to self.normalization. Defaults to 0.

        Returns:
            float: Integration value.
        """
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1.0 / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        # un-translated amplitude
        ai, bi = SolveInterior(
            which_layer,
            self.a0,
            self.bN,
            self.q_list,
            self.phi_list,
            self.kp_list,
            self.thickness_list,
        )
        ab = torch.hstack((ai, bi))
        abMatrix = torch.outer(ab, torch.conj(ab))

        Mt = Matrix_zintegral(q, self.thickness_list[which_layer])
        # overall
        abM = abMatrix * Mt

        # F matrix
        Faxy = torch_dot(torch_dot(kp, phi), torch.diag(1.0 / self.omega / q))
        Faz1 = 1.0 / self.omega * torch_dot(epinv, torch.diag(self.ky))
        Faz2 = -1.0 / self.omega * torch_dot(epinv, torch.diag(self.kx))
        Faz = torch_dot(torch.hstack((Faz1, Faz2)), phi)

        tmp1 = torch.vstack((Faxy, Faz))
        tmp2 = torch.vstack((-Faxy, Faz))
        F = torch.hstack((tmp1, tmp2))

        # consider Mtotal
        Mzeros = torch.zeros_like(Mx)
        Mtotal = torch.vstack(
            (
                torch.hstack((Mx, Mzeros, Mzeros)),
                torch.hstack((Mzeros, My, Mzeros)),
                torch.hstack((Mzeros, Mzeros, Mz)),
            )
        )

        # integral = Tr[ abMatrix * F^\dagger *  Matconv *F ]
        tmp = torch_dot(torch_dot(torch.conj(torch_transpose(F)), Mtotal), F)
        val = torch.trace(torch_dot(abM, tmp))

        if normalize == 1:
            val = val * self.normalization
        return val

    def Solve_ZStressTensorIntegral(self, which_layer):
        """To compute the Maxwell stress tensor, integrated over the z-plane.
        Returns the integral of the electromagnetic stress tensor over a unit cell
        surface normal to the z-direction.
        Returns 2F_x,2F_y,2F_z, integrated over the z-plane.

        Args:
            which_layer (int): The layer in which the integration surface lies.

        Returns:
            torch.tensor: The real and imaginary parts of the x-, y-, and z-components of the stress tensor integrated over the specified surface, assuming a unit normal vector in the +z direction.
        """
        z_offset = 0.0
        eh = self.Solve_FieldFourier(which_layer, z_offset)
        e = eh[0][0]
        h = eh[0][1]
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        # compute D = epsilon E
        ## Dz = epsilon_z E_z = (ky*hx - kx*hy)/self.omega
        dz = (self.ky * hx - self.kx * hy) / self.omega

        ## Dxy = epsilon2 * Exy
        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = torch.hstack((-ey, ex))
            dxy = torch_dot(self.Patterned_ep2_list[self.id_list[which_layer][2]], exy)
            dx = dxy[self.nG :]
            dy = -dxy[: self.nG]

        Tx = torch.sum(ex * torch.conj(dz) + hx * torch.conj(hz))
        Ty = torch.sum(ey * torch.conj(dz) + hy * torch.conj(hz))
        Tz = 0.5 * torch.sum(
            ez * torch.conj(dz)
            + hz * torch.conj(hz)
            - ey * torch.conj(dy)
            - ex * torch.conj(dx)
            - torch.abs(hx) ** 2
            - torch.abs(hy) ** 2
        )

        Tx = torch.real(Tx)
        Ty = torch.real(Ty)
        Tz = torch.real(Tz)

        return Tx, Ty, Tz


def MakeKPMatrix(omega, layer_type, epinv, kx, ky):
    """Makes KP matrix.

    Args:
        omega (float): The angular frequency.
        layer_type (int): Uniform or patterned layer. Set to 0 for uniform layers and >0 for patterned.
        epinv ():
        kx ():
        ky ():

    Returns:
        torch.tensor:
    """
    nG = len(kx)

    # uniform layer, epinv has length 1
    if layer_type == 0:
        # JkkJT = np.block([[np.diag(ky*ky), np.diag(-ky*kx)],
        #                 [np.diag(-kx*ky),np.diag(kx*kx)]])

        Jk = torch.vstack((torch.diag(-ky), torch.diag(kx)))
        JkkJT = torch_dot(Jk, torch_transpose(Jk))

        kp = omega**2 * torch_eye(2 * nG) - epinv * JkkJT
    # patterned layer
    else:
        Jk = torch.vstack((torch.diag(-ky), torch.diag(kx)))
        tmp = torch_dot(Jk, epinv)
        kp = omega**2 * torch_eye(2 * nG) - torch_dot(tmp, torch_transpose(Jk))

    return kp


def SolveLayerEigensystem_uniform(omega, kx, ky, epsilon):
    """Solves the eigensystem for uniform layers.

    Args:
        omega (float): The angular frequency.
        kx ():
        ky ():
        epsilon ():

    Returns:
        torch.tensor, torch.tensor: q, phi
    """
    nG = len(kx)
    q = torch.sqrt(epsilon * omega**2 - kx**2 - ky**2)
    # branch cut choice
    q = torch.where(torch.imag(q) < 0.0, -q, q)

    q = torch.cat((q, q))
    phi = torch_eye(2 * nG)
    return q, phi


def SolveLayerEigensystem(omega, kx, ky, kp, ep2):
    """Solves the eigensystem.

    Args:
        omega (float): The angular frequency. Unused (from GRCWA).
        kx ():
        ky ():
        kp ():
        ep2 ():

    Returns:
        torch.tensor, torch.tensor: q, phi
    """
    nG = len(kx)

    k = torch.vstack((torch.diag(kx), torch.diag(ky)))
    kkT = torch_dot(k, torch_transpose(k))
    M = torch_dot(ep2, kp) - kkT

    q, phi = torch.linalg.eig(M)

    q = torch.sqrt(q)
    # branch cut choice
    q = torch.where(torch.imag(q) < 0.0, -q, q)
    return q, phi


def GetSMatrix(indi, indj, q_list, phi_list, kp_list, thickness_list):
    """Gets the S matrices, S_ij: size 4n*4n.

    Args:
        indi (int): Start layer index.
        indj (int): End layer index. Must be >= indi.
        q_list ():
        phi_list ():
        kp_list ():
        thickness_list (): List of thicknesses for each layer in the stack.

    Returns:
        torch.tensor, torch.tensor, torch.tensor, torch.tensor: S11, S12, S21, S22
    """
    # assert type(indi) == int, 'layer index i must be integer'
    # assert type(indj) == int, 'layer index j must be integer'

    nG2 = len(q_list[0])
    S11 = torch_eye(nG2, dtype=complex)
    S12 = torch.zeros_like(S11)
    S21 = torch.zeros_like(S11)
    S22 = torch_eye(nG2, dtype=complex)
    if indi == indj:
        return S11, S12, S21, S22
    elif indi > indj:
        raise Exception("indi must be < indj")

    for l in range(indi, indj):
        ## next layer
        lp1 = l + 1
        ## Q = inv(phi_l) * phi_lp1
        Q = torch_dot(torch.inverse(phi_list[l]), phi_list[lp1])
        ## P = ql*inv(kp_l*phi_l) * kp_lp1*phi_lp1*q_lp1^-1
        P1 = torch_dot(
            torch.diag(q_list[l]), torch.inverse(torch_dot(kp_list[l], phi_list[l]))
        )
        P2 = torch_dot(
            torch_dot(kp_list[lp1], phi_list[lp1]), torch.diag(1.0 / q_list[lp1])
        )
        P = torch_dot(P1, P2)
        # P1 = torch_dot(kp_list[l],phi_list[l])
        # P2 = torch_dot(torch_dot(kp_list[lp1],phi_list[lp1]),   torch.diag(1./q_list[lp1]))
        # P = np.linalg.solve(P1,P2)
        # P = np.dot(np.diag(q_list[l]),P)

        # T11=T22, T12=T21
        T11 = 0.5 * (Q + P)
        T12 = 0.5 * (Q - P)

        # phase
        d1 = torch.diag(torch.exp(1j * q_list[l] * thickness_list[l]))
        d2 = torch.diag(torch.exp(1j * q_list[lp1] * thickness_list[lp1]))

        # S11 = inv(T11-d1*S12o*T12)*d1*S11o
        P1 = T11 - torch_dot(torch_dot(d1, S12), T12)
        P1 = torch.inverse(P1)  # hold for further use
        S11 = torch_dot(torch_dot(P1, d1), S11)

        # S12 = P1*(d1*S12o*T11-T12)*d2
        P2 = torch_dot(d1, torch_dot(S12, T11)) - T12
        S12 = torch_dot(torch_dot(P1, P2), d2)

        # S21 = S22o*T12*S11+S21o
        S21 = S21 + torch_dot(S22, torch_dot(T12, S11))

        # S22 = S22o*T12*S12+S22o*T11*d2
        P2 = torch_dot(S22, torch_dot(T12, S12))
        P1 = torch_dot(S22, torch_dot(T11, d2))
        S22 = P1 + P2

    return S11, S12, S21, S22


def SolveExterior(a0, bN, q_list, phi_list, kp_list, thickness_list):
    """Given a0, bN, solve for b0, aN.

    Args:
        a0 ():
        bN ():
        q_list ():
        phi_list ():
        kp_list ():
        thickness_list (): List of thicknesses for each layer in the stack.

    Returns:
        torch.tensor, torch.tensor: aN, b0
    """

    Nlayer = len(thickness_list)  # total number of layers
    S11, S12, S21, S22 = GetSMatrix(
        0, Nlayer - 1, q_list, phi_list, kp_list, thickness_list
    )

    aN = torch_dot(S11, a0) + torch_dot(S12, bN)
    b0 = torch_dot(S21, a0) + torch_dot(S22, bN)

    return aN, b0


def SolveInterior(which_layer, a0, bN, q_list, phi_list, kp_list, thickness_list):
    """
    Given a0, bN, solve for ai, bi

    Args:
        which_layer (int): The layer to solve for. Layer numbering starts from 0.
        a0 ():
        bN ():
        q_list ():
        phi_list ():
        kp_list ():
        thickness_list (): List of thicknesses for each layer in the stack.

    Returns:
        torch.tensor, torch.tensor: ai, bi
    """
    Nlayer = len(thickness_list)  # total number of layers
    nG2 = len(q_list[0])

    S11, S12, S21, S22 = GetSMatrix(
        0, which_layer, q_list, phi_list, kp_list, thickness_list
    )
    pS11, pS12, pS21, pS22 = GetSMatrix(
        which_layer, Nlayer - 1, q_list, phi_list, kp_list, thickness_list
    )

    # tmp = inv(1-S12*pS21)
    tmp = torch.inv(torch_eye(nG2) - torch_dot(S12, pS21))
    # ai = tmp * (S11 a0 + S12 pS22 bN)
    ai = torch_dot(tmp, torch_dot(S11, a0) + torch_dot(S12, torch_dot(pS22, bN)))
    # bi = pS21 ai + pS22 bN
    bi = torch_dot(pS21, ai) + torch_dot(pS22, bN)

    return ai, bi


def TranslateAmplitudes(q, thickness, dz, ai, bi):
    """

    Args:
        q (): q for the layer of interest.
        thickness (float): The thickness of the layer of interest.
        dz (float): The z-offset at which to translate the amplitudes.
        ai ():
        bi ():

    Returns:
        torch.tensor, torch.tensor: aim, bim
    """
    aim = ai * torch.exp(1j * q * dz)
    bim = bi * torch.exp(1j * q * (thickness - dz))
    return aim, bim


def GetZPoyntingFlux(ai, bi, omega, kp, phi, q, byorder=0):
    """Returns 2S_z/A, following Victor Liu's notation.

    Args:
        ai ():
        bi ():
        omega (float): The angular frequency.
        kp ():
        phi ():
        q ():
        byorder (int, optional): If the Poynting flux is to be given for each order (byorder > 0) or summed (= 0). Defaults to 0.

    Returns:
        torch.tensor, torch.tensor: The Poynting flux in the forward and backward direction, given as a number or a list of numbers
    """
    n2 = len(ai)
    n = int(n2 / 2)
    # A = kp phi inv(omega*q)
    A = torch_dot(torch_dot(kp, phi), torch.diag(1.0 / omega / q))

    pa = torch_dot(phi, ai)
    pb = torch_dot(phi, bi)
    Aa = torch_dot(A, ai)
    Ab = torch_dot(A, bi)

    # diff = 0.5*(pb* Aa - Ab* pa)
    diff = 0.5 * (torch.conj(pb) * Aa - torch.conj(Ab) * pa)
    # forward = real(Aa* pa) + diff
    forward_xy = torch.real(torch.conj(Aa) * pa) + diff
    backward_xy = -torch.real(torch.conj(Ab) * pb) + torch.conj(diff)

    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if byorder == 0:
        forward = torch.sum(forward)
        backward = torch.sum(backward)

    return forward, backward


def Matrix_zintegral(q, thickness, shift=1e-12):
    """Generate matrix for z-integral.

    Args:
        q ():
        thickness (float): Thickness of the layer of interest.
        shift (float, optional):               . Defaults to 1e-12.

    Returns:
        torch.tensor: Matrix for z-integral
    """
    nG2 = len(q)
    qi, qj = Gmeshgrid(q)

    # # Maa = \int exp(i q_i z)^* exp(i q_j z)
    # #     = [exp(i(q_j-q_i^*)t)-1]/i(q_j-q_i^*)
    # # Mbb = \int exp(i q_i (t-z))^* exp(i q_j (t-z))
    # #     = exp(i(q_j-q_i^*)t)  [exp(i(q_i^*-q_j)t)-1]/i(q_i^*-q_j)
    # #     = Maa
    # # Mab = \int exp(i q_i z)^* exp(i q_j (t-z))
    # #     = exp(iq_j t) [1-exp(-i(q_i^*+q_j)t)]/i(q_i^*+q_j)
    # #     = [exp(iq_j t)-exp(-i q_i^* t)]/i(q_i^*+q_j)
    # # Mba = \int exp(i q_i (t-z))^* exp(i q_j z)
    # #     = exp(-iq_i^* t) [exp(i(q_j+q_i^*)t)-1]/i(q_j+q_i^*)
    # #     = Mab
    # Maa = (np.exp(1j*(qj-np.conj(qi))*thickness)-1)/1j/(qj-np.conj(qi))
    # Mab = (np.exp(1j*qj*thickness)-np.exp(-1j*np.conj(qi)*thickness))/1j/(qj+np.conj(qi))

    # M = t exp(0.5it (qj-qi^*)) sinc(0.5d (sjqj-siqi^*), note in python sinc = sin(pi x)/pi x
    qij = qj - torch.conj(qi) + torch_eye(nG2) * shift
    Maa = (torch.exp(1j * qij * thickness) - 1) / 1j / qij  # this is more robust
    # Maa = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij/np.pi)

    qij2 = qj + torch.conj(qi)
    # Mab = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij2/np.pi)
    Mab = (
        (torch.exp(1j * qj * thickness) - torch.exp(-1j * torch.conj(qi) * thickness))
        / 1j
        / qij2
    )

    tmp1 = torch.vstack((Maa, Mab))
    tmp2 = torch.vstack((Mab, Maa))
    Mt = torch.hstack((tmp1, tmp2))
    return Mt


def Gmeshgrid(x):
    """.

    Args:
        x ():

    Returns:
        torch.tensor, torch.tensor: qi, qj
    """
    N = len(x)
    qj = []
    for i in range(N):
        qj.append(x)
    qj = torch.tensor(qj)
    qi = torch_transpose(qj)
    return qi, qj
