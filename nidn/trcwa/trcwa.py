import torch

from .constants import *

from .utils.fft_funs import Epsilon_fft, get_ifft
from .utils.kbloch import Lattice_Reciprocate, Lattice_getG, Lattice_SetKs


class TRCWA:
    def __init__(self, nG, L1, L2, freq, theta, phi, verbose=1):
        """The time harmonic convention is exp(-i omega t), speed of light = 1

        Two kinds of layers are currently supported: uniform layer,
        patterned layer from grids. Interface for patterned layer by
        direct analytic expression of Fourier series is included, but
        no examples inclded so far.

        nG: truncation order, but the actual truncation order might not be nG
        L1,L2: lattice vectors, in the list format, (x,y)

        """
        self.freq = freq
        self.omega = 2 * torch.pi * freq + 0.0j
        self.L1 = L1
        self.L2 = L2
        self.phi = torch.array(phi)
        self.theta = torch.array(theta)
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
        self.thickness_list.append(torch.array(thickness))

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
        """
        Set up reciprocal lattice (Gmethod:truncation scheme, 0 for circular, 1 for rectangular)
        Pscale: scale the period
        Compute eigenvalues for uniform layers
        Initialize vectors for patterned layers
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
        """
        Front incidence
        """
        self.direction = direction
        theta = self.theta
        phi = self.phi
        a0 = torch.zeros(2 * self.nG, dtype=complex)
        bN = torch.zeros(2 * self.nG, dtype=complex)
        if direction == "forward":
            tmp1 = torch.zeros(2 * self.nG, dtype=complex)
            tmp1[order] = 1.0
            a0 = a0 + tmp1 * (
                -s_amp
                * torch.cos(theta)
                * torch.cos(phi)
                * torch.exp(torch.array(1j * s_phase))
                - p_amp * torch.sin(phi) * torch.exp(torch.array(1j * p_phase))
            )

            tmp2 = torch.zeros(2 * self.nG, dtype=complex)
            tmp2[order + self.nG] = 1.0
            a0 = a0 + tmp2 * (
                -s_amp
                * torch.cos(theta)
                * torch.sin(phi)
                * torch.exp(torch.array(1j * s_phase))
                + p_amp * torch.cos(phi) * torch.exp(torch.array(1j * p_phase))
            )
        elif direction == "backward":
            tmp1 = torch.zeros(2 * self.nG, dtype=complex)
            tmp1[order] = 1.0
            bN = bN + tmp1 * (
                -s_amp
                * torch.cos(theta)
                * torch.cos(phi)
                * torch.exp(torch.array(1j * s_phase))
                - p_amp * torch.sin(phi) * torch.exp(torch.array(1j * p_phase))
            )

            tmp2 = torch.zeros(2 * self.nG, dtype=complex)
            tmp2[order + self.nG] = 1.0
            bN = bN + tmp2 * (
                -s_amp
                * torch.cos(theta)
                * torch.sin(phi)
                * torch.exp(torch.array(1j * s_phase))
                + p_amp * torch.cos(phi) * torch.exp(torch.array(1j * p_phase))
            )

        self.a0 = a0
        self.bN = bN

    def GridLayer_geteps(self, ep_all):
        """
        Fourier transform + eigenvalue for grid layer
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
                    torch.array(ep_all[ptr : ptr + Nx * Ny]), [Nx, Ny]
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
        """
        For patterned layer component = 'xx','xy','yx','yy','zz'
        For uniform layer, currently it's assumed to be isotropic
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
        Reflection and transmission power computation
        Returns 2R and 2T, following Victor's notation
        Maybe because 2* makes S_z = 1 for H=1 in vacuum

        if normalize = 1, it will be divided by n[0]*cos(theta)
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
        """
        returns fourier amplitude
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
        """
        returns fourier amplitude
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
        """
        returns field amplitude in fourier space: [ex,ey,ez], [hx,hy,hz]
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
            fhxy = torch.dot(self.phi_list[which_layer], ai + bi)
            fhx = fhxy[: self.nG]
            fhy = fhxy[self.nG :]

            # ex,ey in Fourier space
            tmp1 = (ai - bi) / self.omega / self.q_list[which_layer]
            tmp2 = torch.dot(self.phi_list[which_layer], tmp1)
            fexy = torch.dot(self.kp_list[which_layer], tmp2)
            fey = -fexy[: self.nG]
            fex = fexy[self.nG :]

            # hz in Fourier space
            fhz = (self.kx * fey - self.ky * fex) / self.omega

            # ez in Fourier space
            fez = (self.ky * fhx - self.kx * fhy) / self.omega
            if self.id_list[which_layer][0] == 0:
                fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
            else:
                fez = torch.dot(
                    self.Patterned_epinv_list[self.id_list[which_layer][2]], fez
                )
            eh.append([[fex, fey, fez], [fhx, fhy, fhz]])
        return eh

    def Solve_FieldOnGrid(self, which_layer, z_offset, Nxy=None):
        # Nxy = [Nx,Ny], if not supplied, will use the number in patterned layer
        # if single z_offset:  output [[ex,ey,ez],[hx,hy,hz]]
        # if z_offset is list: output [[[ex1,ey1,ez1],[hx1,hy1,hz1]],  [[ex2,ey2,ez2],[hx2,hy2,hz2]] ...]

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
        """Mxyz is convolution matrix.
        This function computes 1/A\int_V Mx|Ex|^2+My|Ey|^2+Mz|Ez|^2
        To be consistent with Poynting vector defintion here, the absorbed power will be just omega*output
        """
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1.0 / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        # un-translated amplitdue
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
        Faxy = torch.dot(torch.dot(kp, phi), torch.diag(1.0 / self.omega / q))
        Faz1 = 1.0 / self.omega * torch.dot(epinv, torch.diag(self.ky))
        Faz2 = -1.0 / self.omega * torch.dot(epinv, torch.diag(self.kx))
        Faz = torch.dot(torch.hstack((Faz1, Faz2)), phi)

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
        tmp = torch.dot(torch.dot(torch.conj(torch.transpose(F)), Mtotal), F)
        val = torch.trace(torch.dot(abM, tmp))

        if normalize == 1:
            val = val * self.normalization
        return val

    def Solve_ZStressTensorIntegral(self, which_layer):
        """
        returns 2F_x,2F_y,2F_z, integrated over z-plane
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
            dxy = torch.dot(self.Patterned_ep2_list[self.id_list[which_layer][2]], exy)
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
    nG = len(kx)

    # uniform layer, epinv has length 1
    if layer_type == 0:
        # JkkJT = np.block([[np.diag(ky*ky), np.diag(-ky*kx)],
        #                 [np.diag(-kx*ky),np.diag(kx*kx)]])

        Jk = torch.vstack((torch.diag(-ky), torch.diag(kx)))
        JkkJT = torch.dot(Jk, torch.transpose(Jk))

        kp = omega ** 2 * torch.eye(2 * nG) - epinv * JkkJT
    # patterned layer
    else:
        Jk = torch.vstack((torch.diag(-ky), torch.diag(kx)))
        tmp = torch.dot(Jk, epinv)
        kp = omega ** 2 * torch.eye(2 * nG) - torch.dot(tmp, torch.transpose(Jk))

    return kp


def SolveLayerEigensystem_uniform(omega, kx, ky, epsilon):
    nG = len(kx)
    q = torch.sqrt(epsilon * omega ** 2 - kx ** 2 - ky ** 2)
    # branch cut choice
    q = torch.where(torch.imag(q) < 0.0, -q, q)

    q = torch.cat((q, q))
    phi = torch.eye(2 * nG)
    return q, phi


def SolveLayerEigensystem(omega, kx, ky, kp, ep2):
    nG = len(kx)

    k = torch.vstack((torch.diag(kx), torch.diag(ky)))
    kkT = torch.dot(k, torch.transpose(k))
    M = torch.dot(ep2, kp) - kkT

    q, phi = torch.eig(M)

    q = torch.sqrt(q)
    # branch cut choice
    q = torch.where(torch.imag(q) < 0.0, -q, q)
    return q, phi


def GetSMatrix(indi, indj, q_list, phi_list, kp_list, thickness_list):
    """S_ij: size 4n*4n"""
    # assert type(indi) == int, 'layer index i must be integar'
    # assert type(indj) == int, 'layer index j must be integar'

    nG2 = len(q_list[0])
    S11 = torch.eye(nG2, dtype=complex)
    S12 = torch.zeros_like(S11)
    S21 = torch.zeros_like(S11)
    S22 = torch.eye(nG2, dtype=complex)
    if indi == indj:
        return S11, S12, S21, S22
    elif indi > indj:
        raise Exception("indi must be < indj")

    for l in range(indi, indj):
        ## next layer
        lp1 = l + 1
        ## Q = inv(phi_l) * phi_lp1
        Q = torch.dot(torch.inverse(phi_list[l]), phi_list[lp1])
        ## P = ql*inv(kp_l*phi_l) * kp_lp1*phi_lp1*q_lp1^-1
        P1 = torch.dot(
            torch.diag(q_list[l]), torch.inverse(torch.dot(kp_list[l], phi_list[l]))
        )
        P2 = torch.dot(
            torch.dot(kp_list[lp1], phi_list[lp1]), torch.diag(1.0 / q_list[lp1])
        )
        P = torch.dot(P1, P2)
        # P1 = torch.dot(kp_list[l],phi_list[l])
        # P2 = torch.dot(torch.dot(kp_list[lp1],phi_list[lp1]),   torch.diag(1./q_list[lp1]))
        # P = np.linalg.solve(P1,P2)
        # P = np.dot(np.diag(q_list[l]),P)

        # T11=T22, T12=T21
        T11 = 0.5 * (Q + P)
        T12 = 0.5 * (Q - P)

        # phase
        d1 = torch.diag(torch.exp(1j * q_list[l] * thickness_list[l]))
        d2 = torch.diag(torch.exp(1j * q_list[lp1] * thickness_list[lp1]))

        # S11 = inv(T11-d1*S12o*T12)*d1*S11o
        P1 = T11 - torch.dot(torch.dot(d1, S12), T12)
        P1 = torch.inv(P1)  # hold for further use
        S11 = torch.dot(torch.dot(P1, d1), S11)

        # S12 = P1*(d1*S12o*T11-T12)*d2
        P2 = torch.dot(d1, torch.dot(S12, T11)) - T12
        S12 = torch.dot(torch.dot(P1, P2), d2)

        # S21 = S22o*T12*S11+S21o
        S21 = S21 + torch.dot(S22, torch.dot(T12, S11))

        # S22 = S22o*T12*S12+S22o*T11*d2
        P2 = torch.dot(S22, torch.dot(T12, S12))
        P1 = torch.dot(S22, torch.dot(T11, d2))
        S22 = P1 + P2

    return S11, S12, S21, S22


def SolveExterior(a0, bN, q_list, phi_list, kp_list, thickness_list):
    """
    Given a0, bN, solve for b0, aN
    """

    Nlayer = len(thickness_list)  # total number of layers
    S11, S12, S21, S22 = GetSMatrix(
        0, Nlayer - 1, q_list, phi_list, kp_list, thickness_list
    )

    aN = torch.dot(S11, a0) + torch.dot(S12, bN)
    b0 = torch.dot(S21, a0) + torch.dot(S22, bN)

    return aN, b0


def SolveInterior(which_layer, a0, bN, q_list, phi_list, kp_list, thickness_list):
    """
    Given a0, bN, solve for ai, bi
    Layer numbering starts from 0
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
    tmp = torch.inv(torch.eye(nG2) - torch.dot(S12, pS21))
    # ai = tmp * (S11 a0 + S12 pS22 bN)
    ai = torch.dot(tmp, torch.dot(S11, a0) + torch.dot(S12, torch.dot(pS22, bN)))
    # bi = pS21 ai + pS22 bN
    bi = torch.dot(pS21, ai) + torch.dot(pS22, bN)

    return ai, bi


def TranslateAmplitudes(q, thickness, dz, ai, bi):
    aim = ai * torch.exp(1j * q * dz)
    bim = bi * torch.exp(1j * q * (thickness - dz))
    return aim, bim


def GetZPoyntingFlux(ai, bi, omega, kp, phi, q, byorder=0):
    """
    Returns 2S_z/A, following Victor's notation
    Maybe because 2* makes S_z = 1 for H=1 in vacuum
    """
    n2 = len(ai)
    n = int(n2 / 2)
    # A = kp phi inv(omega*q)
    A = torch.dot(torch.dot(kp, phi), torch.diag(1.0 / omega / q))

    pa = torch.dot(phi, ai)
    pb = torch.dot(phi, bi)
    Aa = torch.dot(A, ai)
    Ab = torch.dot(A, bi)

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
    """Generate matrix for z-integral"""
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
    qij = qj - torch.conj(qi) + torch.eye(nG2) * shift
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
    N = len(x)
    qj = []
    for i in range(N):
        qj.append(x)
    qj = torch.array(qj)
    qi = torch.transpose(qj)
    return qi, qj
