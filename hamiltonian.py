#/usr/local/bin/python
import numpy as np
from system import System, Structure, Lattice, Atom
import system 
from tb_params import HOPPING_INTEGRALS


class Hamiltonian(object):
    E_PREFIX = 'e_'
    def __init__(self, system):
        self.system = system
        self.n_orbitals = len(self.system.all_orbitals)
        self.H_wo_g = np.zeros((self.system.structure.max_image, 
                                self.n_orbitals, self.n_orbitals), dtype=complex)
        self.calc_ham_wo_k()

    @staticmethod
    def get_orb_ind(orbit):
        return Atom.ORBITALS_ALL.index(orbit)

    def get_ham(self, kpt):
        g_mat = self.calc_g(kpt)
        self.g_mat = g_mat
        h = self.H_wo_g * g_mat
        h = np.sum(h, axis=0)

        return h

    def calc_g(self, kpt):
        """ calc g mat as func of bond matrix, dist_mat_vec, and k
            g mat is phase factor
        """
        rec_lat = self.system.structure.lattice.get_rec_lattice()
        kpt_cart = np.dot(kpt, rec_lat)
        g_mat = np.zeros((self.system.structure.max_image, 
                          self.n_orbitals, self.n_orbitals), dtype=complex)

        dist_mat_vec = self.system.structure.dist_mat_vec
        bond_mat = self.system.structure.bond_mat
        # norm_factor = np.sum(bond_mat, axis=0)

        for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(self.system.all_iter):
            for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(self.system.all_iter):
                for image_ind in range(self.system.structure.max_image):
                    if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                        continue
                    dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]

                    phase = np.exp(2*np.pi*1j * np.dot(kpt_cart, dist_vec))
                    g_mat[image_ind, ind_1, ind_2] = phase #/ norm_factor[atom_1_i, atom_2_i]

        g_mat[0, :, :] += np.eye(self.n_orbitals)
        return g_mat

    def calc_ham_wo_k(self):
        """ calc hamiltonian with out k
            all g factor is set to 1
            params look like
                params = {
                            'Si': {
                                'es': -2.15168,
                                'ep': 4.22925,
                            },
                            'SiSi': {
                                'Vsss': -1.95933,
                                'Vsps': 3.02562,
                                'Vppp': -1.51801,
                                'Vpps': 4.10364
                            }
                        }
        """
        def get_dir_cos(dist_vec):
            """ return directional cos of distance vector """
            if np.linalg.norm(dist_vec) == 0:
                return 0, 0, 0
            else:
                return dist_vec / np.linalg.norm(dist_vec)

        params = self.system.params

        # TODO spin interactions

        # off-diagonal
        dist_mat_vec = self.system.structure.dist_mat_vec
        bond_mat = self.system.structure.bond_mat

        # add scaling function
        for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(self.system.all_iter):
            for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(self.system.all_iter):
                # only lower lower-triangular part
                if ind_2 >= ind_1:
                    continue

                hop_int = HOPPING_INTEGRALS[Hamiltonian.get_orb_ind(orbit_1)][Hamiltonian.get_orb_ind(orbit_2)]

                for image_ind in range(self.system.structure.max_image):
                    # skip unbonded pairs
                    if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                        continue

                    param_element = params['{}{}'.format(element_1, element_2)]
                    param_element = self.system.get_params(atom_1_i, atom_2_i, image_ind)
                    hop_int_ele = hop_int.subs(param_element)
                    # get direction cosines
                    dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]
                    l, m, n = get_dir_cos(dist_vec)
                    param_lmn = dict({'l': l, 'm': m, 'n': n,})

                    # sympy might be slow 
                    # May be just function is enough?
                    hop_int_ = hop_int_ele.subs(param_lmn)

                    self.H_wo_g[image_ind, ind_1, ind_2] = hop_int_.subs(param_element)
        
        # real hermitian -> symmetric
        self.H_wo_g += np.transpose(self.H_wo_g, [0, 2, 1])

        # diagonal
        H_ind = 0
        for element, orbit in self.system.all_orbitals:
            e_ele = params[element]\
                          ['{}{}'.format(Hamiltonian.E_PREFIX, orbit[0])]
            self.H_wo_g[0, H_ind, H_ind] = e_ele
            H_ind += 1

def get_kpt_path(sp_kpts, kpt_den=10):
    kpts = []
    kpts.append(sp_kpts[0])
    for kpt_ind, kpt in enumerate(sp_kpts):
        if kpt_ind == len(sp_kpts) - 1:
            break
        kpt_i = kpt
        kpt_f = sp_kpts[kpt_ind + 1]
        for seg_i in range(kpt_den):
            frac = (seg_i + 1.) / float(kpt_den)
            kpt_seg = kpt_f * frac + kpt_i * (1. - frac)
            kpts.append(kpt_seg)
    return kpts

def get_kpt_len(kpts, lat_mat):
    
    rec_lat_mat = np.linalg.inv(lat_mat).T
    
    kpts_cart = []
    for kpt in kpts:
        kpts_cart.append(np.dot(rec_lat_mat, kpt))
    kpts_len = []

    for kpt_ind, kpt in enumerate(kpts_cart):
        
        # continue
        kpt_diff = kpt - kpts_cart[kpt_ind - 1]
        kpts_len.append(np.linalg.norm(kpt_diff))
    kpts_len[0] = 0
    kpts_len = np.cumsum(kpts_len)


    return kpts_len

def main():
    np.set_printoptions(precision=3)
    fcc_ge_sys = system.fcc_ge_sys()
    fcc_ge_ham = Hamiltonian(fcc_ge_sys)

    kpts_1 = get_kpt_path([np.array([0., 0., 0]),
                           np.array([0.0, 0.5, 0.5]),
                           np.array([1/4., 5/8., 5/8.])], 100)
    kpts_2 = get_kpt_path([np.array([3/8., 3/4., 3/8.]),
                           np.array([0, 0, 0])], 100)
    kpts = kpts_1 + kpts_2

    kpts_len_1 = get_kpt_len(kpts_1, fcc_ge_sys.structure.lattice.get_matrix())
    kpts_len_2 = get_kpt_len(kpts_2, fcc_ge_sys.structure.lattice.get_matrix())

    kpts_len = np.append(kpts_len_1, kpts_len_2 + kpts_len_1[-1])

    eigs_k = []

    for kpt in kpts:
        ham = fcc_ge_ham.get_ham(np.array(kpt))
        eigs = np.linalg.eigvalsh(ham)
        eigs_k.append(eigs)

    # n_band = 8
    eigs_k = np.array(eigs_k).T
    
    import matplotlib.pyplot as plt
    for i, eigs in enumerate(eigs_k[:8]):
        x = kpts_len
        plt.plot(x, eigs, '-', color='#FF00FF')
    plt.show()



if __name__ == '__main__':
    main()
