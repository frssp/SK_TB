#/usr/local/bin/python
import numpy as np
from system import System, Structure, Lattice, Atom
from tb_params import get_hop_int


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

        for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(self.system.all_iter):
            for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(self.system.all_iter):
                for image_ind in range(self.system.structure.max_image):
                    if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                        continue
                    dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]

                    phase = np.exp(2*np.pi*1j * np.dot(kpt_cart, dist_vec))
                    g_mat[image_ind, ind_1, ind_2] = phase 
        # non-translated image_ind is self.system.structure.max_image/2
        g_mat[self.system.structure.max_image/2, :, :] += np.eye(self.n_orbitals)
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

        def get_ind(atom_1_i, orbit_1_i, element_1, orbit_1):
            return self.system.all_iter.index((atom_1_i, orbit_1_i, element_1, orbit_1))

        params = self.system.params

        # TODO spin interactions

        # off-diagonal
        dist_mat_vec = self.system.structure.dist_mat_vec
        bond_mat = self.system.structure.bond_mat

        for atom_1_i, atom_1 in enumerate(self.system.structure.atoms):
            for atom_2_i, atom_2 in enumerate(self.system.structure.atoms):
                for image_ind in range(self.system.structure.max_image):
                    if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                        continue
                    param_element = self.system.get_params(atom_1_i, atom_2_i, image_ind)

                    # get direction cosines
                    dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]
                    l, m, n = get_dir_cos(dist_vec)
                    param_lmn = dict({'l': l, 'm': m, 'n': n,})
                    param_element.update(param_lmn)
                    hop_int_pair = get_hop_int(**param_element)

                    for orbit_1_i, orbit_1 in enumerate(atom_1.orbitals):
                        for orbit_2_i, orbit_2 in enumerate(atom_2.orbitals):
                            hop_int_ = hop_int_pair[Hamiltonian.get_orb_ind(orbit_1)][Hamiltonian.get_orb_ind(orbit_2)]                            
                            ind_1 = get_ind(atom_1_i, orbit_1_i, atom_1.element, orbit_1)
                            ind_2 = get_ind(atom_2_i, orbit_2_i, atom_2.element, orbit_2)
                            self.H_wo_g[image_ind, ind_1, ind_2] = hop_int_
        
        # real hermitian -> symmetric
        # self.H_wo_g += np.transpose(self.H_wo_g, [0, 2, 1])#[range(self.H_wo_g.shape[0])[::-1],:,:]

        # diagonal
        H_ind = 0
        for element, orbit in self.system.all_orbitals:
            e_ele = params[element]\
                          ['{}{}'.format(Hamiltonian.E_PREFIX, orbit[0])]
            # non-translated image_ind is self.system.structure.max_image/2
            self.H_wo_g[self.system.structure.max_image/2, H_ind, H_ind] = e_ele
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

if __name__ == '__main__':
    pass
