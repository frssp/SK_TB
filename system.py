import numpy as np
import itertools
from vasp_io import readPOSCAR


class System(object):
    """ atomics structures and tight_binding parameters 
    """
    def __init__(self, structure, orbitals, parameters, scale_params=None):
        self.structure = structure
        self.orbitals = orbitals
        self.set_orbitals()
        self.all_orbitals = self.get_all_orbitals()
        self.all_iter = self.get_all_iter()
        self.params = parameters
        self.scale_params = scale_params

        assert set(self.get_param_key()).issubset(set(self.params.keys())), \
                'wrong parameter set\n' + \
                'given: {}\n'.format(self.params.keys()) + \
                'required: {}'.format(self.get_param_key())
        assert self.chk_scale_param_key(), \
               'The hoping parameters and the exponent parameters are not consistent!'

    def set_orbitals(self):
        for atom in self.structure.atoms:
            atom.set_orbitals(self.orbitals[atom.element])

    def get_all_orbitals(self):
        all_orbitals = []
        for atom in self.structure.atoms:
            for orbit in atom.orbitals:
                all_orbitals.append((atom.element, orbit))
        return all_orbitals

    def get_all_iter(self):
        all_orbitals = []
        for atom_i, atom in enumerate(self.structure.atoms):
            for orbit_i, orbit in enumerate(atom.orbitals):
                all_orbitals.append((atom_i, orbit_i, atom.element, orbit))
        return all_orbitals

    def get_param_key(self):
        elements = self.structure.get_elements()
        key_list = []
        key_list += elements
        for key in itertools.combinations_with_replacement(elements, r=2):
            key_list.append(''.join(key))
        return key_list

    def chk_scale_param_key(self):
        """ check if hoping parameters and exponent parameters are consistent
        """
        if self.scale_params is None:
            return True

        elements = self.structure.get_elements()
        key_list = self.get_param_key()
        for ele in elements:
            key_list.remove(ele)
        # for key in itertools.product(elements, repeat=2):
        #     key_list.append(''.join(key))
        
        # compare hopping term and exponent
        l_consist = True
        for pair in key_list:
            scale_params = self.scale_params[pair]
            if scale_params is None:
                continue
            hop_orbit = set([hop.replace('V_', '') for hop in self.params[pair]
                             if 'V_' in hop])
            exp_orbit = set([hop.replace('n_', '') for hop in scale_params
                             if 'n_' in hop])
            
            l_consist = l_consist and exp_orbit == hop_orbit
        return l_consist

    def get_hop_params(self, atom_1_i, atom_2_i, image_i):
        """ return parameters dictionary
        """
        def get_pair(key_list, ele_1, ele_2):
            # key_list = self.system.get_param_key()
            if '{}{}'.format(ele_1, ele_2) in key_list:
                return '{}{}'.format(ele_1, ele_2)
            elif '{}{}'.format(ele_2, ele_1) in key_list:
                return '{}{}'.format(ele_2, ele_1)
            else:
                return None

        atoms = self.structure.atoms
        pair = get_pair(self.get_param_key(), atoms[atom_1_i].element, atoms[atom_2_i].element)
        scale_params = self.scale_params[pair]
        if scale_params is None:
            return self.params[pair]
        else:
            d_0 = scale_params['d_0']
            d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
            factor = (d_0 / float(d))

            params_scaled = dict()
            hop_params = self.params[pair]
            for key, hop in hop_params.iteritems():
                orbit = key.replace('V_', 'n_')
                params_scaled[key] = hop * factor ** scale_params[orbit]
            return params_scaled

    def calc_volume(self, atom_i):
        """ calc volume of the tetrahedron 
        """ 
        struct = self.structure
        dist_mat_vec = struct.dist_mat_vec
        bond_mat = struct.bond_mat
        dist_vec = dist_mat_vec[:, atom_i, :]
        bond = bond_mat[:, atom_i, :]

        d_mat = dist_vec[bond]
        assert len(d_mat) == 4, 'tetrahedron required! # of bond = {}'.format(len(d_mat))
        a, b, c, d = d_mat
        vol = 1/6. * np.linalg.det([a-d, b-d, c-d])
        print(vol)

    def get_onsite_term(self, atom_i):
        """ calc onsite term
        """
        def get_onsite_s(e_s, vol_ratio, alpha):
            return (e_s + alpha * vol_ratio) * np.eye(1)

        def get_onsite_p(e_p, vol_ratio, alpha, beta_0, beta_1, delta_d, dir_cos):
            b_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):
                beta = beta_0 + beta_1 * d
                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                b_term = np.array([[l ** 2, lm, nl],
                                   [lm, m ** 2, mn],
                                   [nl, mn, n ** 2]]) - 1 / 3. * np.eye(3)
                b_term_sum += beta * b_term
            return (e_p + alpha * vol_ratio) * np.eye(3) + b_term_sum
            
        def get_onsite_d(e_d, vol_ratio, alpha, beta, gamma, delta_d, dir_cos):
            b_term_sum = 0
            g_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.
                v = (3 * n ** 2 - 1.) / 2 * irt3
                b_term = np.array([[l ** 2, -lm, -nl, mn, -irt3*mn],
                                   [-lm, m ** 2, -mn, -nl, -irt3*nl],
                                   [-nl, -mn, n ** 2, 0, 2*irt3*lm],
                                   [mn, -nl, 0, n**2, 2*irt3*u],
                                   [-irt3*mn, -irt3*nl, 2*irt3*lm, 2*irt3*u, -n**2 + 2/3.]]) - 1 / 3. * np.eye(5)
                g_term = np.array([[mn**2, nl*mn, lm*mn, mn*u, mn*v],
                                   [nl*mn, nl**2, nl*lm, nl*u, nl*v],
                                   [lm*mn, lm*nl, lm**2, lm*u, lm*v],
                                   [mn*u, nl*u, lm*u, u**2, u*v],
                                   [mn*v, nl*v, lm*v, u*v, v**2]])

                b_term_sum += beta * b_term
                g_term_sum += gamma * g_term

            return (e_d + alpha * vol_ratio) * np.eye(5) + beta * b_term + gamma * g_term

        def get_onsite_pd(beta_0, beta_1, gamma_0, gamma_1, delta_d, dir_cos):
            b_term_sum = 0
            g_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):
                beta = beta_0 + beta_1 * d
                gamma = gamma_0 + gamma_1 * d

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                lmn = l * m * n
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.
                v = (3 * n ** 2 - 1.) / 2 * irt3

                b_term = np.array([[0, n, m, l, -irt3*l],
                                   [n, 0, l, -m, -irt3*m],
                                   [m, l, 0, 0, 2*irt3*n]])
                g_term = np.array([[lmn, nl*l, lm*l, l*u, l*v],
                                   [mn*m, lmn, lm*m, m*u, m*v],
                                   [mn*n, nl*n, lmn, n*u, n*v]])

                b_term_sum += beta * b_term
                g_term_sum += gamma * g_term
            return b_term_sum + g_term_sum

        def get_onsite_sp(beta, dir_cos):
            b_term_sum = 0
            for dc in dir_cos:
                l, m, n = dc
                b_term = np.array([[l, m, n]])
                b_term_sum += beta * b_term
            return b_term_sum

        def get_onsite_sd(beta, dir_cos):
            b_term_sum = 0
            for dc in dir_cos:

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.
                v = (3 * n ** 2 - 1.) / 2 * irt3
                b_term = np.array([[mn, nl, lm, u, v]])

                b_term_sum += beta * b_term
            return b_term_sum


        atoms = self.structure.atoms
        params = self.params[atoms[atom_i].element]

        if self.scale_params is None or \
            (not atoms[atom_i].element in self.scale_params or \
             self.scale_params[atoms[atom_i].element] is None):
            print('a')
            e_s = params['e_s']
            if 'px' in  atoms[atom_i].orbitals:
                e_p = params['e_p']
            if 'dxy' in  atoms[atom_i].orbitals:
                e_d = params['e_d']
            if 'S' in  atoms[atom_i].orbitals:
                e_S = params['e_S']

            e_orbit_list =[]
            if 's' in atoms[atom_i].orbitals:
                e_orbit_list += [e_s]
            if 'px' in atoms[atom_i].orbitals:
                e_orbit_list += [e_p]
            if 'py' in atoms[atom_i].orbitals:
                e_orbit_list += [e_p]
            if 'pz' in atoms[atom_i].orbitals:
                e_orbit_list += [e_p]
            if 'dxy' in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if 'dyz' in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if 'dxz' in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if 'dx2-y2' in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if 'dz2' in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if 'S' in atoms[atom_i].orbitals:
                e_orbit_list += [e_S]
            return np.diag(e_orbit_list)
        else:
            scale_params = self.scale_params[atoms[atom_i].element]

            d_0 = scale_params['d_0']

            struct = self.structure
            dist_mat_vec = struct.dist_mat_vec
            bond_mat = struct.bond_mat
            dist_vec = dist_mat_vec[:, atom_i, :]
            bond = bond_mat[:, atom_i, :]

            d_mat = dist_vec[bond]

            atom = struct.atoms[atom_i]
            dir_cos = struct.dir_cos[:, atom_i, :, :][bond]
            delta_d = (np.linalg.norm(d_mat, axis=-1) - d_0)/d_0

            orbitals = atom.orbitals
            n_orbitals = len(orbitals)
            # onsite = np.zeros((n_orbitals, n_orbitals))
            # assume ['s',
            #             'px', 'py', 'pz',
            #             'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
            #             'S']
            # TODO generic

            vol = np.average(np.linalg.norm(d_mat, axis=-1))
            vol = (vol**3 - d_0**3)/d_0**3
            vol_ratio = vol

            s_onsite = get_onsite_s(params['e_s'], vol_ratio, scale_params['a_s'])
            S_onsite = get_onsite_s(params['e_S'], vol_ratio, scale_params['a_S'])

            p_onsite = get_onsite_p(params['e_p'], vol_ratio, scale_params['a_p'], 
                                    scale_params['b_p_0'], scale_params['b_p_1'], delta_d, dir_cos)

            d_onsite = get_onsite_d(params['e_d'], vol_ratio, scale_params['a_d'], 
                                    scale_params['b_d_0'], 0, delta_d, dir_cos)
            
            pd_onsite = get_onsite_pd(scale_params['b_pd_0'], scale_params['b_pd_1'], 
                                      0, 0, delta_d, dir_cos)

            sp_onsite = get_onsite_sp(scale_params['b_sp_0'], dir_cos)
            Sp_onsite = get_onsite_sp(scale_params['b_Sp_0'], dir_cos)
            sd_onsite = get_onsite_sd(scale_params['b_sd_0'], dir_cos)
            Sd_onsite = get_onsite_sd(scale_params['b_Sd_0'], dir_cos)
            sS_onsite = np.zeros((1,1))
            pS_onsite = np.zeros((3,1))

            onsite_term = np.bmat(np.r_[np.c_[s_onsite, sp_onsite, sd_onsite, sS_onsite],
                                        np.c_[sp_onsite.T, p_onsite, pd_onsite, pS_onsite],
                                        np.c_[sd_onsite.T, pd_onsite.T, d_onsite, Sd_onsite.T],
                                        np.c_[sS_onsite.T, Sp_onsite, Sd_onsite, S_onsite]])
            return onsite_term

    def _get_soc_mat_i(self, atom_i):
        # only for p_orbitals
        atom = self.structure.atoms[atom_i]
        param = self.params[atom.element]
        orbitals = atom.orbitals

        h_soc = np.zeros((len(orbitals)*2, len(orbitals)*2), dtype=complex)
        if 'lambda' in param.keys():
            assert ''.join(map(str, ['px', 'py', 'pz'])) in ''.join(map(str, orbitals)), \
                   'px, py, and pz should be in orbitals'
            # block_diag_list = []

            for orbit_i, orbit in enumerate(orbitals):
                if 'p' in orbit:
                    break
            lambda_p = param['lambda']
            h_soc_p = np.array([[0,   0, -1j,   0,   0,   1],
                                [0,   0,   0,  1j,   0,   0],
                                [0,   0,   0,   0,   0, -1j],
                                [0,   0,   0,   0, -1j,   0],
                                [0,  -1,   0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0]]) * lambda_p
            h_soc_p += h_soc_p.conj().T
            # orbit_i * 2 for spin 
            h_soc[orbit_i*2: orbit_i*2+6, orbit_i*2: orbit_i*2+6] = h_soc_p
            return h_soc
        else:
            return h_soc

    def get_soc_mat(self):
        import scipy.linalg
        soc_i_list = []
        for atom_i in range(len(self.structure.atoms)):
            soc_i = self._get_soc_mat_i(atom_i)
            soc_i_list.append(soc_i)

        return scipy.linalg.block_diag(*soc_i_list)
 


class Structure(object):
    """ atomic structure
    """

    def __init__(self, lattice, atoms, periodicity=None, name=None, bond_cut=None):
        """Args:
            lattice:
                Lattice object
            atoms:
                list of Atom object
        """
        assert isinstance(lattice, Lattice), 'not Lattice object'
        assert isinstance(atoms, list), 'atoms is not list'
        assert isinstance(atoms[0], Atom), 'atom is not Atom object'
        
        self.name = name or 'system'
        self.lattice = lattice
        self.atoms = atoms
        self.bond_cut = bond_cut
        self.periodicity = periodicity or [True, True, True]
        self.max_image = 3 ** np.sum(self.periodicity)

        self.bond_mat = self.get_bond_mat()
        self.dist_mat_vec = self.get_dist_matrix_vec()
        self.dist_mat = self.get_dist_matrix()
        self.dir_cos = self.get_dir_cos_all()

    def get_bond_mat(self):
        def get_cutoff(atom_1, atom_2):
            ele_1 = atom_1.element
            ele_2 = atom_2.element
            key_list = self.bond_cut.keys()
            if '{}{}'.format(ele_1, ele_2) in key_list:
                pair = '{}{}'.format(ele_1, ele_2)
            elif '{}{}'.format(ele_2, ele_1) in key_list:
                pair = '{}{}'.format(ele_2, ele_1)
            else:
                return None
            return self.bond_cut[pair]

        max_image = self.max_image
        n_atom = len(self.atoms)
        bond_mat = np.zeros((max_image, n_atom, n_atom), dtype=bool)
        dist_mat = self.get_dist_matrix()
        # bond_mat = dist_mat < self.NN_length
        atoms = self.atoms
        periodic_image = []
        for period in self.periodicity:
            if period:
                periodic_image.append(np.arange(3) - 1)
            else:
                periodic_image.append([0])

        for image_i, image in enumerate(itertools.product(*periodic_image)):
            for i, atom1 in enumerate(atoms):
                for j, atom2 in enumerate(atoms):
                    cutoff = get_cutoff(atom1, atom2)['NN']
                    if cutoff is None:
                        continue
                    bond_mat[image_i, i, j] = dist_mat[image_i, i, j] < cutoff
        bond_mat_2 = dist_mat > 0

        return bond_mat * bond_mat_2
    
    def get_dist_matrix(self):
        dist_mat_vec = self.get_dist_matrix_vec()
        dist_mat = np.linalg.norm(dist_mat_vec, axis=-1)
        return dist_mat

    def get_dist_matrix_vec(self):
        def get_dist_vec(pos1, pos2, lat_vecs, l_min=False):
            """ # p1, p2 direct 
                # return angstrom
                # latConst is included in lat_vecs
            """
            diff = np.array(pos1) - np.array(pos2)
            if np.linalg.norm(diff) ==  0:
                return 0
            if l_min:
                diff = diff - np.round(diff)
            diff = np.dot(lat_vecs.T, diff)
            return diff

        n_atom = len(self.atoms)
        max_image = self.max_image

        lat_vecs = self.lattice.get_matrix()
        atoms = self.atoms
        d_mat = np.zeros((max_image, n_atom, n_atom, 3))

        periodic_image = []
        for period in self.periodicity:
            if period:
                periodic_image.append(np.arange(3) - 1)
            else:
                periodic_image.append([0])

        for image_i, image in enumerate(itertools.product(*periodic_image)):
            for i, atom1 in enumerate(atoms):
                for j, atom2 in enumerate(atoms):
                    diff = get_dist_vec(atom1.pos + image, atom2.pos, lat_vecs) #+ np.dot(lat_vecs.T, image)
                    d_mat[image_i, i, j, :] = diff
        return d_mat

    def get_elements(self):
        """return list of elements eg) ['Si', 'O']"""
        from collections import OrderedDict
        return list(OrderedDict.fromkeys([atom.element for atom in self.atoms]))

    @staticmethod
    def read_poscar(file_name='./POSCAR', kwargs={}):
        lat_const, lattice_mat, atom_set_direct, dynamics = readPOSCAR(fileName=file_name)

        atoms = []
        for a in atom_set_direct:
            atoms.append(Atom(a[0], a[1]))

        bravais_lat = np.array(lattice_mat)
        lattice = Lattice(bravais_lat, lat_const)

        structure = Structure(lattice, atoms, **kwargs)
        return structure

    def get_dir_cos(self, image_i, atoms_i, atom_j):
        """ return directional cos of distance vector """
        dist_vec = self.dist_mat_vec[image_i, atoms_i, atom_j, :]
        if np.linalg.norm(dist_vec) == 0:
            return 0, 0, 0
        else:
            return dist_vec / np.linalg.norm(dist_vec)

    def get_dir_cos_all(self):
        dist_vec = self.dist_mat_vec
        dist_norm = np.linalg.norm(dist_vec, axis=-1)
        indx_zero = np.where(dist_norm == 0)
        dist_norm[indx_zero]=1E-10
        dir_cos = dist_vec / dist_norm[:,:,:, np.newaxis]
        return dir_cos

class Lattice:
    """represent lattice of structure
    """
    def __init__(self, *args):
        """
        Args:
            a, b, c, alpha, beta, gamma
        """
        matrix, lat_const = args
        self.matrix = np.array(matrix) * lat_const
        self.a, self.b, self.c, self.alpha, self.beta , self.gamma = \
        self._to_list(matrix, lat_const)

    def _to_list(self, matrix, lat_const):
        """ see http://en.wikipedia.org/wiki/Fractional_coordinates
        """
        from numpy.linalg import norm

        a = matrix[0] * lat_const
        b = matrix[1] * lat_const
        c = matrix[2] * lat_const

        alpha = np.arctan2(norm(np.cross(b, c)), np.dot(b, c))
        beta  = np.arctan2(norm(np.cross(c, a)), np.dot(c, a))
        gamma = np.arctan2(norm(np.cross(a, b)), np.dot(a, b))

        return norm(a), norm(b), norm(c), alpha, beta, gamma

    def get_matrix(self):
        matrix = self._to_matrix()
        return matrix

    def _to_matrix(self):
        # see http://en.wikipedia.org/wiki/Fractional_coordinates
        # For the special case of a monoclinic cell (a common case) where alpha = gamma = 90 degree and beta > 90 degree, this gives: <- special care needed
        # so far, alpha, beta, gamma < 90 degree
        a, b, c, alpha, beta, gamma = self.a, self.b, self.c, self.alpha, self.beta, self.gamma

        v = a * b * c * np.sqrt(1. - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) )

        T = np.zeros((3, 3))
        T = np.array([ \
                  [a, b * np.cos(gamma), c * np.cos(beta)                                                  ] ,\
                  [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)] ,\
                  [0, 0                , v / (a * b * np.sin(gamma))                                       ] 
                      ])
        matrix = np.zeros((3, 3))
        matrix[:,0] = np.dot(T, np.array((1, 0, 0)))
        matrix[:,1] = np.dot(T, np.array((0, 1, 0)))
        matrix[:,2] = np.dot(T, np.array((0, 0, 1)))
        # return matrix.T
        return self.matrix

    def get_rec_lattice(self):
        """
        b_i = (a_j x a_k)/ a_i . (a_j x a_k)
        """
        lat_mat = self.matrix
        rec_lat_mat = np.linalg.inv(lat_mat).T
        return rec_lat_mat

    def __repr__(self):
        _repr = [self.a, self.b, self.c, self.alpha, self.beta , self.gamma]
        _repr = [str(i) for i in _repr]
        return ' '.join(_repr)


class Atom:
    ORBITALS_ALL = ['s',
                    'px', 'py', 'pz',
                    'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
                    'S']
    def __init__(self, element, pos):
        """ Object to represent atom
            Args:
                element:
                    atomic symbol eg) 'Si'
                pos:
                    atom position (fractional coordinate) eg) [0.5, 0.5, 0] 
                orbitals:
                    subset of ['s',
                               'px', 'py', 'pz',
                               'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
                               'S']
        """

        self.element = element
        self.pos = np.array(pos)
        self.orbitals = None

    def to_list(self):
        out_list = [self.element, self.pos, self.dyn]

        return out_list

    def set_orbitals(self, orbitals=None):
        assert set(orbitals).issubset(set(Atom.ORBITALS_ALL)), 'wrong orbitals'
        self.orbitals = orbitals
    
    def __repr__(self):
        return '{} {}'.format(self.element, self.pos)


if __name__ == '__main__':
    pass
