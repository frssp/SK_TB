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

    def get_params(self, atom_1_i, atom_2_i, image_i):
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


class Structure(object):
    """ atomic structure
    """

    def __init__(self, lattice, atoms, NN_length=2.7, periodicity=None, name=None, bond_cut=None):
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
        self.NN_length = NN_length
        self.bond_cut = bond_cut
        self.periodicity = periodicity or [True, True, True]
        self.max_image = 3 ** np.sum(self.periodicity)

        self.bond_mat = self.get_bond_mat()
        self.dist_mat_vec = self.get_dist_matrix_vec()
        self.dist_mat = self.get_dist_matrix()

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
