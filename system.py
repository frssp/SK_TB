import numpy as np
import itertools


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

        assert self.params.keys() == self.get_param_key(), 'wrong parameter set'
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
        for key in itertools.product(elements, repeat=2):
            key_list.append(''.join(key))
        return key_list

    def chk_scale_param_key(self):
        """ check if hoping parameters and exponent parameters are consistent
        """
        if self.scale_params is None:
            return True
        elements = self.structure.get_elements()
        key_list = []
        for key in itertools.product(elements, repeat=2):
            key_list.append(''.join(key))
        
        # compare hopping term and exponent
        l_consist = True
        for pair in key_list:
            hop_orbit = set([hop.replace('V_', '') for hop in self.params[pair]])
            exp_orbit = set([hop.replace('n_', '') for hop in self.scale_params[pair]
                             if 'n_' in hop])
            
            l_consist = l_consist and exp_orbit == hop_orbit
        return l_consist


class Structure(object):
    """ atomic structure
    """

    def __init__(self, lattice, atoms, bond_length=2.7, periodicity=None, name=None):
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
        self.bond_length = bond_length
        self.periodicity = periodicity or [True, True, True]
        self.max_image = 3 ** np.sum(self.periodicity)

        self.bond_mat = self.get_bond_mat()
        self.dist_mat_vec = self.get_dist_matrix_vec()
        self.dist_mat = self.get_dist_matrix()

    def get_bond_mat(self):
        max_image = self.max_image
        n_atom = len(self.atoms)
        bond_mat = np.zeros((max_image, n_atom, n_atom))
        dist_mat = self.get_dist_matrix()
        bond_mat = dist_mat < self.bond_length
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
            diff = np.dot(lat_vecs.T,diff)
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
                    diff = get_dist_vec(atom1.pos + np.array(image), atom2.pos, lat_vecs)
                    d_mat[image_i, i, j, :] = diff
        return d_mat

    def get_elements(self):
        """return list of elements eg) ['Si', 'O']"""
        from collections import OrderedDict
        return list(OrderedDict.fromkeys([atom.element for atom in self.atoms]))


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


def fcc_ge_sys():
    params = {
        'Ge': {
            'e_s': -2.55247,
            'e_p': 4.48593,
            'e_d': 14.01053,
            'e_S': 23.44607,
        },
        'GeGe': {
            'V_sss': -1.86600,
            'V_sps': 2.91067,
            'V_ppp': -1.49207,
            'V_pps': 4.08481,

            'V_sds': -2.23992,
            'V_pds': -1.66657,
            'V_pdp': 2.39936,
            'V_dds': -1.82945,
            'V_ddp': 3.08177,
            'V_ddd': -1.56676,
            
            'V_SSs': -4.51331,
            'V_sSs': -1.39107,
            'V_Sps': 3.06822,
            'V_Sds': -0.77711
        }
    }
    # PhysRevB.57.6493
    params = {
        'Ge': {
            'e_s': -3.2967,
            'e_p': 4.6560,
            'e_d': 13.0143,
            'e_S': 19.1725,
        },
        'GeGe': {
            'V_sss': -1.5003,
            'V_SSs': -3.6029,
            'V_sSs': -1.9206,

            'V_sps': 2.7986,
            'V_Sps': 2.8177,
            'V_sds': -2.8028,
            'V_Sds': -0.6209,
            
            'V_pps': 4.2541,
            'V_ppp': -1.6510,

            'V_pds': -2.2138,
            'V_pdp': 1.9001,

            'V_dds': -1.2172,
            'V_ddp': 2.5054,
            'V_ddd': -2.1389,
            
        }
    }
    # PhysRevB.57.6493
    scale_params = {
        'GeGe':{
            'd_0': 5.6563 * np.sqrt(3) / 4.,
            'n_sss': 3.631,
            'n_SSs': 0, 'n_sSs': 0,
            'n_sps': 3.713,
            'n_pps': 2.030,
            'n_ppp': 4.025,
            'n_sds': 1.931,
            'n_Sps': 1.830,
            'n_pds': 1.759,
            'n_pdp': 1.872,
            'n_dds': 2., 'n_ddp': 2., 'n_ddd': 2., 'n_Sds': 2.
        }
    }
    orbitals = {'Ge': ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2', 'S']}

    fcc_ge = fcc_ge_struct()
    fcc_ge_sys = System(fcc_ge, orbitals, params, scale_params)

    return fcc_ge_sys

def fcc_ge_struct():
    a = 5.6563
    lat = Lattice(np.array([1 / 2. * np.array([1., 1., 0.]),
                            1 / 2. * np.array([0., 1., 1.]),
                            1 / 2. * np.array([1., 0., 1.])]), a)
    atoms = [Atom('Ge', np.array([0., 0., 0.])), 
             Atom('Ge', np.array([1., 1., 1.])/4.)]
    fcc_ge = Structure(lat, atoms, name='fcc_ge')
    return fcc_ge


if __name__ == '__main__':
    fcc_ge_sys()
