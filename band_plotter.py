#/usr/local/python
import numpy as np
import matplotlib.pyplot as plt

def draw_band(kpts_len, eigs_k, n_band=None):
    n_band = n_band or len(eigs_k)
    for i, eigs in enumerate(eigs_k[:n_band]):
        x = kpts_len
        plt.plot(x, eigs, '-', color='#FF00FF')
    plt.show()

def get_kpt_path(sp_kpts, kpt_den=30):
    """ return list of kpoints connecting sp_kpts
        args: 
            sp_kpts: list of k-points paths containing special kpoints
                     [n_path, n_sp_kpt, 3]
            kpt_den: number of k-points btw. sp_kpts
    """
    kpts = []
    for sp_kpt_path in sp_kpts:
        kpts_path = []
        kpts_path.append(sp_kpt_path[0])
        for kpt_ind, kpt in enumerate(sp_kpt_path):
            if kpt_ind == len(sp_kpt_path) - 1:
                break
            kpt_i = np.array(kpt)
            kpt_f = np.array(sp_kpt_path[kpt_ind + 1])
            for seg_i in range(kpt_den):
                frac = (seg_i + 1.) / float(kpt_den)
                kpt_seg = kpt_f * frac + kpt_i * (1. - frac)
                kpts_path.append(kpt_seg)
        kpts.append(kpts_path)
    return kpts

def get_kpt_len(kpts_path, lat_mat):
    
    rec_lat_mat = np.linalg.inv(lat_mat).T
    
    kpts_path_cart = []
    for kpts in kpts_path:
        kpts_cart = []
        for kpt in kpts:
            kpts_cart.append(np.dot(rec_lat_mat, kpt))
        kpts_path_cart.append(kpts_cart)

    kpts_path_len = []
    for kpts_cart in kpts_path_cart:
        kpts_len = []
        for kpt_ind, kpt in enumerate(kpts_cart):
        
            kpt_diff = kpt - kpts_cart[kpt_ind - 1]
            kpts_len.append(np.linalg.norm(kpt_diff))
        kpts_len[0] = 0
        kpts_path_len.append(kpts_len)
    kpts_path_len = [kpt for kpt_path_seg in kpts_path_len
                            for kpt in kpt_path_seg]
    
    kpts_path_len = np.cumsum(kpts_path_len)

    return kpts_path_len


if __name__ == '__main__':
    pass


