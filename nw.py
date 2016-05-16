#/usr/local/python
import sys_input
import numpy as np
from hamiltonian import Hamiltonian
from band_plotter import draw_band

def main():
    nw_input = sys_input.Sys_input('./input_GeH_NW.yaml')
    nw_sys = nw_input.system
    k_all_path, kpts_len = nw_input.get_kpts()

    nw_ham = Hamiltonian(nw_sys)
    
    eigs_k = []

    for kpt in k_all_path:
        ham = nw_ham.get_ham(np.array(kpt))
        eigs = np.linalg.eigvalsh(ham)
        eigs_k.append(eigs)

    eigs_k = np.array(eigs_k).T
    
    draw_band(kpts_len, eigs_k)


if __name__ == '__main__':
    main()