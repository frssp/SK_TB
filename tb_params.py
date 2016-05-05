from sympy import symbols
from numpy import sqrt
(l, m, n) = symbols('l m n')
(V_sss, V_sps, V_pps, V_ppp) = symbols('V_sss V_sps V_pps V_ppp')
(V_sds, V_pds, V_pdp, V_dds, V_ddp, V_ddd) = symbols('V_sds V_pds V_pdp V_dds V_ddp V_ddd')
(V_SSs, V_sSs, V_Sps, V_Sds) = symbols('V_SSs, V_sSs, V_Sps, V_Sds')

#[s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]
HOPPING_INTEGRALS = [[None for _ in xrange(10)] for __ in xrange(10)]

HOPPING_INTEGRALS[0][0] = V_sss
HOPPING_INTEGRALS[0][1] = l * V_sps
HOPPING_INTEGRALS[0][2] = m * V_sps
HOPPING_INTEGRALS[0][3] = n * V_sps
HOPPING_INTEGRALS[1][0] = -HOPPING_INTEGRALS[0][1]
HOPPING_INTEGRALS[2][0] = -HOPPING_INTEGRALS[0][2]
HOPPING_INTEGRALS[3][0] = -HOPPING_INTEGRALS[0][3]
HOPPING_INTEGRALS[0][4] = sqrt(3) * l * m * V_sds
HOPPING_INTEGRALS[0][5] = sqrt(3) * m * n * V_sds
HOPPING_INTEGRALS[0][6] = sqrt(3) * l * n * V_sds
HOPPING_INTEGRALS[4][0] = HOPPING_INTEGRALS[0][4]
HOPPING_INTEGRALS[5][0] = HOPPING_INTEGRALS[0][5]
HOPPING_INTEGRALS[6][0] = HOPPING_INTEGRALS[0][6]
HOPPING_INTEGRALS[0][7] = sqrt(3) / 2. * (l ** 2 - m ** 2) * V_sds
HOPPING_INTEGRALS[7][0] = HOPPING_INTEGRALS[0][7]
HOPPING_INTEGRALS[0][8] = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_sds
HOPPING_INTEGRALS[8][0] = HOPPING_INTEGRALS[0][8]

HOPPING_INTEGRALS[1][1] = l ** 2 * V_pps + (1. - l ** 2) * V_ppp
HOPPING_INTEGRALS[1][2] = l * m * (V_pps - V_ppp)
HOPPING_INTEGRALS[2][1] = HOPPING_INTEGRALS[1][2]
HOPPING_INTEGRALS[1][3] = l * n * (V_pps - V_ppp)
HOPPING_INTEGRALS[3][1] = HOPPING_INTEGRALS[1][3]

HOPPING_INTEGRALS[1][4] = sqrt(3) * l ** 2 * m * V_pds + m * (1. - 2 * l ** 2) * V_pdp
HOPPING_INTEGRALS[1][5] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
HOPPING_INTEGRALS[1][6] = sqrt(3) * l ** 2 * n * V_pds + n * (1. - 2 * l ** 2) * V_pdp
HOPPING_INTEGRALS[4][1] = -HOPPING_INTEGRALS[1][4]
HOPPING_INTEGRALS[5][1] = -HOPPING_INTEGRALS[1][5]
HOPPING_INTEGRALS[6][1] = -HOPPING_INTEGRALS[1][6]

HOPPING_INTEGRALS[1][7] = 0.5 * sqrt(3) * l * (l ** 2 - m ** 2) * V_pds + l * (1. - l ** 2 + m ** 2) * V_pdp
HOPPING_INTEGRALS[1][8] = l * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_pds - sqrt(3) * l * n ** 2 * V_pdp
HOPPING_INTEGRALS[7][1] = -HOPPING_INTEGRALS[1][7]
HOPPING_INTEGRALS[8][1] = -HOPPING_INTEGRALS[1][8]

HOPPING_INTEGRALS[2][2] = m ** 2 * V_pps + (1. - m ** 2) * V_ppp
HOPPING_INTEGRALS[2][3] = m * n * (V_pps - V_ppp)
HOPPING_INTEGRALS[3][2] = HOPPING_INTEGRALS[2][3]

HOPPING_INTEGRALS[2][4] = sqrt(3) * m ** 2 * l * V_pds + l * (1. - 2 * m ** 2) * V_pdp
HOPPING_INTEGRALS[2][5] = sqrt(3) * m ** 2 * n * V_pds + n * (1. - 2 * m ** 2) * V_pdp
HOPPING_INTEGRALS[2][6] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
HOPPING_INTEGRALS[4][2] = -HOPPING_INTEGRALS[2][4]
HOPPING_INTEGRALS[5][2] = -HOPPING_INTEGRALS[2][5]
HOPPING_INTEGRALS[6][2] = -HOPPING_INTEGRALS[2][6]

HOPPING_INTEGRALS[2][7] = 0.5 * sqrt(3) * m * (l ** 2 - m ** 2) * V_pds - m * (1. + l ** 2 - m ** 2) * V_pdp
HOPPING_INTEGRALS[2][8] = m * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_pds - sqrt(3) * m * n ** 2 * V_pdp
HOPPING_INTEGRALS[7][2] = -HOPPING_INTEGRALS[2][7]
HOPPING_INTEGRALS[8][2] = -HOPPING_INTEGRALS[2][8]

HOPPING_INTEGRALS[3][3] = n ** 2 * V_pps + (1. - n ** 2) * V_ppp

HOPPING_INTEGRALS[3][4] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
HOPPING_INTEGRALS[3][5] = sqrt(3) * n ** 2 * m * V_pds + m * (1. - 2 * n ** 2) * V_pdp
HOPPING_INTEGRALS[3][6] = sqrt(3) * n ** 2 * l * V_pds + l * (1. - 2 * n ** 2) * V_pdp
HOPPING_INTEGRALS[4][3] = -HOPPING_INTEGRALS[3][4]
HOPPING_INTEGRALS[5][3] = -HOPPING_INTEGRALS[3][5]
HOPPING_INTEGRALS[6][3] = -HOPPING_INTEGRALS[3][6]

HOPPING_INTEGRALS[3][7] = 0.5 * sqrt(3) * n * (l ** 2 - m ** 2) * V_pds - n * (l ** 2 - m ** 2) * V_pdp
HOPPING_INTEGRALS[3][8] = n * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_pds + sqrt(3) * n * (l ** 2 + m ** 2) * V_pdp
HOPPING_INTEGRALS[7][3] = -HOPPING_INTEGRALS[3][7]
HOPPING_INTEGRALS[8][3] = -HOPPING_INTEGRALS[3][8]

HOPPING_INTEGRALS[4][4] = l ** 2 * m ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + (l ** 2 + m ** 2) * V_ddp + n ** 2 * V_ddd
HOPPING_INTEGRALS[5][5] = m ** 2 * n ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + (m ** 2 + n ** 2) * V_ddp + l ** 2 * V_ddd
HOPPING_INTEGRALS[6][6] = n ** 2 * l ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + (n ** 2 + l ** 2) * V_ddp + m ** 2 * V_ddd

HOPPING_INTEGRALS[4][5] = l * m ** 2 * n * (3 * V_dds - 4 * V_ddp + V_ddd) + l * n * (V_ddp - V_ddd)
HOPPING_INTEGRALS[4][6] = n * l ** 2 * m * (3 * V_dds - 4 * V_ddp + V_ddd) + n * m * (V_ddp - V_ddd)
HOPPING_INTEGRALS[5][6] = m * n ** 2 * l * (3 * V_dds - 4 * V_ddp + V_ddd) + m * l * (V_ddp - V_ddd)
HOPPING_INTEGRALS[5][4] = HOPPING_INTEGRALS[4][5]
HOPPING_INTEGRALS[6][4] = HOPPING_INTEGRALS[4][6]
HOPPING_INTEGRALS[6][5] = HOPPING_INTEGRALS[5][6]

HOPPING_INTEGRALS[4][7] = 0.5 * l * m * (l ** 2 - m ** 2) * (3 * V_dds - 4 * V_ddp + V_ddd)
HOPPING_INTEGRALS[5][7] = 0.5 * m * n *((l ** 2 - m ** 2) * (3 * V_dds - 4 * V_ddp + V_ddd) - 2 * (V_ddp - V_ddd))
HOPPING_INTEGRALS[6][7] = 0.5 * n * l *((l ** 2 - m ** 2) * (3 * V_dds - 4 * V_ddp + V_ddd) + 2 * (V_ddp - V_ddd))

HOPPING_INTEGRALS[7][4] = HOPPING_INTEGRALS[4][7]
HOPPING_INTEGRALS[7][5] = HOPPING_INTEGRALS[5][7]
HOPPING_INTEGRALS[7][6] = HOPPING_INTEGRALS[6][7]

HOPPING_INTEGRALS[4][8] = sqrt(3) * (l * m * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_dds - 2 * l * m * n ** 2 * V_ddp + 0.5 * l * m * (1. + n ** 2) * V_ddd)
HOPPING_INTEGRALS[5][8] = sqrt(3) * (m * n * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_dds + m * n * (l ** 2 + m ** 2 - n ** 2) * V_ddp - 0.5 * m * n * (l ** 2 + m ** 2) * V_ddd)
HOPPING_INTEGRALS[6][8] = sqrt(3) * (n * l * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_dds + n * l * (l ** 2 + m ** 2 - n ** 2) * V_ddp - 0.5 * n * l * (l ** 2 + m ** 2) * V_ddd)
HOPPING_INTEGRALS[8][4] = HOPPING_INTEGRALS[4][8]
HOPPING_INTEGRALS[8][5] = HOPPING_INTEGRALS[5][8]
HOPPING_INTEGRALS[8][6] = HOPPING_INTEGRALS[6][8]

HOPPING_INTEGRALS[7][7] = 0.25 * (l ** 2 - m ** 2) ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + (l ** 2 + m ** 2) * V_ddp + n ** 2 * V_ddd
HOPPING_INTEGRALS[8][8] = 0.75 * (l ** 2 + m ** 2) ** 2 * V_ddd + 3 * (l ** 2 + m ** 2) * n ** 2 * V_ddp + 0.25 * (l ** 2 + m ** 2 - 2*n ** 2) ** 2 * V_dds
HOPPING_INTEGRALS[7][8] = 0.25 * (l ** 2 - m ** 2) * (n ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + V_ddd - V_dds)
HOPPING_INTEGRALS[8][7] = 0.25 * (l ** 2 - m ** 2) * (n ** 2 * (3 * V_dds - 4 * V_ddp + V_ddd) + V_ddd - V_dds)

HOPPING_INTEGRALS[9][9] = V_SSs
HOPPING_INTEGRALS[0][9] = V_sSs
HOPPING_INTEGRALS[9][0] = V_sSs
HOPPING_INTEGRALS[9][1] = l * V_Sps
HOPPING_INTEGRALS[9][2] = m * V_Sps
HOPPING_INTEGRALS[9][3] = n * V_Sps
HOPPING_INTEGRALS[1][9] = -HOPPING_INTEGRALS[9][1]
HOPPING_INTEGRALS[2][9] = -HOPPING_INTEGRALS[9][2]
HOPPING_INTEGRALS[3][9] = -HOPPING_INTEGRALS[9][3]
HOPPING_INTEGRALS[9][4] = sqrt(3) * l * m * V_Sds
HOPPING_INTEGRALS[9][5] = sqrt(3) * m * n * V_Sds
HOPPING_INTEGRALS[9][6] = sqrt(3) * l * n * V_Sds
HOPPING_INTEGRALS[4][9] = HOPPING_INTEGRALS[9][4]
HOPPING_INTEGRALS[5][9] = HOPPING_INTEGRALS[9][5]
HOPPING_INTEGRALS[6][9] = HOPPING_INTEGRALS[9][6]
HOPPING_INTEGRALS[9][7] = sqrt(3) / 2. * (l * l - m * m) * V_Sds
HOPPING_INTEGRALS[7][9] = HOPPING_INTEGRALS[9][7]
HOPPING_INTEGRALS[9][8] = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * V_Sds
HOPPING_INTEGRALS[8][9] = HOPPING_INTEGRALS[9][8]


