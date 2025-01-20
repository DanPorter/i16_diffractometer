"""
Import and use functions from the diffractometer modules

By Dan Porter
I16, Diamond
2023
"""

import numpy as np
import I16_Diffractometer_equations as de
import I16_Diffractometer_rotations as dr

np.set_printoptions(precision=3, suppress=True)

phi, eta, chi, mu = 0, 25.5/2, 90, 0

r1 = dr.rotmatrix_diffractometer(phi, chi, eta, mu)
r2 = np.dot(dr.rotmatrix_x(mu), np.dot(dr.rotmatrix_z(-eta), np.dot(dr.rotmatrix_y(chi), dr.rotmatrix_z(-phi))))

print(f"rotmatrix_diffractometer({phi}, {chi}, {eta}, {mu})")
print(r1)
print('\ncombining matrices:')
print(r2)
print(f"\nDifference: {np.sum(np.abs(r1-r2)):.3f}")

a, b, c, alpha, beta, gamma = 2.85, 2.85, 10.8, 90, 90, 120.
b_matrix = dr.bmatrix(a, b, c, alpha, beta, gamma)
print(f"\n\nB Matrix:\n{b_matrix}")

u_matrix = np.eye(3)
r_matrix = dr.rotmatrix_diffractometer(phi, chi, eta, mu)
lab_transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
print(f"U Matrix:\n{u_matrix}")
print(f"R Matrix:\n{r_matrix}")
print(f"Lab transform:\n{lab_transform}")

# Unit vectors in real space
avec, bvec, cvec = np.linalg.inv(b_matrix)
print(f"\na_vec: {avec}")
print(f"b_vec: {bvec}")
print(f"c_vec: {cvec}")

mx, my, mz = 0, 0, 3
momentmag = np.sqrt(np.sum(np.square([mx, my, mz])))
momentxyz = np.dot([mx, my, mz], [avec, bvec, cvec])
moment = momentmag * momentxyz / np.sqrt(np.sum(np.square(momentxyz)))  # broadcast n*1 x n*3 = n*3
moment[np.isnan(moment)] = 0.
print(f"\n\nMagnetic vector: ({mx},{my},{mz})")
print(f"Moment in cartesian (sample) coordinates: {moment}")

# Convert to lab coordinates
moment_lab = np.dot(lab_transform, np.dot(r_matrix, np.dot(u_matrix, moment)))
print(f"Moment in lab coordinates: {moment_lab}")

# Alternative
ub_matrix = np.dot(u_matrix, b_matrix)
ub_rl_matrix = np.dot(lab_transform, np.dot(r_matrix, ub_matrix))
avec1, bvec1, cvec1 = np.linalg.inv(ub_rl_matrix)
print('\n\nAlternative version')
print(f"a_vec': {avec1}")
print(f"b_vec': {bvec1}")
print(f"c_vec': {cvec1}")

momentmag = np.sqrt(np.sum(np.square([mx, my, mz])))
momentxyz = np.dot([mx, my, mz], np.linalg.inv(ub_rl_matrix))
moment = momentmag * momentxyz / np.sqrt(np.sum(np.square(momentxyz)))  # broadcast n*1 x n*3 = n*3
moment[np.isnan(moment)] = 0.
print(f"Moment in lab coordinates (in uB): {moment}")
momentuvw = np.linalg.norm(moment) * np.dot(moment, ub_rl_matrix) / np.linalg.norm(np.dot(moment, ub_rl_matrix))
print(f"Moment in crystal (in uB): {momentuvw}")


# scattering-plane basis
energy_kev = 8
hkl = [1, 0, 6]
wl = de.photon_wavelength(8)
dspace = 1 / np.linalg.norm(np.dot(hkl, b_matrix))
tth = de.bragg_wl(wl, dspace)
det_delta, det_gamma = tth, 0
print(f"\n\nhkl: {hkl} at {energy_kev} keV, tth = {tth:.3f} deg")
ki = de.wavevector_i(wl)
kf = de.wavevector_f(wl, det_delta, det_gamma)
q = kf - ki

u3 = q / np.linalg.norm(q)
u1 = (kf + ki) / np.linalg.norm(kf + ki)
u2 = np.cross(u3, u1)
print(f"Scattering Plane:\nU1: {u1}\nU2: {u2}\nU3: {u3}")
scat_mom = np.dot(moment, np.linalg.inv([u1, u2, u3]))
scat_mom = scat_mom / np.linalg.norm(scat_mom)
print(f"Moment in scattering plane coordinates: {scat_mom}")


"""Results:
rotmatrix_diffractometer(0, 90, 30, 30)
[[ 0.     0.5    0.866]
 [ 0.5    0.75  -0.433]
 [-0.866  0.433 -0.25 ]]

combining matrices:
[[ 0.     0.5    0.866]
 [ 0.5    0.75  -0.433]
 [-0.866  0.433 -0.25 ]]

Difference: 0.000


B Matrix:
[[ 0.405  0.203 -0.   ]
 [ 0.     0.351 -0.   ]
 [ 0.     0.     0.093]]
U Matrix:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
R Matrix:
[[ 0.     0.5    0.866]
 [ 0.5    0.75  -0.433]
 [-0.866  0.433 -0.25 ]]
Lab transform:
[[0 0 1]
 [1 0 0]
 [0 1 0]]

a_vec: [ 2.468 -1.425  0.   ]
b_vec: [0.   2.85 0.  ]
c_vec: [ 0.   0.  10.8]


Magnetic vector: (1,0,3)
Moment in cartesian (sample) coordinates: [ 0.24  -0.139  3.15 ]
Moment in lab coordinates: [-1.055  2.659 -1.348]


Alternative version
a_vec': [-2.755 -0.712  0.165]
b_vec': [1.234 1.425 2.138]
c_vec': [-2.7    9.353 -4.677]
Moment in lab coordinates (in uB): [-1.055  2.659 -1.348]
"""