'''
Standard Coor Transformation
@author Khari Secario
@author Hung-Hsuan Lin

Requirements:
    numpy
    mendeleev
Changelog:
    ver 1.00    2022 Feb 15
	Modfified by Hugn-Hsuan Lin to integrated into the script for predicting FMO coupling
    ver b.01    2021 Nov 17
    ver beta    2021 Nov 02
Additional Info:
    Input coordinate using QChem format
    Highest-Lowest PA = z-,y-,x- axis
'''

'''======================================================
Rotate and Translate a Molecule into Standard Cooordinate
======================================================'''

'''==============================
0. Import all needed module & Subroutine
=============================='''
import sys
import numpy as np
from mendeleev import element
from math import sqrt
import ase
import time

''' 0.1. Subroutines
   =============================='''


def get_inertia_tensor(atoms, coor):
    mass = np.array([round(element(atom).atomic_weight) for atom in atoms])
    # print(mass)
    Ixx = np.sum((pow(coor[:, 1], 2) + pow(coor[:, 2], 2)) * mass)
    Iyy = np.sum((pow(coor[:, 0], 2) + pow(coor[:, 2], 2)) * mass)
    Izz = np.sum((pow(coor[:, 0], 2) + pow(coor[:, 1], 2)) * mass)
    Ixy = -np.sum(coor[:, 0] * coor[:, 1] * mass)
    Iyz = -np.sum(coor[:, 1] * coor[:, 2] * mass)
    Ixz = -np.sum(coor[:, 0] * coor[:, 2] * mass)
    return np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])


def check_parallelity(vec1, vec2):
    cosine = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    rad = np.arccos(np.clip(cosine, -1.0, 1.0))
    deg = np.rad2deg(rad)
    # print("cosine value is: %f with angle %f degree" % (cosine, deg))
    return cosine


def check_right_hand(eigenvector):  # HAS BUG!!!!
    fixed_vector = eigenvector
    if check_parallelity(np.cross(fixed_vector[0], fixed_vector[1]), fixed_vector[2]) < 0:
        fixed_vector[2] = -fixed_vector[2]
        # print('cosine < 0, swapping vector direction...')
    return fixed_vector


def get_rotation_matrix(vec1, vec2):
    # Note: each vector should already be normalized!
    c = np.dot(vec1, vec2)
    # print('cosine between 2 vectors %f' % c)
    v = np.cross(vec1, vec2)
    '''
    #check if rotation axis and cross are align, otherwise flip the sign
    if c < 0:
        print('negative rotation')
        v = -v
    '''
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_standard_orientation(molecule):
    '''=============================
	Usage: insert the coordinate and
	return the molecule in the standard orientation
	and the rotatioanl matrix
	============================='''
    '''==============================
	1. Read Input 
	=============================='''
    ''' 1.1. Separate Atom and Coordinates
	   =============================='''
    atoms = molecule.get_chemical_symbols()
    coor = molecule.get_positions()
    '''==============================
	2. Rearrange to Center of Mass
	=============================='''
    ''' 2.1. Rearrange to Center of Charge
	   =============================='''
    com = molecule.get_center_of_mass()

    ''' 2.2. Translating COC to Origin Point
	   =============================='''
    translation_vector = np.array(com)  # to translate it back, just add final coordinate with this vector
    coor = coor - translation_vector
    # print('new coordinate COM:\n %s \n =====' % coor)
    '''==============================
	3. Get Inertia Tensor
	=============================='''
    # it = get_inertia_tensor(atoms, coor)
    # print('inertia tensor:\n %s \n =====' % it)
    '''==============================
	4. Obtain the Principal Axes
	=============================='''
    # pa_val, pa_vec = np.linalg.eig(it)
    pa_val, pa_vec = molecule.get_moments_of_inertia(vectors=True)[0], molecule.get_moments_of_inertia(vectors=True)[
        1].T
    # print('principal axes moment of inertia:\n %s' % pa_val) #principal moments? #eigenval
    # print('principal axes:\n %s \n =====' % pa_vec) #eigenvect eigenfunctions is principal axes
    ''' 4.1. Ordering PA (from largest eigval to smallest eigval)
	   =============================='''
    pa_order = pa_val.argsort()[::-1]  # from max to min
    # pa_order = pa_val.argsort()           # from min to max
    ''' 4.2. Check any special case (linear molecules, 1 zero eigval and 2 equal eigval)
	   =============================='''
    # cross_pa = np.cross(pa_vec[pa_order[0]], pa_vec[pa_order[1]])
    check_pa = np.clip(np.dot(np.cross(pa_vec[pa_order[0]], pa_vec[pa_order[1]]), pa_vec[pa_order[2]]), -1.0, 1.0)
    if pa_val[pa_order[0]] == pa_val[pa_order[1]] and check_pa < 0:
        # print('Equal eigenvalue found; \nOrder is not match with permutation... \nSwapping order...')
        pa_vec_ordered = pa_vec[:, [pa_order[1], pa_order[0], pa_order[2]]]
        pa_val_ordered = pa_val[pa_order]
    else:
        pa_vec_ordered = pa_vec[:, pa_order]
        pa_val_ordered = pa_val[pa_order]
    # pa_vec_ordered = check_right_hand(pa_vec_ordered)
    # print('principal axes Ordered:\n %s \n =====' % pa_vec_ordered) #eigenvect eigenfunctions is principal axes
    ''' 4.3. Align the smallest principal axis (X) to the first atom x-direction
	   =============================='''
    check_x = np.clip(np.dot(coor[0], pa_vec_ordered[:, 2]), -1.0, 1.0)
    if check_x < 0:
        pa_vec_ordered = -pa_vec_ordered
    # print('principal axes Ordered:\n %s \n =====' % pa_vec_ordered) #eigenvect eigenfunctions is principal axes
    '''==============================
	5. Rotating the Axes
	=================
	============='''
    ''' 5.1. Align the Largest PA to z-axis
	   =============================='''
    cartesian = np.eye(3)
    rot_matrix_z = get_rotation_matrix(pa_vec_ordered[:, 0], cartesian[2])
    pa_z = rot_matrix_z.dot(pa_vec_ordered[:, 0])
    pa_y = rot_matrix_z.dot(pa_vec_ordered[:, 1])
    pa_x = rot_matrix_z.dot(pa_vec_ordered[:, 2])
    coor_prime = rot_matrix_z.dot(coor.T).T

    ''' 5.2. Align the second Largest PA to y-axis
	   =============================='''
    rot_matrix_y = get_rotation_matrix(pa_y, cartesian[1])
    pa_z = rot_matrix_y.dot(pa_z)
    pa_y = rot_matrix_y.dot(pa_y)
    pa_x = rot_matrix_y.dot(pa_x)

    '''==============================
	6. Rotating Coordinates
	=============================='''
    ''' 6.1. Get full rotation matrix
	   =============================='''
    # translation_vector
    full_rotation_matrix = np.matmul(rot_matrix_y, rot_matrix_z)

    '''==============================
	7. Additional Rotation to Algin the Atom Sequence
	=============================='''
    ''' 7.1 Check if the first atom sits in the first quadrant
	   =============================='''
    '''rotate the molecule by 180 degree around z-axis if the first C atom does not sit
	   in the first quadrant'''
    flip_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    flip_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if (full_rotation_matrix.dot(coor.T).T[2, 0] > 0 and full_rotation_matrix.dot(coor.T).T[2, 1] > 0):
        full_rotation_matrix = np.matmul(flip_x, full_rotation_matrix)
    #		coor_prime = full_rotation_matrix.dot(coor.T).T
    elif (full_rotation_matrix.dot(coor.T).T[2, 0] < 0 and full_rotation_matrix.dot(coor.T).T[2, 1] > 0):
        full_rotation_matrix = np.matmul(flip_z, full_rotation_matrix)
    #                coor_prime = full_rotation_matrix.dot(coor.T).T
    elif (full_rotation_matrix.dot(coor.T).T[2, 0] < 0 and full_rotation_matrix.dot(coor.T).T[2, 1] < 0):
        full_rotation_matrix = np.matmul(flip_y, full_rotation_matrix)
    #                coor_prime = full_rotation_matrix.dot(coor.T).T

    coor_prime = full_rotation_matrix.dot(coor.T).T
    return coor_prime.flatten(), full_rotation_matrix


# print('Full Rotation Matrix:\n %s \n =====' % full_rotation_matrix)
# np.savetxt(str(sys.argv[1])+'.RotationMatrix.txt',full_rotation_matrix.flatten(),fmt="%15.10f")
# coor_prime = full_rotation_matrix.dot(coor.T).T
# print('Rotated coordinate:\n %s' % coor_prime)
# np.savetxt(str(sys.argv[1])+'.SO.txt',[coor_prime.flatten()], fmt="%12.7f")


def rotate_molecule(RotationMatrix, molecule):
    '''==============================
        1. Read Input 
        =============================='''
    ''' 1.1. Separate Atom and Coordinates
           =============================='''
    atoms = molecule.get_chemical_symbols()
    coor = molecule.get_positions()

    '''==============================
        2. Rotate the Molecule
        =============================='''
    coor_prime = RotationMatrix.dot(coor.T).T
    return coor_prime


'''================================
Rotate A Molecular Orbital Function
================================'''


def rotate_orbitals_of_carbon(RotationMatrix_P, RotationMatrix_D, MO):
    ### Split MO based on atomic orbital basis
    s_orbitals = MO[0:4]
    two_p_orbitals = MO[4:7]
    three_p_orbitals = MO[7:10]
    three_d_orbitals = MO[10:15]
    ### Rotate p orbitals back to the dimer
    two_p_orbitals = RotationMatrix_P.dot(two_p_orbitals)
    three_p_orbitals = RotationMatrix_P.dot(three_p_orbitals)
    ### Rotate d orbitals back to the dimer
    three_d_orbitals = RotationMatrix_D.dot(three_d_orbitals)
    ### Save the rotated molecular orbitals
    res = np.hstack([s_orbitals, np.hstack([two_p_orbitals, np.hstack([three_p_orbitals, three_d_orbitals])])])
    return res


def get_rotation_matrix_d(RotationMatrix):
    ''' === def rotation matrix === '''
    RotationMatrix = RotationMatrix.reshape(3, 3).T
    a, b, c, d, e, f, g, h, o = RotationMatrix.flatten()
    d11 = (pow(a, 2) - pow(d, 2) - pow(b, 2) + pow(e, 2)) / 2
    d12 = (a * g) - (b * h)
    d13 = (2 * pow(g, 2) - pow(d, 2) - pow(a, 2) - 2 * pow(h, 2) + pow(e, 2) + pow(b, 2)) / sqrt(12)
    d14 = (d * g) - (e * h)
    d15 = (a * d) - (b * e)
    d1 = [d15, d14, d13, d12, d11]
    d21 = (a * c) - (d * f)
    d22 = (a * o) + (c * g)
    d23 = ((2 * g * o) - (d * f) - (a * c)) / sqrt(3)
    d24 = (d * o) + (f * g)
    d25 = (a * f) + (c * d)
    d2 = [d25, d24, d23, d22, d21]
    d31 = (2 * pow(c, 2) - pow(b, 2) - pow(a, 2) - 2 * pow(f, 2) + pow(e, 2) + pow(d, 2)) / sqrt(12)
    d32 = ((2 * c * o) - (b * h) - (a * g)) / sqrt(3)
    d33 = (4 * pow(o, 2) - 2 * (pow(h, 2) + pow(g, 2) + pow(f, 2) + pow(c, 2)) + pow(e, 2) + pow(d, 2) + pow(b,
                                                                                                             2) + pow(a,
                                                                                                                      2)) / 6
    d34 = ((2 * o * f) - (e * h) - (d * g)) / sqrt(3)
    d35 = ((2 * c * f) - (e * b) - (a * d)) / sqrt(3)
    d3 = [d35, d34, d33, d32, d31]
    d41 = (b * c) - (e * f)
    d42 = (b * o) + (c * h)
    d43 = ((2 * h * o) - (e * f) - (c * b)) / sqrt(3)
    d44 = (e * o) + (f * h)
    d45 = (b * f) + (c * e)
    d4 = [d45, d44, d43, d42, d41]
    d51 = (a * b) - (d * e)
    d52 = (a * h) + (b * g)
    d53 = ((2 * g * h) - (e * d) - (a * b)) / sqrt(3)
    d54 = (d * h) + (e * g)
    d55 = (a * e) + (b * d)
    d5 = [d55, d54, d53, d52, d51]
    d1 = np.array([d55, d45, d35, d25, d15])
    d2 = np.array([d54, d44, d34, d24, d14])
    d3 = np.array([d53, d43, d33, d23, d13])
    d4 = np.array([d52, d42, d32, d22, d12])
    d5 = np.array([d51, d41, d31, d21, d11])
    RotationMatrix_D = np.array([d1, d2, d3, d4, d5])
    return RotationMatrix_D


def rotate_molecular_orbital(RotationMatrix, MO, pyscf_obj):
    RotationMatrix_P = RotationMatrix.reshape(3, 3).T
    RotationMatrix_D = get_rotation_matrix_d(RotationMatrix)
    ### Rotate the orbitals of the first C atom
    res = rotate_orbitals_of_carbon(RotationMatrix_P, RotationMatrix_D, MO[0:15])
    ### find the number of C atom in the molecule
    atom_nr = pyscf_obj.natm
    nr_C = 0
    ### Rotate the orbitals of the rest of C atoms
    for i in range(atom_nr):
        if pyscf_obj.atom_symbol(i) == 'C':
            nr_C += 1
    NMODim = int(pyscf_obj.nao)
    for i in range(1, nr_C):
        atom_orbitals = MO[15 * i:15 * (i + 1)]
        tmp = rotate_orbitals_of_carbon(RotationMatrix_P, RotationMatrix_D, atom_orbitals)
        res = np.hstack([res, tmp])
    ### Attach the rest of MO
    res = np.hstack([res, MO[15 * nr_C:NMODim]])
    return res
