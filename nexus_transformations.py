"""
Functions to read NeXus (.nxs) files and extract transformations

Note:
The NeXus coordinate system is the same as the Diamond Lab-coordinate system:
    x-axis: horizontal, in plane of ring, away from ring
    y-axis: vertical, normal to ring, against gravity
    z-axis: along beam

This file requires Python version 3.10+
"""

from typing import List, Tuple
import numpy as np
import h5py
import matplotlib.pyplot as plt

from I16_Diffractometer_equations import photon_energy, photon_wavelength, wavevector
from I16_Diffractometer_rotations import (
    norm_vector, rotation_t_matrix, translation_t_matrix, bmatrix, transform_by_t_matrix
)


NX_CLASS = 'NX_class'
NX_DEFAULT = 'default'
NX_DEPON = 'depends_on'
NX_VECTOR = 'vector'
NX_OFFSET = 'offset'
NX_TTYPE = 'transformation_type'
NX_UNITS = 'units'
NX_WL = 'incident_wavelength'
NX_EN = 'incident_energy'
NX_ENTRY = 'NXentry'
NX_INST = 'NXinstrument'
NX_DET = 'NXdetector'
NX_SAMPLE = 'NXsample'
NX_MODULE = 'NXdetector_module'
NX_BEAM = 'NXbeam'
NX_SAMPLE_NAME = 'name'
NX_SAMPLE_UC = 'unit_cell'
NX_SAMPLE_OM = 'orientation_matrix'
NX_SAMPLE_UB = 'ub_matrix'
NX_MODULE_ORIGIN = 'data_origin'
NX_MODULE_SIZE = 'data_size'
NX_MODULE_OFFSET = 'module_offset'
NX_MODULE_FAST = 'fast_pixel_direction'
NX_MODULE_SLOW = 'slow_pixel_direction'


def check_nexus_class(hdf_group: h5py.Group, nxclass: str) -> bool:
    """
    Check if hdf_group is a certain NX_class
    :param hdf_group: hdf or nexus group object
    :param nxclass: str name in NX_class attribute
    :return: True/False
    """
    return (hdf_group and
            (group_class := hdf_group.attrs.get(NX_CLASS)) is not None and
            (group_class.decode() if isinstance(group_class, bytes) else group_class) == nxclass)


def nx_first_nxclass(group: h5py.File | h5py.Group, nxclass: str) -> str:
    """
    Find first object within group that has correct NX_class attribute
    :param group:
    :param nxclass:
    :return:
    """
    if NX_DEFAULT in group.attrs and check_nexus_class(group.get(path := group.attrs[NX_DEFAULT]), nxclass):
        return path.decode() if isinstance(path, bytes) else path
    return next(path for path in group if check_nexus_class(group.get(path), nxclass))


def get_depends_on(path: str, hdf_file: h5py.File) -> str:
    """
    Returns 'depends_on' path from this group or dataset
    The returned path will point to a dataset, based on NeXus rules
    :param path:
    :return:
    """
    obj = hdf_file[path]
    if NX_DEPON in obj.attrs:
        do_path = obj.attrs[NX_DEPON]
    elif isinstance(obj, h5py.Group) and NX_DEPON in obj:
        do_path = obj[NX_DEPON][()]
    else:
        return '.'

    do_path = do_path.decode() if isinstance(do_path, bytes) else do_path
    if do_path in hdf_file:
        return do_path
    else:
        return f"{path}/{do_path}"  # relative path


def get_dataset_value(path: str, group: h5py.Group | h5py.File, default):
    """
    Get value from dataset in group, or return default
    :param path: hdf path of dataset in group
    :param group: hdf group
    :param default: returned if path doesn't exist
    :return: group[path][()]
    """
    if path in group:
        dataset = group[path]
        if np.issubdtype(dataset, np.number):
            return np.squeeze(dataset[()])
        return dataset.asstr()[()]
    return default


def nx_depends_on_chain(path: str, hdf_file: h5py.File) -> List[str]:
    """
    Returns list of paths in a transformation chain, linked by 'depends_on'
    :param path: hdf path of initial dataset or group
    :param hdf_file:
    :return:
    """
    depends_on = get_depends_on(path, hdf_file)
    out = []
    if depends_on != '.':
        out.append(depends_on)
        out.extend(nx_depends_on_chain(depends_on, hdf_file))
    return out


def nx_direction(path: str, hdf_file: h5py.File) -> np.ndarray:
    """
    Return the direction from a dataset
    :param path: hdf path of NXtransformation path or component group with 'depends_on'
    :param hdf_file: Nxus file object
    :return:
    """
    depends_on = get_depends_on(path, hdf_file)
    if depends_on == '.':
        dataset = hdf_file[path]
    else:
        dataset = hdf_file[depends_on]

    vector = np.asarray(dataset.attrs.get('vector', (0, 0, 0)))
    return norm_vector(vector)


def nx_transformations_max_size(path: str, hdf_file: h5py.File) -> int:
    """
    Return the maximum dataset size from a chain of transformations
    :param path: hdf dataset path of NX transformation, or group containing 'depends_on'
    :param hdf_file: Nexus file object
    :return: int : largest dataset.size
    """
    dataset = hdf_file[path]
    dataset_size = dataset.size if isinstance(dataset, h5py.Dataset) else 0
    depends_on = get_depends_on(path, hdf_file)
    if depends_on != '.':
        size = nx_transformations_max_size(depends_on, hdf_file)
        return size if size > dataset_size else dataset_size
    return dataset_size


def nx_transformations(path: str, index: int, hdf_file: h5py.File, print_output=False) -> List[np.ndarray]:
    """
    Create list of 4x4 transformation matrices matching transformations along an NXtransformations chain
    :param path: str hdf path of the first point in the chain (Group or Dataset)
    :param index: int index of point in scan
    :param hdf_file: Nexus file object
    :param print_output: bool, if true the operations will be printed
    :return: list of 4x4 arrays [T1, T2, T3, ... Tn]
    """
    dataset = hdf_file[path]
    depends_on = get_depends_on(path, hdf_file)
    if print_output:
        print(f"{dataset}, depends on: {depends_on}")

    if isinstance(dataset, h5py.Group):
        return nx_transformations(depends_on, index, hdf_file, print_output)

    this_index = index if dataset.size > 1 else 0
    value = dataset[np.unravel_index(this_index, dataset.shape)]

    transformation_type = dataset.attrs.get('transformation_type', b'').decode()
    vector = np.array(dataset.attrs.get('vector', (1, 0, 0)))
    offset = dataset.attrs.get('offset', (0, 0, 0))
    units = dataset.attrs.get('units', b'').decode()

    if transformation_type == 'rotation':
        if print_output:
            print(f"Rotating about {vector} by {value} {units}  | {path}")
        matrix = rotation_t_matrix(value, units, vector, offset)
    elif transformation_type == 'translation':
        if print_output:
            print(f"Translating along {vector} by {value} {units}  | {path}")
        matrix = translation_t_matrix(value, units, vector, offset)
    else:
        if print_output:
            print(f"transformation type of '{path}' not recognized: {transformation_type}")
        matrix = np.eye(4)

    if depends_on == '.':  # end chain
        return [matrix]
    return [matrix] + nx_transformations(depends_on, index, hdf_file, print_output)


def nx_transformations_matrix(path: str, index: int, hdf_file: h5py.File) -> np.ndarray:
    """
    Combine chain of transformation operations into single matrix
    :param path: str hdf path of the first point in the chain (Group or Dataset)
    :param index: int index of point in scan
    :param hdf_file: Nexus file object
    :return: 4x4 array
    """
    matrices = nx_transformations(path, index, hdf_file)
    # Combine the transformations in reverse
    return np.linalg.multi_dot(matrices[::-1])  # multiply transformations Tn..T3.T2.T1


def nx_transform_vector(xyz, path: str, index: int, hdf_file: h5py.File) -> np.ndarray:
    """
    Transform a vector or position [x, y, z] by an NXtransformations chain
    :param xyz: 3D coordinates, n*3 [[x, y, z], ...]
    :param path: hdf path of first object in NXtransformations chain
    :param index: int index of point in scan
    :param hdf_file: Nexus file object
    :return: n*3 array([[x, y, z], ...]) transformed by operations
    """
    xyz = np.reshape(xyz, (-1, 3))
    t_matrix = nx_transformations_matrix(path, index, hdf_file)
    return (np.dot(t_matrix[:3, :3], xyz.T) + t_matrix[:3, 3:]).T


def nx_beam_energy(beam: h5py.Group):
    """
    Return beam energy in keV and wavelength in A
    :param beam: Nexus NXbeam group
    :return: incident_energy, incident_wavelength
    """
    if NX_WL in beam:
        dataset = beam[NX_WL]
        units = dataset.attrs.get('units', b'nm').decode()
        wl = dataset[()]
        if units == 'nm':
            wl = 10 * wl  # wavelength in Angstroms
        return photon_energy(wl), wl
    elif NX_EN in beam:
        dataset = beam[NX_WL]
        units = dataset.attrs.get('units', b'ev').decode()
        en = dataset[()]
        if units.lower() == 'ev':
            en = en / 1000.  # wavelength in keV
        return en, photon_wavelength(en)
    else:
        raise KeyError(f"{beam} contains no '{NX_WL}' or '{NX_EN}'")


class NXBeam:
    """
    NXbeam object
    """
    def __init__(self, path: str, hdf_file: h5py.File):
        self.file = hdf_file
        self.path = path
        self.beam = hdf_file[path]

        self.direction = nx_direction(path, hdf_file)
        self.en, self.wl = self.energy_wavelength()
        self.wv = wavevector(self.wl)
        self.incident_wavevector = self.wv * self.direction

    def energy_wavelength(self):
        """
        Return beam energy in keV and wavelength in A
        :return: incident_energy, incident_wavelength
        """
        if NX_WL in self.beam:
            dataset = self.beam[NX_WL]
            units = dataset.attrs.get('units', b'nm').decode()
            wl = dataset[()]
            if units == 'nm':
                wl = 10 * wl  # wavelength in Angstroms
            return photon_energy(wl), wl
        elif NX_EN in self.beam:
            dataset = self.beam[NX_WL]
            units = dataset.attrs.get('units', b'ev').decode()
            en = dataset[()]
            if units.lower() == 'ev':
                en = en / 1000.  # wavelength in keV
            return en, photon_wavelength(en)
        else:
            raise KeyError(f"{self.beam} contains no '{NX_WL}' or '{NX_EN}'")

    def __repr__(self):
        return f"NXBeam({self.beam})"


class NXSsample:
    """
    NXsample object
    """
    def __init__(self, path: str, hdf_file: h5py.File):
        self.file = hdf_file
        self.path = path
        self.sample = hdf_file[path]

        self.name = get_dataset_value(NX_SAMPLE_NAME, self.sample, 'none')
        self.unit_cell = get_dataset_value(NX_SAMPLE_UC, self.sample, np.array([1., 1, 1, 90, 90, 90]))
        self.orientation_matrix = get_dataset_value(NX_SAMPLE_OM, self.sample, np.eye(3))
        self.ub_matrix = get_dataset_value(NX_SAMPLE_UB, self.sample, bmatrix(*self.unit_cell))

        self.size = nx_transformations_max_size(path, hdf_file)
        self.transforms = [
            nx_transformations_matrix(path, n, hdf_file)
            for n in range(self.size)
        ]  # list of 4x4 transformation matrices

    def __repr__(self):
        return f"NXSsample({self.sample})"

    def hkl2q(self, hkl: Tuple[float, float, float] | np.ndarray):
        """
        Returns wavecector direction for given hkl
        :param hkl: Miller indices, in units of reciprocal lattice vectors
        :return: Q position in inverse Angstroms
        """
        hkl = np.reshape(hkl, (-1, 3))
        z = self.transforms[0][:3, :3]
        ub = 2 * np.pi * self.ub_matrix
        return np.dot(z, np.dot(ub, hkl.T)).T


class NXDetectorModule:
    """
    NXdetector_module object
    """
    def __init__(self, path: str, hdf_file: h5py.File):
        self.file = hdf_file
        self.path = path
        self.module = hdf_file[path]

        self.data_origin = get_dataset_value(NX_MODULE_ORIGIN, self.module, np.array([0, 0]))
        self.data_size = get_dataset_value(NX_MODULE_SIZE, self.module, np.array([1, 1]))

        self.module_offset_path = f"{self.path}/{NX_MODULE_OFFSET}"
        self.fast_pixel_direction_path = f"{self.path}/{NX_MODULE_FAST}"
        self.slow_pixel_direction_path = f"{self.path}/{NX_MODULE_SLOW}"

        self.size = nx_transformations_max_size(self.module_offset_path, hdf_file)
        self.offset_transforms = [
            nx_transformations_matrix(self.module_offset_path, n, hdf_file)
            for n in range(self.size)
        ]  # list of 4x4 transformation matrices
        self.fast_transforms = [
            nx_transformations_matrix(self.fast_pixel_direction_path, n, hdf_file)
            for n in range(self.size)
        ]  # list of 4x4 transformation matrices
        self.slow_transforms = [
            nx_transformations_matrix(self.slow_pixel_direction_path, n, hdf_file)
            for n in range(self.size)
        ]  # list of 4x4 transformation matrices

    def __repr__(self):
        return f"NXDetectorModule({self.module})"

    def shape(self):
        """
        Return scan shape of module
            (n, i, j)
        Where:
            n = frames in scan
            i = pixels along slow axis
            j = pixels along fast axis
        """
        return self.size, self.data_size[0], self.data_size[1]

    def pixel_wavevector(self, point: Tuple[int, int, int], wavelength_a) -> np.ndarray:
        """
        Return wavevector of pixel
        :param point: (n, i, j) == (frame, slow_axis_pixel, fast_axis_pixel)
        :param wavelength_a: wavelength in Angstrom
        :return: [dx, dy, dz] unit vector
        """
        return wavevector(wavelength_a) * self.pixel_direction(point)

    def pixel_direction(self, point: Tuple[int, int, int]) -> np.ndarray:
        """
        Return direction of pixel
        :param point: (n, i, j) == (frame, slow_axis_pixel, fast_axis_pixel)
        :return: [dx, dy, dz] unit vector
        """
        return norm_vector(self.pixel_position(point))

    def pixel_position(self, point: Tuple[int, int, int]) -> np.ndarray:
        """
        Return position of pixel (n, i, j)
            n = frame in scan
            i = pixel along slow axis
            j = pixel along fast axis
        """
        index, ii, jj = point

        module_origin = transform_by_t_matrix([0, 0, 0], self.offset_transforms[index])
        fast_pixel = transform_by_t_matrix([0, 0, 0], self.fast_transforms[index])
        slow_pixel = transform_by_t_matrix([0, 0, 0], self.slow_transforms[index])

        fast_direction = fast_pixel - module_origin
        slow_direction = slow_pixel - module_origin
        return np.squeeze(ii * slow_direction + jj * fast_direction + module_origin)

    def corners(self, frame: int) -> np.ndarray:
        shape = self.shape()
        corners = np.vstack([
            self.pixel_position((frame, 0, 0)),  # module origin
            self.pixel_position((frame, shape[1], 0)),  # module origin + slow pixels
            self.pixel_position((frame, shape[1], shape[2])),  # o + slow + fast
            self.pixel_position((frame, 0, shape[2])),  # o + fast
            self.pixel_position((frame, 0, 0)),  # module origin
        ])
        return corners


class NXDetector:
    """
    NXdetector object
    """
    def __init__(self, path: str, hdf_file: h5py.File):
        self.file = hdf_file
        self.path = path
        self.detector = hdf_file[path]
        self.size = nx_transformations_max_size(path, hdf_file)
        self.position = nx_transform_vector((0, 0, 0), path, self.size // 2, hdf_file)

        self.modules = [
            NXDetectorModule(f"{self.path}/{p}", hdf_file)
            for p, obj in self.detector.items()
            if obj.attrs.get(NX_CLASS) == NX_MODULE.encode()
        ]

    def __repr__(self):
        return f"NXDetector({self.detector}) with {len(self.modules)} modules"


class NXScan:
    """
    NXScan object
    """
    def __init__(self, hdf_file: h5py.File):
        self.file = hdf_file

        self.entry = hdf_file[nx_first_nxclass(hdf_file, NX_ENTRY)]
        self.instrument = self.entry[nx_first_nxclass(self.entry, NX_INST)]

        self.detectors = [
            NXDetector(f"{self.instrument.name}/{p}", hdf_file)
            for p, obj in self.instrument.items()
            if obj.attrs.get(NX_CLASS) == NX_DET.encode()
        ]

        sample_obj = self.entry[nx_first_nxclass(self.entry, NX_SAMPLE)]
        self.sample = NXSsample(sample_obj.name, hdf_file)
        beam_obj = sample_obj[nx_first_nxclass(sample_obj, NX_BEAM)]
        self.beam = NXBeam(beam_obj.name, hdf_file)

    def __repr__(self):
        return f"NXScan({self.file})"

    def shape(self):
        detector_module = self.detectors[0].modules[0]
        return detector_module.shape()

    def detector_q(self, point: Tuple[int, int, int] = (0, 0, 0)):
        wavelength = self.beam.wl
        ki = self.beam.incident_wavevector
        detector_module = self.detectors[0].modules[0]
        kf = detector_module.pixel_wavevector(point, wavelength)
        return kf - ki

    def hkl(self, point: Tuple[int, int, int] = (0, 0, 0)):
        q = self.detector_q(point)
        z = self.sample.transforms[point[0]][:3, :3]
        ub = 2 * np.pi * self.sample.ub_matrix

        inv_ub = np.linalg.inv(ub)
        inv_z = np.linalg.inv(z)

        hphi = np.dot(inv_z, q)
        return np.dot(inv_ub, hphi).T

    def hkl2q(self, hkl: Tuple[float, float, float] | np.ndarray):
        """
        Returns wavecector direction for given hkl
        :param hkl: Miller indices, in units of reciprocal lattice vectors
        :return: Q position in inverse Angstroms
        """
        return self.sample.hkl2q(hkl)

    def plot_wavevectors(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        pixel_centre = tuple([i // 2 for i in self.shape()])
        ki = self.beam.incident_wavevector
        detector_module = self.detectors[0].modules[0]
        kf = detector_module.pixel_wavevector(pixel_centre, self.beam.wl)
        q = kf - ki

        ax.plot([-ki[0], 0], [-ki[2], 0], [-ki[1], 0], '-k')
        ax.plot([0, kf[0]], [0, kf[2]], [0, kf[1]], '-k')
        ax.plot([0, q[0]], [0, q[2]], [0, q[1]], '-r')

        shape = self.shape()
        wl = self.beam.wl
        for frame in range(shape[0]):
            corners = np.vstack([
                detector_module.pixel_wavevector((frame, 0, 0), wl),  # module origin
                detector_module.pixel_wavevector((frame, shape[1], 0), wl),  # module origin + slow pixels
                detector_module.pixel_wavevector((frame, shape[1], shape[2]), wl),  # o + slow + fast
                detector_module.pixel_wavevector((frame, 0, shape[2]), wl),  # o + fast
                detector_module.pixel_wavevector((frame, 0, 0), wl),  # module origin
            ])
            ax.plot(corners[:, 0], corners[:, 2], corners[:, 1], '-k')
            corners_q = corners - ki
            ax.plot(corners_q[:, 0], corners_q[:, 2], corners_q[:, 1], '-r')

        # plot Reciprocal lattice
        astar, bstar, cstar = self.hkl2q(np.eye(3))
        ax.plot([0, astar[0]], [0, astar[2]], [0, astar[1]], '-g')
        ax.plot([0, bstar[0]], [0, bstar[2]], [0, bstar[1]], '-g')
        ax.plot([0, cstar[0]], [0, cstar[2]], [0, cstar[1]], '-g')
        ax.text(astar[0], astar[2], astar[1], s='a*')
        ax.text(bstar[0], bstar[2], bstar[1], s='b*')
        ax.text(cstar[0], cstar[2], cstar[1], s='c*')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f"HKL: {self.hkl(pixel_centre)}")
        ax.set_aspect('equal')
        fig.show()

    def plot_hkl(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        shape = self.shape()
        pixel_centre = tuple([i // 2 for i in shape])
        for frame in range(shape[0]):
            corners = np.vstack([
                self.hkl((frame, 0, 0)),  # module origin
                self.hkl((frame, shape[1], 0)),  # module origin + slow pixels
                self.hkl((frame, shape[1], shape[2])),  # o + slow + fast
                self.hkl((frame, 0, shape[2])),  # o + fast
                self.hkl((frame, 0, 0)),  # module origin
            ])
            ax.plot(corners[:, 0], corners[:, 2], corners[:, 1], '-r')
        origin = self.hkl((0, 0, 0))
        ax.plot(origin[0], origin[2], origin[1], '+k')

        ax.set_xlabel('H')
        ax.set_ylabel('L')
        ax.set_zlabel('K')
        ax.set_title(f"HKL: {self.hkl(pixel_centre)}")
        ax.set_aspect('equal')
        fig.show()


if __name__ == '__main__':

    f = r"1075689.nxs"

    with h5py.File(f) as nxs:
        print('nx_transformations:')
        print(nx_transformations('/entry/instrument/pil3_100k', 0, nxs, print_output=True))

        print('\n\nNXScan:')
        scan = NXScan(nxs)
        cen = tuple([i // 2 for i in scan.shape()])
        print('hkl:')
        print(scan.hkl(cen))

        scan.plot_wavevectors()
        scan.plot_hkl()

