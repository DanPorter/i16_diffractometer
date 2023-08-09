"""
Python module: Diffractometer Equations

Collection of useful functions to calcualte diffraction angles

Based on functions from
Dans_Diffraction.functions_crystallography.py
https://github.com/DanPorter/Dans_Diffraction

By Dan Porter
Beamline I16, Diamond Light Source Ltd
August 2023
"""

import numpy as np


# Constants
pi = np.pi  # mmmm tasty Pi
e = 1.6021733E-19  # C  electron charge
h = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
u0 = 4 * pi * 1e-7  # H m-1 Magnetic permeability of free space
me = 9.109e-31  # kg Electron rest mass
mn = 1.6749e-27 # kg Neutron rest mass
Na = 6.022e23  # Avagadro's No
A = 1e-10  # m Angstrom
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)
Cu = 8.048  # Cu-Ka emission energy, keV
Mo = 17.4808  # Mo-Ka emission energy, keV
# Mo = 17.4447 # Mo emission energy, keV


" --------------------------------------------------------------------- "
" ----------------------- Wavelength / Energy ------------------------- "
" --------------------------------------------------------------------- "


def photon_wavelength(energy_kev):
    """
    Converts energy in keV to wavelength in A
     wavelength_a = photon_wavelength(energy_kev)
     lambda [A] = h*c/E = 12.3984 / E [keV]
    """

    # Electron Volts:
    E = 1000 * energy_kev * e

    # SI: E = hc/lambda
    lam = h * c / E
    wavelength = lam / A
    return wavelength


def photon_energy(wavelength_a):
    """
    Converts wavelength in A to energy in keV
     energy_kev = photon_energy(wavelength_a)
     Energy [keV] = h*c/L = 12.3984 / lambda [A]
    """

    # SI: E = hc/lambda
    lam = wavelength_a * A
    E = h * c / lam

    # Electron Volts:
    energy = E / e
    return energy / 1000.0


def neutron_wavelength(energy_mev):
    """
    Calcualte the neutron wavelength in Angstroms using DeBroglie's formula
      lambda [A] ~ sqrt( 81.82 / E [meV] )
    :param energy_mev: neutron energy in meV
    :return: wavelength in Anstroms
    """
    return h / np.sqrt(2 * mn * energy_mev * e / 1000) / A


def neutron_energy(wavelength_a):
    """
    Calcualte the neutron energy in milli-electronvolts using DeBroglie's formula
      E [meV] ~ 81.82 / lambda^2 [A]
    :param wavelength_a: neutron wavelength in Angstroms
    :return: energy in meV
    """
    return h**2 / (2 * mn * (wavelength_a*A)**2 * e / 1000)


def electron_wavelength(energy_ev):
    """
    Calcualte the electron wavelength in Angstroms using DeBroglie's formula
      lambda [nm] ~ sqrt( 1.5 / E [eV] )
    :param energy_ev: electron energy in eV
    :return: wavelength in Anstroms
    """
    return h / np.sqrt(2 * me * energy_ev * e) / A


def electron_energy(wavelength_a):
    """
    Calcualte the electron energy in electronvolts using DeBroglie's formula
      E [eV] ~ 1.5 / lambda^2 [nm]
    :param wavelength_a: electron wavelength in Angstroms
    :return: energy in eV
    """
    return h**2 / (2 * me * (wavelength_a * A)**2 * e)


def debroglie_wavelength(energy_kev, mass_kg):
    """
    Calcualte the wavelength in Angstroms using DeBroglie's formula
      lambda [A] = h  / sqrt( 2 * mass [kg] * E [keV] * 1e3 * e )
    :param energy_kev: energy in keV
    :param mass_kg: mass in kg
    :return: wavelength in Anstroms
    """
    return h / (np.sqrt(2 * mass_kg * energy_kev * 1000 * e) * A)


def debroglie_energy(wavelength_a, mass_kg):
    """
    Calcualte the energy in electronvolts using DeBroglie's formula
      E [keV] = h^2 / (2 * e * mass [kg] * A^2 * lambda^2 [A] * 1e3)
    :param wavelength_a: wavelength in Angstroms
    :param mass_kg: mass in kg
    :return: energy in keV
    """
    return h ** 2 / (2 * mass_kg * (wavelength_a * A) ** 2 * e * 1e3)


" --------------------------------------------------------------------- "
" --------------------------- Wavevectors ----------------------------- "
" --------------------------------------------------------------------- "


def calqmag(twotheta, wavelength_a):
    """
    Calculate |Q| at a particular 2-theta (deg) for energy in keV
     magQ = calqmag(twotheta, wavelength_a=1.5)
    """
    theta = twotheta * pi / 360  # theta in radians
    magq = np.sin(theta) * 4 * pi / wavelength_a
    return magq


def cal2theta(qmag, wavelength_a):
    """
    Calculate 2theta of |Q|
     twotheta = cal2theta(q_mag, wavelength_a=1.5)
    """
    tth = 2 * np.arcsin(qmag * wavelength_a / (4 * pi))
    tth = tth * 180 / pi
    return tth


def caldspace(twotheta, wavelength_a):
    """
    Calculate d-spacing from two-theta
     dspace = caldspace(tth, wavelength_a=1.5)
    """
    qmag = calqmag(twotheta, wavelength_a)
    dspace = q2dspace(qmag)
    return dspace


def q2dspace(qmag):
    """
    Calculate d-spacing from |Q|
         dspace = q2dspace(Qmag)
    """
    return 2 * pi / qmag


def dspace2q(dspace):
    """
    Calculate d-spacing from |Q|
         Qmag = q2dspace(dspace)
    """
    return 2 * pi / dspace


def wavevector(wavelength_a):
    """Return wavevector = 2pi/lambda"""
    return 2 * pi / wavelength_a


def bragg_en(energy_kev, d_space):
    """Returns the Bragg angle for a given d-space at given photon energy in keV"""
    return np.rad2deg(np.arcsin(6.19922 / (energy_kev * d_space)))


def bragg_wl(wavelength_a, d_space):
    """Returns the Bragg angle for a given d-space at given wavelength in Angstroms"""
    return np.rad2deg(np.arcsin(wavelength_a / (2 * d_space)))


" --------------------------------------------------------------------- "
" ----------------------------- Misc ---------------------------------- "
" --------------------------------------------------------------------- "


def scherrer_size(fwhm, twotheta, wavelength_a, shape_factor=0.9):
    """
    Use the Scherrer equation to calculate the size of a crystallite from a peak width
      L = K * lambda / fwhm * cos(theta)
    See: https://en.wikipedia.org/wiki/Scherrer_equation
    :param fwhm: full-width-half-maximum of a peak, in degrees
    :param twotheta: 2*Bragg angle, in degrees
    :param wavelength_a: incident beam wavelength, in Angstroms
    :param shape_factor: dimensionless shape factor, dependent on shape of crystallite
    :return: float, crystallite domain size in Angstroms
    """

    delta_theta = np.deg2rad(fwhm)
    costheta = np.cos(np.deg2rad(twotheta / 2.))
    return shape_factor * wavelength_a / (delta_theta * costheta)


def scherrer_fwhm(size, twotheta, wavelength_a, shape_factor=0.9):
    """
    Use the Scherrer equation to calculate the size of a crystallite from a peak width
      L = K * lambda / fwhm * cos(theta)
    See: https://en.wikipedia.org/wiki/Scherrer_equation
    :param size: crystallite domain size in Angstroms
    :param twotheta: 2*Bragg angle, in degrees
    :param wavelength_a: incident beam wavelength, in Angstroms
    :param shape_factor: dimensionless shape factor, dependent on shape of crystallite
    :return: float, peak full-width-at-half-max in degrees
    """
    costheta = np.cos(np.deg2rad(twotheta / 2.))
    return np.rad2deg(shape_factor * wavelength_a / (size * costheta))


def resolution2energy(res, twotheta=180.):
    """
    Calcualte the energy required to achieve a specific resolution at a given two-theta
    :param res: measurement resolution in A (==d-spacing)
    :param twotheta: Bragg angle in Degrees
    :return: float
    """
    theta = twotheta * pi / 360  # theta in radians
    return (h * c * 1e10) / (res * np.sin(theta) * e * 2 * 1000.)


def diffractometer_twotheta(delta=0, gamma=0):
    """Return the Bragg 2-theta angle for diffractometer detector rotations delta (vertical) and gamma (horizontal)"""
    delta = np.deg2rad(delta)
    gamma = np.deg2rad(gamma)
    twotheta = np.arccos(np.cos(delta) * np.cos(gamma))
    return np.rad2deg(twotheta)


def you_normal_vector(eta=0, chi=90, mu=0):
    """
    Determine the normal vector using the You diffractometer angles
      you_normal_vector(0, 0, 0) = [1, 0, 0]
      you_normal_vector(0, 90, 0) = [0, 1, 0]
      you_normal_vector(90, 90, 0) = [0, 0, -1]
      you_normal_vector(0, 0, 90) = [0, 0, -1]
    :param eta: angle (deg) along the x-axis
    :param mu: angle (deg) about the z-axis
    :param chi: angle deg) a
    :return: array
    """
    eta = np.deg2rad(eta)
    chi = np.deg2rad(chi)
    mu = np.deg2rad(mu)
    normal = np.array([np.sin(mu) * np.sin(eta) * np.sin(chi) + np.cos(mu) * np.cos(chi),
                       np.cos(eta) * np.sin(chi),
                       -np.cos(mu) * np.sin(eta) * np.sin(chi) - np.sin(mu) * np.cos(chi)])
    return normal


def wavevector_i(wavelength_a):
    """
    Returns a 3D wavevector for the initial wavevector
    """
    k = wavevector(wavelength_a)
    return np.array([0, 0, k])


def wavevector_f(wavelength_a, delta=0, gamma=0):
    """
    Returns a 3D wavevector for the final wavevector
    """
    k = wavevector(wavelength_a)
    sd = np.sin(np.deg2rad(delta))
    sg = np.sin(np.deg2rad(gamma))
    cd = np.cos(np.deg2rad(delta))
    cg = np.cos(np.deg2rad(gamma))
    return k * np.array([sg * cd, sd, cg * cd])


def wavevector_t(delta=0, gamma=0, energy_kev=None):
    """
    Returns the wavevector transfer in inverse-Angstroms
      Q = kf - ki
    """
    ki = wavevector_i(energy_kev)
    kf = wavevector_f(delta, gamma, energy_kev)
    return kf - ki


def polerisation_sigma(delta=0, gamma=0):
    """
    Returns the scattered polerisation vector in the sigma' channel
    """
    sd = np.sin(np.deg2rad(delta))
    sg = np.sin(np.deg2rad(gamma))
    cd = np.cos(np.deg2rad(delta))
    cg = np.cos(np.deg2rad(gamma))
    return np.array([cg, 0, -sg])


def polerisation_pi(delta=0, gamma=0):
    """
    Returns the scattered polerisation vector in the Pi' channel
    """
    sd = np.sin(np.deg2rad(delta))
    sg = np.sin(np.deg2rad(gamma))
    cd = np.cos(np.deg2rad(delta))
    cg = np.cos(np.deg2rad(gamma))
    return np.array([-sg * sd, cd, -cg * sd])


