# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from read_srt import read_spectra, Spectra
plt.rcParams['font.size'] = 30

S = read_spectra('az_el_scan_RPSG_sun.rad')


i = 8
plt.plot(S[i].freq, S[i].spectrum)
plt.axvline(S[i].freq[10], color="r")
plt.axvline(S[i].freq[-10], color="r")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (K)")
plt.title(r"Raw Frequency Spectrum, Az=$191.5^\circ$, El=$37.1^\circ$, dAz=$-24.3^\circ$, dEl=$2.6^\circ$")
plt.show()
print(S[i].az)
print(S[i].el)
print(S[i].d_az)
print(S[i].d_el)




step = 2  # 2 degree step size
scansize = 60  # +/- 30 degrees in each direction

# hardcode the initial commands for the azimuth and elevation scans
i0az = [i for i in range(len(S)) if 'offset -24.3 2.6' in S.COMMAND[i]][0]
i0el = [i for i in range(len(S)) if 'offset 5.7 -27.4' in S.COMMAND[i]][0]
nsteps = abs(i0el - i0az)  # assuming no other commands between az and el scan

azscan = Spectra([S[i] for i in range(i0az, i0az + nsteps)])
elscan = Spectra([S[i] for i in range(i0el, i0el + nsteps)])

freqbinlimit = 15
azP, azdP = azscan.average_power()  # lower=freqbinlimit, upper=freqbinlimit)
elP, eldP = elscan.average_power()  # lower=freqbinlimit, upper=freqbinlimit)

# Graphing Azimuth Data for Sun ########


normAzRange = np.linspace(-30, 30, num=len(azP))


fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(normAzRange, azP, yerr=azdP, fmt="o", label="Measured Data")
plt.title("Measured Temperature Variation by Azimuth")
ax.set_xlabel('Azimuth offset (deg)')
ax.set_ylabel(r'Temperature (K)')


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-1 * (x - x0) ** 2 / (2 * sigma ** 2))


# Creating Fit for Azimuthal Data
azbounds = ([100, 600, -10, 0], [200, 700, 10, 10])

poptAz, pcovAz = curve_fit(gauss, normAzRange, azP, sigma=azdP)

bigAzRange = np.linspace(-30, 30, num=1100)


fit_ydata = gauss(bigAzRange, poptAz[0], poptAz[1], poptAz[2], poptAz[3])

perrAz = np.sqrt(np.diag(pcovAz))
print(perrAz)

plt.errorbar(bigAzRange, fit_ydata, label="Gaussian Fit")
plt.legend()
plt.show()
# Printing fit details for azimuth data
print(r"The fit parameters:"
      "\n"
      r"Offset: {} \pm {}"
      "\n"
      r"Mean: {} \pm {}".format(poptAz[0], perrAz[0], poptAz[2], perrAz[2]))
print(r"Standard Deviation: {} \pm {}"
      "\n"
      "Amplitude: {} \pm {}".format(poptAz[3], perrAz[3], poptAz[1], perrAz[1]))

# Calculating Important values **note that HPBW/FWHM value must be corrected by cos(mean elevation)**
az_els = []
for i in range(len(azscan)):
    az_els.append(azscan[i].el)
az_mean_el = np.mean(az_els)  # not all elevations for azimuth scan are equal

hpbw_corr = np.cos(az_mean_el * np.pi / 180)
hpbw_az_fit = 2 * np.sqrt(2 * np.log(2)) * poptAz[3] * hpbw_corr
hpbw_az_fit_err = 2 * np.sqrt(2 * np.log(2)) * perrAz[3] * hpbw_corr
print("\n" "Values from fit: \n"
      r"$T_sys$ = {} $\pm$ {}"
      "\n"
      "$T_ant$ = {} $\pm$ {}".format(poptAz[0], perrAz[0], poptAz[1], perrAz[1]))
print(r"HPBW = {} $\pm$ {}".format(hpbw_az_fit, hpbw_az_fit_err))


# Plotting Azimuthal residuals
az_residuals = azP - gauss(normAzRange, poptAz[0], poptAz[1], poptAz[2], poptAz[3])
plt.scatter(normAzRange, az_residuals)
plt.axhline(0, color="r")
plt.title("Residuals Graph of Azimuth Scan Values compared to Fit")
plt.xlabel('Azimuth offset (deg)')
plt.ylabel('Average Measured Temperature - Curve Fit (K)')
plt.show()


# 1st Attempt at Plotting Elevation Scan Data

normElRange = np.linspace(-30, 30, num=len(elP))

fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(normElRange, elP, yerr=eldP, fmt="o")
plt.title("Measured Temperature Variation by Elevation")
ax.set_xlabel('Elevation offset (deg)')
ax.set_ylabel(r'Temperature (K)')
plt.show()


# Plotting Elevation Data and Fitting #########

el_slice = np.where(normElRange > -10)  # excludes ground reflection data
elPsl = elP[el_slice]
eldPsl = eldP[el_slice]
smElRange = np.linspace(-10, 30, num=len(elPsl))
# shrunkRange = np.arange(-10, 11, step)
bigShrunk = np.linspace(-10, 30, num=1000)
# bigShrunk = np.linspace(-10, 10, num=1000)
fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(smElRange, elPsl, yerr=eldPsl, fmt="o", label="Measured Data")
# ax.errorbar(shrunkRange, elP[10:21], yerr=eldP[10:21], fmt="o", label="Measured Data")
plt.title("Measured Temperature Variation by Elevation")
ax.set_xlabel('Elevation offset (deg)')
ax.set_ylabel(r'Temperature (K)')
# ax.set(xlim=(-10, 10))

poptEl, pcovEl = curve_fit(gauss, smElRange, elPsl, p0=poptAz, sigma=eldPsl)
# poptEl, pcovEl = curve_fit(gauss, shrunkRange, elP[10:21], sigma=eldP[10:21], bounds=azbounds)

perrEl = np.sqrt(np.diag(pcovEl))
print(perrEl)

el_fit_ydata = gauss(bigShrunk, poptEl[0], poptEl[1], poptEl[2], poptEl[3])
plt.plot(bigShrunk, el_fit_ydata, label="Gaussian fit")
plt.legend()
plt.show()

# Printing fit details for elevation data:
print(r"The fit parameters:"
      "\n"
      r"Offset: {} \pm {}"
      "\n"
      r"Mean: {} \pm {}".format(poptEl[0], perrEl[0], poptEl[2], perrEl[2]))
print(r"Standard Deviation: {} \pm {}"
      "\n"
      "Amplitude: {} \pm {}".format(poptEl[3], perrEl[3], poptEl[1], perrEl[1]))

# Calculating Important values
hpbw_el_fit = 2 * np.sqrt(2 * np.log(2)) * poptEl[3]
hpbw_el_fit_err = 2 * np.sqrt(2 * np.log(2)) * perrEl[3]
print("\n" "Values from fit: \n"
      r"$T_sys$ = {} \pm {}"
      "\n"
      "$T_ant$ = {} \pm {}".format(poptEl[0], perrEl[0], poptEl[1], perrEl[1]))
print(r"HPBW = {} \pm {}".format(hpbw_el_fit, hpbw_el_fit_err))


# Plotting Residuals of Elevation Data
elResiduals = elPsl - gauss(smElRange, poptEl[0], poptEl[1], poptEl[2], poptEl[3])

plt.errorbar(smElRange, elResiduals, yerr=eldPsl, fmt="o")
plt.axhline(0, color="r")
plt.title("Residuals Graph of Elevation Scan Values compared to Fit")
plt.xlabel('Elevation offset (deg)')
plt.ylabel('Average Measured Temperature - Curve Fit (K)')
plt.show()


# Calculating Brightness Temperature and Solar Flux Density ###########

# to determine "peak" (average) frequency in azimuth:
freak = []
for i in range(39):
    a = np.mean(S[i].freq)
    freak.append(a)

for i in range(48, len(S) - 1):
    a = np.mean(S[i].freq)
    freak.append(a)

meanfreak = np.mean(freak)
freak0 = S[i].freq0
print("The mean of all frequencies in the azimuth dataset {} MHz".format(meanfreak))
print(freak0)

# Define constants:
au = 1.496e11  # m
h = 6.626e-34  # J/Hz
R_sol = 6.957e8  # m
c = 2.997e8  # m/s
k_b = 1.381e-23  # J/K
nu = meanfreak * 10 ** 6
nu0 = freak0 * 10 ** 6
dnu = S[0].dfreq * 10 ** 6
eff = 0.6  # percentage of dish actually used
R_ant = 2.286  # m, converted from 7.5 feet

# Averaging HPBW/FWHM of Azimuth and Elevation data:
HPBW = hpbw_az_fit + hpbw_el_fit / 2
HPBW_err = np.sqrt(hpbw_az_fit_err ** 2 + hpbw_el_fit_err ** 2) / 2

# Calculating Solid Angles and uncertainty:
# remembering that HPBW is deg and np.cos takes radians
Omega_ant = 2 * np.pi * (1 - np.cos(HPBW / 2 * (np.pi / 180)))
dOmega_ant = np.pi * np.sin(HPBW / 2 * np.pi / 180) * HPBW_err
Omega_sol = 2 * np.pi * (1 - np.cos(np.arctan(R_sol / au)))

# Averaging Antenna Temperature and carrying over uncertainty:
T_ant = (poptAz[1] + poptEl[1]) / 2
dT_ant = np.sqrt(perrAz[1] ** 2 + perrEl[1] ** 2) / 2

T_ant2 = 2 * T_ant
dT_ant2 = 2 * dT_ant

# Estimating Brightness Temperature and uncertainty
T_B = Omega_ant * T_ant2 / Omega_sol
dT_B = np.sqrt((T_ant2 * dOmega_ant / Omega_sol) ** 2 + (Omega_ant * dT_ant2 / Omega_sol) ** 2)

# Solar spectral radiance (Planck's Law):
x = (k_b * T_B)
exPlanck = np.exp((h * nu) / (x))

B_nu = (2 * h * nu ** 3) / ((c ** 2) * (exPlanck - 1))

# B_nu uncertainty is gonna be a bad time
dB_nu1 = ((2 * h * (nu ** 2) * ((h * nu - 3 * x) * exPlanck + 3 * x)) / (c ** 2 * x * (exPlanck - 1) ** 2) * dnu)
dB_nu2 = ((2 * h ** 2 * nu ** 4 * exPlanck) / (c ** 2 * x * T_B * (exPlanck - 1) ** 2) * dT_B)
dB_nu = np.sqrt(dB_nu1 ** 2 + dB_nu2 ** 2)

# Solar Flux Density
F_nu = B_nu * Omega_sol
dF_nu = dB_nu * Omega_sol

# Printing Our values:
val_set = [HPBW, HPBW_err, T_ant, dT_ant, T_ant2, dT_ant2, T_B, dT_B, F_nu, dF_nu]
solar_vals = (r"Averaged HPBW: {:.5} $\pm$ {:.5}"
              "\n"
              r"Measured Antenna Temperature = {:.5} $\pm$ {:.5} K"
              "\n"
              r"Full Antenna Temperature $T_ant$ = {:.5} $\pm$ {:.5} K"
              "\n"
              r"Estimated Brightness Temperature $T_B$ = {:.5} $\pm$ {:.5} K"
              "\n"
              r"Solar Flux Density $F_\nu$ = {:.5} $\pm$ {:.5} W m^-2 Hz^-1".format(*val_set))
print(solar_vals)
print("Date of Measurement was {}".format(S[50].date))

f = open("solar_values.txt", "w")

f.write(solar_vals)
f.close()
