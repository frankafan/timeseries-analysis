import numpy as np
import matplotlib.pyplot as plt
from argon import *

S = np.append(np.flip(YanData[1:]),
              YanData)  # Create even array of structure factor S(k)
# Initialize lists
P = []
k_axis = []


def RDFcalc(S, dk, rho, data=YanData):
    """Return the radial distribution function g(r) and its corresponding r, given S, dk, rho, and YanData"""
    # Create global variable k_axis that stores the wavenumbers of S(k)
    global k_axis
    k_axis = np.linspace(-len(data) * dk, len(data) * dk, len(S))

    # Create global variable P that stores P(k), the Fourier transformed integral part
    global P
    P = []
    for i in range(len(k_axis)):  # Loop through k values
        P.append(np.pi / 1j * k_axis[i] * (S[i] - 1))

    p = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(P) / dk * 2 * np.pi))  # Inverse Fourier transform P(k)
    r = np.fft.fftshift(np.fft.fftfreq(len(k_axis), dk))  # Create time domain
    g = np.real(1 + 1 / (2 * np.pi ** 2 * rho * r) * p)  # Calculate radial distribution function
    return g, r * 2 * np.pi


[gn, rn] = RDFcalc(S, dk, massRho)

# Plot P(k) and g(r) with original k_max and dk
plt.figure()
plt.plot(k_axis, np.imag(P))
plt.xlim([-10, 10])
plt.xlabel('$k$')
plt.ylabel('Imaginary $P(k)$')
plt.title("Imaginary $P(k)$")
plt.savefig("Imaginary P(k)")

plt.figure()
plt.plot(rn, gn)
plt.xlim([0, rn[-1]])
plt.xlabel('$r$')
plt.ylabel('Real $g(r)$')
plt.title("Real $g(r)$")
plt.savefig("Real g(r)")

k_max = [15.24, 7.56, 3.72, 1.8, 0.84]  # Create list of k_max values
plt.figure()
for i in range(len(k_max)):  # Loop through k_max values
    YanData_temp = YanData.copy()[:int(k_max[i] / dk)]  # Concat structure factor
    S = np.append(np.flip(YanData_temp[1:]), YanData_temp)  # Create even array of structure factor S(k)

    # Clear arrays
    P = []
    k_axis = []
    # Create and plot g(r) with the updated structure factor
    [gn, rn] = RDFcalc(S, dk, massRho, YanData_temp)
    plt.plot(rn, gn, label=f'$k_\max=${k_max[i]}')
plt.legend()
plt.xlim([0, 30])
plt.xlabel('$r$')
plt.ylabel('Real $g(r)$')
plt.title("Real $g(r)$ with decreasing $k_\max$")
plt.savefig("Real g(r) with decreasing k_max")

dk_temps = [0.12, 0.24, 0.48, 0.96, 1.92]  # Create list of dk values
plt.figure()
for i in range(len(dk_temps)):  # Loop through dk values
    dk_temp = dk_temps[i]
    YanData_temp = YanData.copy()[::int(dk_temp / dk)]  # Slice structure factor
    S = np.append(np.flip(YanData_temp[1:]), YanData_temp)  # Create even array of structure factor S(k)
    # Clear arrays
    P = []
    k_axis = []
    # Create and plot g(r) with the updated dk and structure factor
    [gn, rn] = RDFcalc(S, dk_temp, massRho, YanData_temp)
    plt.plot(rn, gn, label=f'$dk=${dk_temp}')
plt.legend()
plt.xlim([0, 10])
plt.xlabel('$r$')
plt.ylabel('Real $g(r)$')
plt.title("Real $g(r)$ with increasing $dk$")
plt.savefig("Real g(r) with increasing dk")

plt.show()
