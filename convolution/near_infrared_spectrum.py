import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("FTIR_ETL_TCCON.asc")

wavenumber = []
signal = []
for i in range(len(data)):
    wavenumber.append(data[i][0])
    signal.append(data[i][1])

dv = 0.007533
v = np.arange(-4, 4, dv)

d1 = 1
d2 = 4
f1 = 2 * d1 * np.sin(2 * np.pi * v * d1) / (2 * np.pi * v * d1)
f2 = 2 * d2 * np.sin(2 * np.pi * v * d2) / (2 * np.pi * v * d2)

conv1 = np.convolve(signal, f1, mode='same') * dv
conv2 = np.convolve(signal, f2, mode='same') * dv

plt.figure()
plt.plot(wavenumber, signal)
plt.xlabel("$v(cm^{-1})$")
plt.ylabel("Spectral signal")
plt.title("Spectrum at East Trout Lake")
plt.savefig("3-1")

plt.figure()
plt.plot(v, f1, label="$\Delta=1$")
plt.plot(v, f2, label="$\Delta=4$")
plt.legend()
plt.xlabel("$v(cm^{-1})$")
plt.title("Sine function with $\Delta=1$ and $\Delta=4$")
plt.savefig("3-2")

plt.figure()
plt.plot(wavenumber, signal, label='Original')
plt.plot(wavenumber, conv1, label='Convolved with $\Delta=1$')
plt.plot(wavenumber, conv2, label='Convolved with $\Delta=4$')
plt.legend()
plt.xlim([4000, 4050])
plt.xlabel("$v(cm^{-1})$")
plt.ylabel("Spectral signal")
plt.title("Original vs. Convolved time series")
plt.savefig("3-3")

plt.figure()
plt.plot(wavenumber, signal, label='Original')
plt.plot(wavenumber, conv1, label='Convolved with $\Delta=1$')
plt.legend()
plt.xlim([4000, 4050])
plt.xlabel("$v(cm^{-1})$")
plt.ylabel("Spectral signal")
plt.title("Original vs. Convolved time series with $\Delta=1$")
plt.savefig("3-4")

plt.figure()
plt.plot(wavenumber, signal, label='Original')
plt.plot(wavenumber, conv2, label='Convolved with $\Delta=4$')
plt.legend()
plt.xlim([4000, 4050])
plt.xlabel("$v(cm^{-1})$")
plt.ylabel("Spectral signal")
plt.title("Original vs. Convolved time series with $\Delta=4$")
plt.savefig("3-5")

plt.figure()
plt.plot(wavenumber, signal, label='Original')
plt.plot(wavenumber, conv1, label='Convolved with $\Delta=1$')
plt.plot(wavenumber, conv2, label='Convolved with $\Delta=4$')
plt.legend()
plt.xlabel("$v(cm^{-1})$")
plt.ylabel("Spectral signal")
plt.title("Original vs. Convolved time series")
plt.savefig("3-6")

plt.show()
