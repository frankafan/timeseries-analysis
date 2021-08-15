from co2data import *
import matplotlib.pyplot as plt
from rat_filter import ratFilter

# define constants
fs = 12  # [cycle/year]
f0 = 1  # [cycle/year]
M = 1.05
e = 0.05
q = np.exp(-1j * 2 * np.pi * f0 / fs)
p = (1 + e) * q
N = np.array([q * np.conj(q), -q - np.conj(q), 1]) * M
D = np.array([p * np.conj(p), -p - np.conj(p), 1])


def line_fit(time, data):
    """Return the y values of the original data's line-fit"""
    y = []
    fit = np.polyfit(time, data, 1)
    for t in time:
        y.append(
            fit[0] * t + fit[1])  # convert polynomial coefficients to y values
    return np.array(y)


trend = line_fit(time, co2Data)  # store trend data
detrendData = co2Data - trend  # remove trend

dt = (co2TimeRange[1] - co2TimeRange[0]) / len(
    time)  # find dt from original data
FT = np.fft.fftshift(
    np.fft.fft(detrendData))  # Fourier transform de-trended data
FT_freq = np.fft.fftshift(
    np.fft.fftfreq(len(detrendData), dt))  # obtain frequency spectrum

FT2 = FT.copy()  # create new memory address for Fourier transformed time series
for i in range(len(FT_freq)):
    # change Fourier transform value to 0 for frequencies beyond 0.9 cycles per year
    if abs(FT_freq[i]) > 0.9:
        FT2[i] = 0
IFT = np.fft.ifft(np.fft.ifftshift(
    FT2))  # inverse Fourier transform to obtain filtered time series

# plot results
plt.figure()
plt.plot(time, co2Data, label='Original data')
plt.plot(time, detrendData, label='De-trended data')
plt.legend()
plt.xlabel('$t$ [year]')
plt.ylabel('$CO_2$ [ppm]')
plt.title("Atmospheric $CO_2$ over time")
# plt.savefig("Atmospheric CO2 over time")

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(FT_freq, abs(FT))
plt.title("Fourier transformed time series")
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(FT_freq, abs(FT))
plt.xlabel('$f$ [cycle/year]')
plt.ylabel('Amplitude')
plt.xlim([0, 3.5])
# plt.savefig("Fourier transformed time series")

plt.figure()
plt.plot(time, co2Data, label='Original data')
plt.plot(time, ratFilter(N, D, detrendData) + trend, label='Notch filter')
plt.plot(time, IFT + trend, label='Fourier transform')
plt.xlabel('$t$ [year]')
plt.ylabel('$CO_2$ [ppm]')
plt.legend()
plt.title("Original and filtered time series of $CO_2$ levels")
# plt.savefig("Original and filtered time series of CO2 levels")

plt.show()
