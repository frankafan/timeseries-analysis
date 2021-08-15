import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

FILE = 'nwao.vh1'

data = np.genfromtxt(FILE)
dt = 10.0  # [s]

t = data[:, 0]
v = data[:, 1]

power_spectrum = abs(np.fft.fftshift(np.fft.fft(v))) ** 2
w = np.fft.fftshift(np.fft.fftfreq(len(data), dt))


def line_fit(time, data):
    """Return the y values of the original data's line-fit"""
    y = []
    fit = np.polyfit(time, data, 1)
    for t in time:
        y.append(
            fit[0] * t + fit[1])  # convert polynomial coefficients to y values
    return np.array(y)


trend = line_fit(t, v)
window = 1 - np.cos(2 * np.pi * np.arange(len(v)) / len(v))
v_clean = (v - trend) * window
power_spectrum_clean = abs(np.fft.fftshift(np.fft.fft(v_clean))) ** 2

plt.figure()
plt.plot(t / 60, v)
plt.xlabel('time [hour]')
plt.ylabel('velocity')
plt.title("NWAO raw seismic data")
plt.savefig("NWAO raw seismic data")

plt.figure()
plt.plot(w * 1e3, power_spectrum)
plt.xlabel('$\omega$ [mHz]')
plt.ylabel('intensity')
plt.title("Power spectrum of raw seismic data")
plt.savefig("Power spectrum of raw seismic data")

plt.figure()
plt.plot(w * 1e3, power_spectrum_clean)
plt.xlabel('$\omega$ [mHz]')
plt.ylabel('intensity')
plt.title("Power spectrum of filtered and de-trended seismic data")
plt.savefig("Power spectrum of filtered and de-trended seismic data")

plt.figure()
plt.plot(w * 1e3, power_spectrum, label='Original data')
plt.plot(w * 1e3, power_spectrum_clean, label='Filtered and de-trended data')
plt.legend()
plt.xlim([0.1, 2.6])
plt.ylim([0, 6e14])
plt.xlabel('$\omega$ [mHz]')
plt.ylabel('intensity')
plt.title("Power spectrum of seismic data")
plt.savefig("Power spectrum of seismic data")

plt.figure()
plt.plot(w * 1e6, power_spectrum_clean)
plt.xlim([300, 10000])
plt.ylim([0, 1e14])
plt.xlabel('$\omega$ [$\mu$Hz]')
plt.ylabel('intensity')
plt.title("Power spectrum of filtered and de-trended seismic data")
plt.savefig("Power spectrum of filtered and de-trended seismic data")

# peaks, _ = find_peaks(power_spectrum_clean, height=0.2e14)
#
# peak_dict = {}
# for peak in peaks:
#     peak_dict[w[peak] * 1e6] = peak
#
# print(peak_dict)

plt.figure()
plt.plot(w * 1e6, power_spectrum_clean)
# plt.plot(w[peaks] * 1e6, power_spectrum_clean[peaks], '.')
plt.xlim([300, 10000])
plt.ylim([0, 1e14])
plt.xlabel('$\omega$ [$\mu$Hz]')
plt.ylabel('intensity')
plt.title("Power spectrum of seismic data with labelled modes")

plt.plot(
    np.array(
        [w[13461], w[11866], w[11934], w[12017], w[12026], w[12165], w[12168],
         w[12170], w[12219], w[12249], w[12336], w[12410], w[12515], w[12618],
         w[12661], w[12765], w[12842], w[12972]]) * 1e6,
    [power_spectrum_clean[13461], power_spectrum_clean[11866],
     power_spectrum_clean[11934], power_spectrum_clean[12017],
     power_spectrum_clean[12026], power_spectrum_clean[12165],
     power_spectrum_clean[12168], power_spectrum_clean[12170],
     power_spectrum_clean[12219], power_spectrum_clean[12249],
     power_spectrum_clean[12336], power_spectrum_clean[12410],
     power_spectrum_clean[12515], power_spectrum_clean[12618],
     power_spectrum_clean[12661], power_spectrum_clean[12765],
     power_spectrum_clean[12842], power_spectrum_clean[12972]],
    '.')
plt.annotate("${}_{12}S_{13}$", (w[13461] * 1e6, power_spectrum_clean[13461]))
plt.annotate("${}_{0}T_{5}$", (w[11866] * 1e6, power_spectrum_clean[11866]))
plt.annotate("${}_{0}T_{7}$", (w[11934] * 1e6, power_spectrum_clean[11934]))
plt.annotate("${}_{0}S_{9}$", (w[12017] * 1e6, power_spectrum_clean[12017]))
plt.annotate("${}_{0}T_{10}$", (w[12026] * 1e6, power_spectrum_clean[12026]))
plt.annotate("${}_{0}T_{15}$", (w[12165] * 1e6, power_spectrum_clean[12165]))
plt.annotate("${}_{2}S_{9}$", (w[12168] * 1e6, power_spectrum_clean[12168]))
plt.annotate("${}_{3}S_{5}$", (w[12170] * 1e6, power_spectrum_clean[12170]))
plt.annotate("${}_{0}T_{17}$", (w[12219] * 1e6, power_spectrum_clean[12219]))
plt.annotate("${}_{2}S_{11}$", (w[12249] * 1e6, power_spectrum_clean[12249]))
plt.annotate("${}_{3}S_{9}$", (w[12336] * 1e6, power_spectrum_clean[12336]))
plt.annotate("${}_{6}S_{5}$", (w[12410] * 1e6, power_spectrum_clean[12410]))
plt.annotate("${}_{1}T_{17}$", (w[12515] * 1e6, power_spectrum_clean[12515]))
plt.annotate("${}_{5}S_{10}$", (w[12618] * 1e6, power_spectrum_clean[12618]))
plt.annotate("${}_{0}S_{36}$", (w[12661] * 1e6, power_spectrum_clean[12661]))
plt.annotate("${}_{0}S_{41}$", (w[12765] * 1e6, power_spectrum_clean[12765]))
plt.annotate("${}_{4}T_{12}$", (w[12842] * 1e6, power_spectrum_clean[12842]))
plt.annotate("${}_{2}T_{24}$", (w[12972] * 1e6, power_spectrum_clean[12972]))
plt.savefig("Power spectrum of seismic data with labelled modes")

plt.show()
