import numpy as np
import matplotlib.pyplot as plt
from notch_filter import W
from scipy.signal import unit_impulse

# define constants
fs = 12  # [cycle/year]
f0 = 1  # [cycle/year]
M = 1.05
e = 0.05
q = np.exp(-1j * 2 * np.pi * f0 / fs)
p = (1 + e) * q
f = np.linspace(-fs / 2, fs / 2, 1200)

dt = 1 / fs
t0 = 0
tmax = 100  # [year]


def inverse_filter(a, x):
    """Return the input y from the convolved output x, in the convolution operation y*a=x."""
    y = np.zeros(len(x) - len(
        a) + 1)  # initialize input array using the convolution lengths
    a0 = a[0]
    for i in range(len(y)):  # iterate through the input
        ay = 0
        for k in range(1, len(a)):
            if i - k >= 0:  # system is causal, therefore values with negative time indeces are 0
                ay += a[k] * y[i - k]
        y[i] = 1 / a0 * (x[i] - ay)
    return y


def ratFilter(N, D, x):
    """Return filtered time series y for a given input x"""
    return inverse_filter(D, np.convolve(x, N))


N = np.array([q * np.conj(q), -q - np.conj(q), 1]) * M
D = np.array([p * np.conj(p), -p - np.conj(p), 1])

t = np.arange(t0, tmax, dt)
impulse = ratFilter(N, D, unit_impulse(len(t)))  # obtain impulse response
frequency = np.fft.fftshift(np.fft.fft(impulse))  # obtain frequency response

if __name__ == '__main__':
    # plot results
    plt.figure()
    plt.plot(t, impulse)
    plt.xlim([0, 6])
    plt.xlabel('$t$ [year]')
    plt.title("Notch filter impulse response with original $f_{fwmh}$")
    # plt.savefig("Notch filter impulse response with original f_fwmh")
    plt.figure()
    plt.plot(f, abs(W(f, M, p, q)),
             label='Theoretical')
    plt.plot(f, abs(frequency), '--', label='ratFilter')
    plt.legend()
    plt.xlim([0, 6])
    plt.xlabel('$f$ [cycle/year]')
    plt.ylabel('$|W(f)|$')
    plt.title("Notch filter frequency response with original $f_{fwmh}$")
    # plt.savefig("Notch filter frequency response with original f_fwmh")

    plt.show()
