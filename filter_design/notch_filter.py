import numpy as np
import matplotlib.pyplot as plt

# define constants
fs = 12  # [cycle/year]
f0 = 1  # [cycle/year]
M = 1.05
e = 0.05

q = np.exp(-1j * 2 * np.pi * f0 / fs)
p = (1 + e) * q

f = np.linspace(-fs / 2, fs / 2, 1200)


def W(f, M, p, q):
    """Return the theoretical z-transform of the notch filter."""
    return M * (z_from_f(f) - q) / (z_from_f(f) - p) * (z_from_f(f) - np.conj(q)) / (z_from_f(f) - np.conj(p))


def z_from_f(f):
    """Return the imaginary number z, given frequency f"""
    return np.exp(-1j * (2 * np.pi * f) * 1 / fs)


if __name__ == '__main__':
    # plot results
    plt.figure()
    plt.plot(f, abs(W(f, M, p, q) * np.conj(W(f, M, p, q))))
    plt.xlabel('$f$ [cycle/year]')
    plt.ylabel('$|W(f)|^2$')
    plt.title("Notch filter power spectrum")
    plt.show()
