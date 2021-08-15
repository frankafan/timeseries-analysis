import numpy as np
import matplotlib.pyplot as plt

BIT_CONVERT = True

X_FILE = 'PHL_data.txt'
Y_FILE = 'MLAC_data.txt'
dt = 1.0  # [s]

x = np.genfromtxt(X_FILE).flatten()
y = np.genfromtxt(Y_FILE).flatten()
t = np.arange(0, len(y) * dt, dt)
if BIT_CONVERT:
    x = np.sign(x)
    y = np.sign(y)

x_padded = np.pad(x, (0, len(x) - 1), 'constant')
y_padded = np.pad(y, (0, len(y) - 1), 'constant')
w = np.fft.fftshift(
    np.fft.ifft(np.fft.fft(np.flip(x_padded)) * np.fft.fft(y_padded)))
tau = np.fft.fftshift(np.append(t, np.flip(-t)[:-1]))

plt.figure()
plt.plot(tau, w)
plt.xlim([-240, 240])
plt.xlabel('lag [s]')
plt.ylabel('cross-correlation')
if BIT_CONVERT:
    plt.title("Bit-converted cross-correlation")
    plt.savefig("Bit-converted cross-correlation")
else:
    plt.title("Cross-correlation")
    plt.savefig("Cross-correlation")
plt.show()
