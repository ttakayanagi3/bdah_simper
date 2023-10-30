import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.arange(0, 1, 1/fs)

freq1 = 5
sin_wave1 = np.sin(2 * np.pi * freq1 * t)

freq2 = 20
sin_wave2 = np.sin(2 * np.pi * freq2 * t)

freq3 = 50
sin_wave3 = np.sin(2 * np.pi * freq3 * t)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, sin_wave1)
plt.title(f'Sin Wave - Frequency {freq1} Hz')

plt.subplot(3, 1, 2)
plt.plot(t, sin_wave2)
plt.title(f'Sin Wave - Frequency {freq2} Hz')

plt.subplot(3, 1, 3)
plt.plot(t, sin_wave3)
plt.title(f'Sin Wave - Frequency {freq3} Hz')

plt.tight_layout()
plt.show()
