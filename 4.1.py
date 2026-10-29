import numpy as np
import matplotlib.pyplot as plt

A = 1.0       # Амплитуда сигнала (Вольты)
f_sig = 120   # Частота сигнала (Герцы)
dt = 0.001    # Шаг дискретизации (1 миллисекунда)
N = 49      # Количество отсчетов (длина выборки)


# Создаем ось времени: от 0 до (N-1)*dt
t = np.arange(N) * dt

# Генерируем сам синусоидальный сигнал по формуле A * sin(2 * pi * f * t)
signal = A * np.sin(2 * np.pi * f_sig * t)


# np.fft.rfft - берет наш сигнал и выдает комплексные числа (спектр)
# Используем rfft,он выдает только положительные частоты 
fft_result = np.fft.rfft(signal)

# Генерируем ось частот 
freqs = np.fft.rfftfreq(N, d=dt)


# Функция Фурье выдает числа, которые больше реальной амплитуды в (N/2) раз. (так как учитываются только положительные)
# Поэтому мы берем модуль (np.abs) и умножаем на 2/N.
amplitudes = (2.0 / N) * np.abs(fft_result)

# Находим индекс (номер) элемента в массиве, где частота равна 120 Гц
index_120 = np.argmax(amplitudes) # Функция argmax сама найдет самый высокий пик

# Достаем амплитуду из спектра на этой частоте
fft_amp = amplitudes[index_120]
real_freq = freqs[index_120]

print(f"Частота пика: {real_freq:.1f} Гц")
print(f"Амплитуда по Фурье: {fft_amp:.4f} В")
print(f"Аналитическая амплитуда: {A:.4f} В")
print(f"Разница: {abs(A - fft_amp):.10f} В")

plt.figure(figsize=(10, 5))
# График 1: Сам сигнал
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title("Исходный сигнал (120 Гц)")
plt.xlabel("Время (с)")
plt.ylabel("Вольты")
plt.grid(True)
plt.xlim(0, 0.05)
plt.plot(t, signal, 'o', color='red', markersize=3, label='Отсчеты АЦП')
plt.legend()

# График 2: Спектр 
plt.subplot(1, 2, 2)
plt.plot(freqs, amplitudes)
plt.title("Спектр Фурье")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда (В)")
plt.xlim(0, 300) 
plt.grid(True)

plt.tight_layout()
plt.show()