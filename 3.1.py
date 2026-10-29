# Одномерные случайные блуждания N частиц.
# x - координата частицы, y - порядковый номер частицы.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Параметры модели
N       = 60        # число частиц
dh      = 1.0       # максимальный шаг частицы
n_steps = 300       # число шагов по времени

# Инициализация: все частицы в точке x = 0
x = np.zeros(N)

# Массив траекторий: traj[i, k] - координата i-й частицы на k-м шаге
traj = np.zeros((N, n_steps + 1))
traj[:, 0] = x

# Основной цикл: на каждом шаге каждая частица получает
# независимое случайное смещение, равномерное в [-dh, dh]
for k in range(1, n_steps + 1):
    step = np.random.uniform(-dh, dh, N)
    x = x + step
    traj[:, k] = x

# Теоретическое СКО после n_steps шагов
sigma_theor = dh * np.sqrt(n_steps / 3)

# График
fig, ax = plt.subplots(figsize=(10, 8))

# Снимки состояния системы в несколько моментов времени.
# По оси x - координата частицы, по оси y - её номер.
times  = [0, 30, 60, 120, 200, n_steps]
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(times)))

for t, c in zip(times, colors):
    ax.scatter(traj[:, t], np.arange(N),
               s=22, color=c, zorder=3, label=f'шаг {t}')

# Для наглядности покажем траектории первых нескольких частиц
for i in range(5):
    ax.plot(traj[i, :], np.full(n_steps + 1, i),
            lw=0.6, color='gray', alpha=0.5, zorder=2)

ax.axvline(0, color='k', lw=0.6, alpha=0.6)
ax.set_xlabel('x — координата частицы')
ax.set_ylabel('i — номер частицы')
ax.set_title(f'Одномерные случайные блуждания: N = {N}, dh = {dh}, шагов = {n_steps}\n'
             f'теоретическое СКО после {n_steps} шагов: sigm = {sigma_theor:.2f}')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()

# Численная проверка: СКО всех частиц на последнем шаге
sigma_num = np.std(traj[:, -1]) # встроенная функция нампи

print(f"СКО (числ.)  = {sigma_num:.3f}")
print(f"СКО (теор.)  = {sigma_theor:.3f}")
print(f"Отношение    = {sigma_num / sigma_theor:.3f}")