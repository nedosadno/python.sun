import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 50000        # число частиц
dh = 1.0         # шаг броуновского движения
b = 0.1          # шаг дрейфа (сила тяжести)
n_steps = 10000  # число шагов

# Теоретическая высота
H_theor = dh**2 / (6 * b)

# Создаем окно с сеткой 2x2
fig, axes = plt.subplots(1, 3, figsize=(14, 10))
ax_dyn, ax_hist, ax_dep = axes[0], axes[1], axes[2]

 
# 1. Моделирование динамики 
x = np.zeros(N)
mean_x_history = np.zeros(n_steps + 1)

for k in range(n_steps):
    xi = np.random.uniform(-dh, dh, N)
    x = x + xi - b
    x[x < 0] = -x[x < 0] # Упругое отражение
    mean_x_history[k+1] = x.mean()

ax_dyn.plot(mean_x_history, lw=1, color='C0', label=r'Эксп.')
ax_dyn.axhline((H_theor + b/2), color='C3', ls='--', lw=2, label=f'Теория (H={H_theor:.2f})')
ax_dyn.set_xlabel('Номер шага (n)')
ax_dyn.set_ylabel(r'Средняя высота ')
ax_dyn.set_title('Выход на стационар')
ax_dyn.grid(alpha=0.3)
ax_dyn.legend()

 
# 2. Гистограмма распределения 
x_final = x # Берем конечные координаты

ax_hist.hist(x_final, bins=100, density=True, alpha=0.5, color='C0', label='Эксп. данные')
xs = np.linspace(0, max(x_final), 500)
ax_hist.plot(xs, (1/H_theor) * np.exp(-xs/H_theor), 'C3--', lw=2, label='Теория (p(x))')

# Подгонка экспоненты по экспериментальному среднему
mean_x_exp = x_final.mean()
ax_hist.plot(xs, (1/mean_x_exp) * np.exp(-xs/mean_x_exp), 'C1-', lw=2, 
             label=f'={mean_x_exp:.2f}')

ax_hist.set_xlabel('Высота (x)')
ax_hist.set_ylabel('Плотность вероятности (p(x))')
ax_hist.set_title(f'Стационарное распределение (N={N})')
ax_hist.grid(alpha=0.3)
ax_hist.legend()

 
# 4. Зависимость H от b 
b_vals = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
H_exp = []

for b_val in b_vals:
    x_temp = np.zeros(N)
    for _ in range(2000):
        xi = np.random.uniform(-dh, dh, N)
        x_temp = x_temp + xi - b_val
        x_temp[x_temp < 0] = -x_temp[x_temp < 0]
    H_exp.append(x_temp.mean())

H_exp = np.array(H_exp)
H_th = dh**2 / (6 * b_vals)  + (b_vals/2)

ax_dep.loglog(b_vals, H_exp, 'o-', color='C0', label='Эксперимент')
ax_dep.loglog(b_vals, abs(H_th - H_exp), 'C1', lw=2, label=r'Разница теории и эксперимента')
ax_dep.loglog(b_vals, H_th, 'r--', lw=2, label=r'Теория')
ax_dep.set_xlabel('Дрейф (b)')
ax_dep.set_ylabel(r'Средняя высота (H)')
ax_dep.set_title('Зависимость (H(b))')
ax_dep.grid(alpha=0.3, which='both')
ax_dep.legend()

plt.tight_layout()
plt.show()

print(f"Теоретическая высота: {H_theor:.4f}")
print(f"Экспериментальная высота: {mean_x_exp:.4f}")
print(f"Относительная ошибка: {abs(mean_x_exp - H_theor)/H_theor * 100:.2f}%")