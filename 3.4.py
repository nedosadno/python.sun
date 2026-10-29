import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
#   X_n = N(0, sigma_n^2), Y_n = N(0, sigma_n^2), sigma_n^2 = n*dh^2/3.
#   Плотность в плоскости (x,y):
#       p(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2)).
#   Переход к полярным координатам (r = sqrt(x^2+y^2)):
#       p(r) = r/sigma^2 * exp(-r^2/(2*sigma^2)) - распределение .
#   Моменты :
#       <r^2> = 2*sigma^2   (совпадает с точным <R_n^2> = 2*n*dh^2/3);
#       <r>   = sigma*sqrt(pi/2) = sqrt(pi*n*dh^2/6).
#
#   По экспериментальным данным мы считаем <r>_exp и <r^2>_exp,
#   строим по ним аналитическую кривую  и сравниваем с теорией.

# Параметры
N       = 10000      # число частиц
dh      = 1.0        # максимальный шаг по каждой оси
n_steps = 200        # число шагов по времени

# Моделирование: два независимых одномерных блуждания по x и по y
# Шаги по x и по y независимы, поэтому генерируем их отдельно.
steps_x = np.random.uniform(-dh, dh, size=(n_steps, N))
steps_y = np.random.uniform(-dh, dh, size=(n_steps, N))

# Накопительные суммы вдоль времени -> координаты x(t), y(t) для каждой частицы.
# traj_x[k-1, i] - координата x i-й частицы после k шагов, аналогично для y.
traj_x = np.cumsum(steps_x, axis=0)
traj_y = np.cumsum(steps_y, axis=0)

# Расстояние от начала координат для каждого момента времени
# r[k-1, i] = sqrt(x_k^2 + y_k^2) - расстояние i-й частицы от центра
# после k шагов.
r = np.sqrt(traj_x**2 + traj_y**2)

# Работаем с последним моментом времени (после n_steps шагов).
r_final = r[-1]

# Экспериментальные моменты по выборке
r_mean_exp = r_final.mean()                # <r>_exp
r2_mean_exp = (r_final**2).mean()          # <r^2>_exp

# Из экспериментального <r^2> строим параметр :
#     <r^2> = 2*sigma^2  =>  sigma^2 = <r^2>/2.
sigma2_exp = r2_mean_exp / 2.0

# Теоретические моменты 
sigma2_theor = n_steps * dh**2 / 3.0       # sigma^2 = n*dh^2/3
r2_mean_theor = 2.0 * sigma2_theor         # <r^2> = 2*sigma^2
r_mean_theor  = sigma2_theor**0.5 * np.sqrt(np.pi / 2)   # <r> = sigma*sqrt(pi/2)

# Печать сравнения моментов
print(f"Число шагов n = {n_steps}, частиц N = {N}")
print(f"<r^2> эксп: {r2_mean_exp:.4f},  теор: {r2_mean_theor:.4f},  "
      f"|Δ|/теор = {abs(r2_mean_exp - r2_mean_theor)/r2_mean_theor*100:.3f}%")
print(f"<r>   эксп: {r_mean_exp:.4f},  теор: {r_mean_theor:.4f},  "
      f"|Δ|/теор = {abs(r_mean_exp - r_mean_theor)/r_mean_theor*100:.3f}%")
print(f"Отношение <r^2>/<r>^2  эксп: {r2_mean_exp/r_mean_exp**2:.4f},  "
      f"теор (4/pi): {4/np.pi:.4f}")

#Гистограмма расстояний + две аналитические кривые 
fig, ax = plt.subplots(figsize=(10, 6))

# Гистограмма расстояний.
# density=True нормирует на 1, чтобы можно было накладывать плотности.
bins = 80
ax.hist(r_final, bins=bins, density=True, alpha=0.55, color='C0',
        label='гистограмма r')

# Сетка по r для гладкой отрисовки аналитических кривых.
rs = np.linspace(1e-6, r_final.max(), 500)

# Аналитическая функция по экспериментальному <r^2>.
#     f(r) = r/sigma^2 * exp(-r^2/(2*sigma^2)).
f_exp = rs / sigma2_exp * np.exp(-rs**2 / (2.0 * sigma2_exp))

# Аналитическая функция  по теоретическому sigma^2 = n*dh^2/3.
f_theor = rs / sigma2_theor * np.exp(-rs**2 / (2.0 * sigma2_theor))

ax.plot(rs, f_exp,   'C1',  lw=2,
        label=f'Кривая эксп: sigma^2 = {sigma2_exp:.2f}')
ax.plot(rs, f_theor, 'C3-', lw=2,
        label=f'Кривая теор: sigma^2 = {sigma2_theor:.2f}')

ax.set_xlabel('r — расстояние от начала координат')
ax.set_ylabel('p(r)')
ax.set_title(f'Распределение частиц по расстоянию (n = {n_steps}, N = {N})')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
