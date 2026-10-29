import numpy as np                                    # импорт numpy для массивов и математики
import matplotlib.pyplot as plt                       # импорт pyplot для графиков

G     = 6.674e-8          # гравитационная постоянная, см^3/(г*с^2)
M_sun = 1.989e33          # масса Солнца, г
AU    = 1.496e13          # астрономическая единица, см
mu    = G * M_sun         # гравитационный параметр mu = G*M_sun, см^3/с^2

# Орбита задаётся перигелием r_p и эксцентриситетом e
e   = 0.7                          # эксцентриситет (сильно вытянутая орбита)
r_p = 0.5 * AU                     # перигелий, см

# Скорость в перигелии. В перигелии вектор скорости перпендикулярен
# радиус-вектору, поэтому вся скорость тангенциальная. Формула выводится
# из интеграла энергии: v^2/2 - mu/r = -mu/(2a), где a = r_p/(1-e).
# Отсюда v_p = sqrt(mu*(1+e)/r_p).
v_p = np.sqrt(mu * (1 + e) / r_p)  # скаляр, скорость в перигелии, см/с

# Большая полуось из связи r_p = a*(1-e)
a   = r_p / (1 - e)                # скаляр, большая полуось, см

# Период из третьего закона Кеплера: T^2 = 4*pi^2*a^3/mu.
T_orb = 2 * np.pi * np.sqrt(a**3 / mu)  # скаляр, период обращения, с

# Начальные условия: планета в перигелии на оси x, скорость вдоль y.
r0 = np.array([r_p, 0.0])          # начальная позиция (x, y)
v0 = np.array([0.0, v_p])          # начальная скорость (vx, vy)


def simulate(N):
    # Один прогон: N шагов за период T_orb.
    # Возвращает dt и массив позиций R.
    #
    #  Систематический дрейф - это
    # медленное накопление ошибки в одну сторону, из-за которого
    # за много шагов величина уходит далеко от истинного значения.

    # a = -mu*r/|r|^3 параллельно радиус-вектору r, поэтому [r x a] = 0
    # и момент импульса сохраняется.
    dt = T_orb / N                 # шаг по времени, с
    r  = r0.copy()                 # текущая позиция
    v  = v0.copy()                 # текущая скорость

    R = np.zeros((N + 1, 2))       # вся траектория: R[k] = (x_k, y_k)
    R[0] = r                       # начальная позиция

    for n in range(N):
        # 1 - половина сдвига скорости по старой силе.
        # мы как бы разбиваем один шаг на половинку, чтоб увидеть значения на концах шага
        rn     = np.linalg.norm(r)
        v_half = v + 0.5 * (-mu * r / rn**3) * dt

        # 2 - полный шаг координаты со средней скоростью v_half.
        r_new  = r + v_half * dt

        # 3 - половина сдвига скорости по новой силе.
        # Схема 1-2-3 симметрична
        rn_new = np.linalg.norm(r_new)
        v_new  = v_half + 0.5 * (-mu * r_new / rn_new**3) * dt

        r, v = r_new, v_new
        R[n+1] = r

    return dt, R


def area(R, t, T_0, tau):
    # Площадь криволинейного сектора, заметаемого радиус-вектором
    # за интервал [T_0, T_0 + tau].
    #
    # Считаем прямым суммированием элементарных треугольников.
    # Элементарный треугольник: вершины в центре, в r_k и r_{k+1}.
    # Площадь одного: dS_k = 0.5*|[r_k x r_{k+1}]|.
    # Множитель 0.5 - потому что площадь треугольника равна половине
    # площади параллелограмма на тех же двух векторах.
    dt = t[1] - t[0]

    i0 = max(0, min(int(T_0 / dt), len(t) - 2))              # индекс начала окна
    i1 = max(i0 + 1, min(int((T_0 + tau) / dt), len(t) - 1)) # индекс конца окна

    S = 0.0
    for k in range(i0, i1):
        x1, y1 = R[k]
        x2, y2 = R[k+1]
        dx = x2-x1
        dy = y2 - y1
        S += 0.5 * abs(x1*dy - y1*dx)
    return S

# Основной расчёт
N  = 400
dt, R = simulate(N)
t  = np.arange(N + 1) * dt

# Секторная скорость L/(2m): в перигелии L = r_p * v_p
sec = 0.5 * r_p * v_p


# Данные для графика S(T_0, tau=const)
tau     = T_orb / 20
T0_grid = np.linspace(0, T_orb - tau, 200)
S_T0    = np.array([area(R, t, T0, tau) for T0 in T0_grid])

# Данные для графика S(T_0=const, tau)
T0_const   = 0.0
tau_values = np.linspace(0, T_orb, 200)
S_tau      = np.array([area(R, t, T0_const, tv) for tv in tau_values])

# Восстановление скорости из координат (центральная разность)
# и вычисление момента импульса L/m = x*vy - y*vx
N_pts = len(t)
vx = np.zeros(N_pts)
vy = np.zeros(N_pts)
for i in range(1, N_pts - 1):
    vx[i] = (R[i+1, 0] - R[i-1, 0]) / (t[i+1] - t[i-1])
    vy[i] = (R[i+1, 1] - R[i-1, 1]) / (t[i+1] - t[i-1])
vx[0]  = (R[1, 0] - R[0, 0]) / dt
vy[0]  = (R[1, 1] - R[0, 1]) / dt
vx[-1] = (R[-1, 0] - R[-2, 0]) / dt
vy[-1] = (R[-1, 1] - R[-2, 1]) / dt
L = R[:, 0] * vy - R[:, 1] * vx


fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (1) Траектория - замкнутый эллипс с Солнцем в фокусе
ax[0, 0].plot(R[:, 0] / AU, R[:, 1] / AU)
ax[0, 0].plot(0, 0, 'yo', ms=10)
ax[0, 0].set_aspect('equal')
ax[0, 0].grid(alpha=0.3)
ax[0, 0].set_xlabel('x, а.е.')
ax[0, 0].set_ylabel('y, а.е.')
ax[0, 0].set_title(f'Траектория\nr_p = {r_p/AU:.2f} а.е., e = {e}, a = {a/AU:.2f} а.е.')

# (2) S(T_0, tau = const) - горизонталь на уровне L*tau/(2m)
ax[0, 1].plot(T0_grid / T_orb, S_T0, label='числ.')
ax[0, 1].axhline(sec * tau, color='r', ls='--', label=r'$L/(2m)\cdot\tau$')
ax[0, 1].set_xlabel(r'$T_0 / T_{orb}$')
ax[0, 1].set_ylabel(r'$S$, см$^2$')
ax[0, 1].set_title(r'$S(T_0,\ \tau=$const$)$')
ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

# (3) S(T_0 = const, tau) - прямая с наклоном L/(2m)
ax[1, 0].plot(tau_values / T_orb, S_tau, label='числ.')
ax[1, 0].plot(tau_values / T_orb, sec * tau_values, 'r--', label=r'$L/(2m)\cdot\tau$')
ax[1, 0].set_xlabel(r'$\tau / T_{orb}$')
ax[1, 0].set_ylabel(r'$S$, см$^2$')
ax[1, 0].set_title(r'$S(T_0=$const$,\ \tau)$')
ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

# (4) Сохранение момента импульса L(t)
ax[1, 1].plot(t / T_orb, L)
ax[1, 1].set_xlabel(r'$t / T_{orb}$')
ax[1, 1].set_ylabel(r'$L/m$, см$^2$/с')
ax[1, 1].set_title('Сохранение момента импульса')
ax[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()