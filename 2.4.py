import numpy as np
import matplotlib.pyplot as plt

G     = 6.674e-8          # гравитационная постоянная, см^3/(г·с^2)
M_sun = 1.989e33          # масса Солнца, г
AU    = 1.496e13          # астрономическая единица, см


def integrate_kepler(r0, v0, mu, dt, N):
    # метод Верле (kick-drift-kick), 2-й порядок точности.
    # симплектический: сохраняет момент импульса L = [r x v] точно,
    # без систематического дрейфа. Обычный Эйлер даёт линейно растущую
    # ошибку L и разрушает орбиту за несколько оборотов.
    r = np.array(r0, dtype=float)   # текущий радиус-вектор
    v = np.array(v0, dtype=float)   # текущая скорость
    R = np.zeros((N + 1, 2))        # массив для хранения всей траектории
    t = np.arange(N + 1) * dt       # массив времён
    R[0] = r                        # записываем начальное положение

    for k in range(N):
        # половина сдвига скорости по старой силе.
        # ускорение a = -mu*r/|r|^3 параллельно r, поэтому [r x a] = 0
        # и момент импульса сохраняется точно.
        rn     = np.linalg.norm(r)                      # модуль радиус-вектора
        v_half = v + 0.5 * (-mu * r / rn**3) * dt       # полшага по скорости

        # полный шаг координаты со «средней» скоростью v_half
        r_new  = r + v_half * dt

        # половина сдвига скорости по новой силе.
        # симметрия двух kick'ов — источник 2-го порядка точности
        rn_new = np.linalg.norm(r_new)                  # модуль нового r
        v_new  = v_half + 0.5 * (-mu * r_new / rn_new**3) * dt

        r, v = r_new, v_new             # обновляем состояние
        R[k + 1] = r                    # записываем в историю

    return R, t


def find_extrema(y):
    # индексы локальных максимумов и минимумов массива y.
    # локальный максимум — точка, где y[k-1] < y[k] > y[k+1];
    # локальный минимум — наоборот. Для кеплеровой орбиты все
    # локальные минимумы r(t) совпадают (перигелий — ближайшая
    # к Солнцу точка орбиты), все максимумы тоже (афелий — самая
    # дальняя точка). Если бы орбита прецессировала, значения
    # экстремумов медленно менялись бы — это был бы сигнал, что
    # поле не чисто ньютоновское.
    imax = [k for k in range(1, len(y) - 1) if y[k] > y[k-1] and y[k] > y[k+1]]
    imin = [k for k in range(1, len(y) - 1) if y[k] < y[k-1] and y[k] < y[k+1]]
    return np.array(imax), np.array(imin)


def orbit_params(r_p, e, M, N_periods=5, steps_per_period=2000):
    # r_p — перигелийное расстояние (задаёт масштаб орбиты)
    # e   — эксцентриситет (задаёт вытянутость)
    # M   — масса центрального тела

    mu = G * M                # гравитационный параметр mu = G*M

    # скорость в перигелии из интеграла энергии. В перигелии v ⟂ r,
    # поэтому вся скорость тангенциальная. v_p = sqrt(mu*(1+e)/r_p)
    v_p = np.sqrt(mu * (1 + e) / r_p)

    # аналитические ожидания для сверки с численным результатом:
    # большая полуось из r_p = a(1-e), период из 3-го закона Кеплера
    a_th = r_p / (1 - e)
    T_th = 2 * np.pi * np.sqrt(a_th**3 / mu)

    # шаг привязан к теоретическому периоду: ровно steps_per_period шагов
    dt = T_th / steps_per_period

    # интегрируем несколько периодов, чтобы получить несколько экстремумов
    # для усреднения
    N = N_periods * steps_per_period
    R, t = integrate_kepler([r_p, 0], [0, v_p], mu, dt, N)

    r = np.linalg.norm(R, axis=1)     # модуль радиус-вектора вдоль траектории

    imax, imin = find_extrema(r)      # локальные экстремумы r(t)

    # усредняем по всем найденным экстремумам: истинный экстремум может
    # попасть между узлами сетки, усреднение снижает влияние дискретизации
    r_max = np.mean(r[imax])                      # афелий
    r_min = np.mean(r[imin])                      # перигелий

    # большая полуось из определения a = (r_min + r_max) / 2
    a_num = (r_max + r_min) / 2

    # период — среднее расстояние по времени между соседними максимумами r(t)
    T_num = np.mean(np.diff(t[imax]))

    return a_num, T_num, R, t, imax, imin


#  Часть 1. Проверка третьего закона Кеплера: T(a)
#  при фиксированной массе M варьируем размер орбиты r_p
#  ожидание: T ~ a^(3/2), т.е. наклон log T от log a равен 1.5
M = M_sun
e = 0.5                                       # фиксированный эксцентриситет
r_p_values = np.linspace(0.3, 1.0, 8) * AU    # 8 разных размеров орбит

a_list, T_list = [], []
for r_p in r_p_values:
    a_num, T_num, *_ = orbit_params(r_p, e, M)   # находим a и T для орбиты
    a_list.append(a_num)
    T_list.append(T_num)
    print(f"r_p = {r_p/AU:.2f} а.е.  a = {a_num/AU:.3f} а.е.  T = {T_num:.3e} с")

a_arr = np.array(a_list)
T_arr = np.array(T_list)

# аппроксимация в двойных логарифмических координатах.
# если T = C * a^p, то log T = log C + p * log a — прямая с наклоном p.
# polyfit — МНК, intercept — численное значение log C
slope, intercept = np.polyfit(np.log(a_arr), np.log(T_arr), 1)
print(f"\nНаклон log T vs log a = {slope:.4f}  (ожидается 3/2 = 1.5)")


fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

# левый график: T(a) в двойных лог-координатах
ax[0].loglog(a_arr / AU, T_arr, 'o-', label='числ.')
# опорная прямая через первую точку с теоретическим наклоном 1.5
ax[0].loglog(a_arr / AU,
             T_arr[0] * (a_arr / a_arr[0])**1.5,
             'r--', label=r'$\propto a^{3/2}$')
ax[0].set_xlabel(r'$a$, а.е.')
ax[0].set_ylabel(r'$T$, с')
ax[0].set_title(f'Третий закон Кеплера\nнаклон = {slope:.3f}')
ax[0].grid(alpha=0.3, which='both')
ax[0].legend()


# Демонстрация: одна орбита и точки её экстремумов.
# берём орбиту среднего размера и отмечаем на ней найденные
# перигелии (зелёные) и афелии (красные)
R_demo, t_demo, imax_demo, imin_demo = orbit_params(0.5*AU, 0.5, M)[2:]

ax[1].plot(R_demo[:, 0] / AU, R_demo[:, 1] / AU, lw=0.8)
ax[1].plot(R_demo[imax_demo, 0] / AU, R_demo[imax_demo, 1] / AU, 'ro',
           label='максимумы $r$')
ax[1].plot(R_demo[imin_demo, 0] / AU, R_demo[imin_demo, 1] / AU, 'go',
           label='минимумы $r$')
ax[1].plot(0, 0, 'yo', ms=10)                 # Солнце в фокусе
ax[1].set_aspect('equal')
ax[1].set_xlabel('x, а.е.')
ax[1].set_ylabel('y, а.е.')
ax[1].set_title('Точки экстремумов $r(t)$')
ax[1].grid(alpha=0.3)
ax[1].legend()


#  Часть 2. Зависимость T^2/a^3 от массы звезды
#  фиксируем орбиту (a = const), меняем массу M
#  ожидание: T^2/a^3 = 4*pi^2/(G*M) ~ 1/M, наклон в лог-лог = -1
a_target = 1.0 * AU                            # большая полуось 1 а.е.
e2 = 0.3                                       # эксцентриситет орбиты
r_p2 = a_target * (1 - e2)                     # соответствующее r_p

M_values = np.linspace(0.5, 3.0, 8) * M_sun    # от половины до трёх масс Солнца

ratio = []                                     # сюда складываем T^2/a^3
for M_i in M_values:
    a_num, T_num, *_ = orbit_params(r_p2, e2, M_i)
    ratio.append(T_num**2 / a_num**3)
ratio = np.array(ratio)

# теоретическая кривая T^2/a^3 = 4*pi^2/(G*M)
theor = 4 * np.pi**2 / (G * M_values)

# наклон в лог-лог должен быть -1, если зависимость 1/M
slope2, intercept2 = np.polyfit(np.log(M_values / M_sun), np.log(ratio), 1)
print(f"Наклон log(T^2/a^3) vs log M = {slope2:.4f}  (ожидается -1)")

ax[2].loglog(M_values / M_sun, ratio, 'o-', label='числ.')
ax[2].loglog(M_values / M_sun, theor, 'r--', label=r'$4\pi^2/(GM)$')
ax[2].set_xlabel(r'$M / M_\odot$')
ax[2].set_ylabel(r'$T^2 / a^3$, с$^2$/см$^3$')
ax[2].set_title(f'Зависимость $T^2/a^3$ от $M$\nнаклон = {slope2:.3f}')
ax[2].grid(alpha=0.3, which='both')
ax[2].legend()

plt.tight_layout()
plt.show()