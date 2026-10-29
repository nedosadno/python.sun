import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-8              # гравитационная постоянная, см^3/(г·с^2)
M_sun = 1.989e33          # масса Солнца, г
AU = 1.496e13             # астрономическая единица, см


def integrate_kepler(r0, v0, mu, dt, N):
    r = np.array(r0, dtype=float)   # текущий радиус-вектор
    v = np.array(v0, dtype=float)   # текущая скорость
    R = np.zeros((N + 1, 2))        # массив для траектории
    t = np.arange(N + 1) * dt       # массив времён
    R[0] = r                        # записываем начальное положение

    for k in range(N):
        rn = np.linalg.norm(r)                              # |r|
        v_half = v + 0.5 * (-mu * r / rn**3) * dt          
        r_new = r + v_half * dt                           
        rn_new = np.linalg.norm(r_new)                      # |r_new|
        v_new = v_half + 0.5 * (-mu * r_new / rn_new**3) * dt  
        r, v = r_new, v_new
        R[k + 1] = r

    return R, t


def curve2order(x, y):
    # определяем тип кривой второго порядка и её параметры.
    

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # нормализация координат для устойчивости МНК (убираем разницу масштабов)
    s = max(np.max(np.abs(x)), np.max(np.abs(y)))
    x_norm = x / s
    y_norm = y / s

    M = np.column_stack([x_norm**2, x_norm*y_norm, y_norm**2, x_norm, y_norm])
    rhs = -np.ones_like(x_norm)
    coeffs, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    A, B, C, D, E = coeffs
    F = 1.0

    # тип кривой по дискриминанту
    delta = B**2 - 4*A*C
    if delta < 0:
        type_ = 1     # эллипс
    else:
        type_ = -1    # гипербола

    # координаты центра — решение линейной системы
    #     2 A x0 + B y0 + D = 0
    #     B x0 + 2 C y0 + E = 0
    det = 4*A*C - B**2
    if abs(det) > 1e-15:
        x0 = (B*E - 2*C*D) / det
        y0 = (B*D - 2*A*E) / det
    else:
        x0, y0 = 0.0, 0.0

    # свободный член K после переноса начала координат в центр
    K = -(A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F)

    # собственные значения и векторы квадратичной формы
    Mmat = np.array([[A, B/2], [B/2, C]])
    eigvals, eigvecs = np.linalg.eigh(Mmat)   # eigvals отсортированы по возрастанию
    lam1, lam2 = eigvals[0], eigvals[1]
    v1, v2 = eigvecs[:, 0], eigvecs[:, 1]

    # полуоси: s² = K / λ. Знак λ относительно K определяет
    # действительную/мнимую ось
    s1 = np.sqrt(abs(K / lam1))
    s2 = np.sqrt(abs(K / lam2))

    if type_ == 1:
        # эллипс: большая полуось = max(s1, s2)
        if s1 >= s2:
            a, b = s1, s2
            v_major = v1
        else:
            a, b = s2, s1
            v_major = v2
    else:
        # гипербола: действительная ось — вдоль того λ, где K/λ > 0
        if K / lam1 > 0:
            a, b = s1, s2
            v_major = v1
        else:
            a, b = s2, s1
            v_major = v2

    # угол большой (действительной) оси с осью x.
    # ось не имеет направления — фиксируем представление с y ≥ 0
    if v_major[1] < 0:
        v_major = -v_major
    cos_beta = v_major[0]

    # возвращаем в исходных единицах, умножая на масштаб
    return a*s, b*s, x0*s, y0*s, cos_beta, type_


#  Тест 1. Связанная орбита — должна быть эллипсом
M = M_sun                          # масса Солнца
mu = G * M                         # гравитационный параметр
e = 0.5                            # эксцентриситет связанной орбиты
r_p = 0.5 * AU                     # перигелий

v_p =( np.sqrt(mu * (1 + e) / r_p) *0.7 ) # скорость в перигелии (v ⟂ r)
a_th = r_p / (1 - e)               # теоретическая большая полуось
T_th = 2 * np.pi * np.sqrt(a_th**3 / mu)  # теоретический период

# шаг и число шагов — примерно 3 периода, чтобы эллипс был полностью пройден
dt = T_th / 5000
N = 3 * 5000
R, t = integrate_kepler([r_p, 0], [0.5*v_p, v_p], mu, dt, N)

a_num, b_num, x0_num, y0_num, cos_beta_num, type_num = curve2order(R[:, 0], R[:, 1])

# сравнение с теорией
a_th = r_p / (1 - e)               # теоретическая большая полуось
b_th = a_th * np.sqrt(1 - e**2)    # теоретическая малая полуось

print("Тест 1. Связанная орбита (должен быть эллипс):")
print(f"  тип кривой            = {type_num}  (ожидается +1)")
print(f"  a числ / a теор       = {a_num/AU:.4f} / {a_th/AU:.4f} а.е.")
print(f"  b числ / b теор       = {b_num/AU:.4f} / {b_th/AU:.4f} а.е.")
print(f"  x0 числ / x0 теор     = {x0_num/AU:.4f} / {-a_th*e/AU:.4f} а.е.")
print(f"  y0 числ               = {y0_num/AU:.4e} а.е.  (ожидается 0)")
print(f"  cos(beta) числ / теор = {cos_beta_num:.4f} / 1.0000")


#  Тест 2. Уходящая орбита — должна быть гиперболой
v_esc = np.sqrt(2 * mu / r_p)      # вторая космическая на этом расстоянии
v_p_hyp = 1.4 * v_esc              # заведомо больше v_esc — уходящая траектория

dt2 = T_th / 5000
N2 = 2000
R2, t2 = integrate_kepler([r_p, 0], [0, v_p_hyp], mu, dt2, N2)

a_num2, b_num2, x0_num2, y0_num2, cos_beta_num2, type_num2 = curve2order(R2[:, 0], R2[:, 1])

print("\nТест 2. Уходящая орбита (должен быть гипербола):")
print(f"  тип кривой        = {type_num2}  (ожидается -1)")
print(f"  a числ            = {a_num2/AU:.4f} а.е.")
print(f"  b числ            = {b_num2/AU:.4f} а.е.")
print(f"  x0 числ           = {x0_num2/AU:.4f} а.е.")
print(f"  y0 числ           = {y0_num2/AU:.4e} а.е.")
print(f"  cos(beta) числ    = {cos_beta_num2:.4f}")


#  Графики: траектории с наложенной подогнанной кривой
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# параметрическое построение подогнанной кривой по её параметрам
def draw_conic(ax, a, b, x0, y0, cos_beta, type_, color='r'):
    beta = np.arccos(cos_beta)
    if type_ == 1:
        theta = np.linspace(0, 2*np.pi, 600)
        X, Y = a*np.cos(theta), b*np.sin(theta)
    else:
        t_par = np.linspace(-2.5, 2.5, 600)
        X, Y = a*np.cosh(t_par), b*np.sinh(t_par)
    # поворот на угол beta
    xr = X*np.cos(beta) - Y*np.sin(beta) + x0
    yr = X*np.sin(beta) + Y*np.cos(beta) + y0
    ax.plot(xr/AU, yr/AU, color, lw=1.2, ls='--',
            label='подогнанная кривая')

# левый график: эллипс
ax[0].plot(R[:, 0]/AU, R[:, 1]/AU, lw=1.0, label='численная траектория')
ax[0].plot(0, 0, 'yo', ms=10)                # Солнце в фокусе
draw_conic(ax[0], a_num, b_num, x0_num, y0_num, cos_beta_num, type_num)
ax[0].set_aspect('equal'); ax[0].grid(alpha=0.3)
ax[0].set_xlabel('x, а.е.'); ax[0].set_ylabel('y, а.е.')
ax[0].set_title(f'Эллипс: a = {a_num/AU:.3f}, b = {b_num/AU:.3f} а.е.')
ax[0].legend()

# правый график: гипербола
ax[1].plot(R2[:, 0]/AU, R2[:, 1]/AU, lw=1.0, label='численная траектория')
ax[1].plot(0, 0, 'yo', ms=10)
draw_conic(ax[1], a_num2, b_num2, x0_num2, y0_num2, cos_beta_num2, type_num2)
ax[1].set_aspect('equal'); ax[1].grid(alpha=0.3)
ax[1].set_xlabel('x, а.е.'); ax[1].set_ylabel('y, а.е.')
ax[1].set_title(f'Гипербола: a = {a_num2/AU:.3f}, b = {b_num2/AU:.3f} а.е.')
ax[1].legend()

plt.tight_layout()
plt.show()