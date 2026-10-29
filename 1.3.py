import numpy as np #для вычислений
import matplotlib.pyplot as plt #для графиков

p = 1.0   # дипольный момент: заряды ±1 на расстоянии d = 1 вдоль z 

# Увеличиваем фигуру, чтобы уместить 4 графика (2x2)
fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(221, projection='3d') # Первый график (3D силовые линии)
theta = np.linspace(0.02, np.pi, 400)
phi_list = np.linspace(0, 2*np.pi, 12, endpoint=False)

for C in np.linspace(0.6, 5.0, 10):
    r = C * np.sin(theta)**2
    # Диполь вдоль оси X: угол θ отсчитывается от оси X
    x = r * np.cos(theta)          # координата вдоль диполя
    rho = r * np.sin(theta)        # расстояние от оси X
    for phi in phi_list:
        y = rho * np.cos(phi)
        z = rho * np.sin(phi)
        ax1.plot(x, y, z, color='steelblue', lw=0.6, alpha=0.6)

# Заряды: + справа (x = +0.5), − слева (x = -0.5)
ax1.scatter([+0.5, -0.5], [0, 0], [0, 0],
            c=['red', 'blue'], s=80, depthshade=False)

ax1.set(xlabel='x, см', ylabel='y, см', zlabel='z, см',
        title='3D Силовые линии диполя',
        xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))
ax1.set_box_aspect((1, 1, 1))     # чтобы куб был кубом, а не растянутым прямоугольником


ax2 = fig.add_subplot(222, projection='3d') # Второй график (3D эквипотенциалы)

# th — 120 значений полярного угла, ph — 120 значений азимутального
th = np.linspace(0.01, np.pi - 0.01, 120)
ph = np.linspace(0, 2*np.pi, 120)
# np.meshgrid строит двумерные сетки из двух одномерных массивов:
# TH[i,j] = th[i], PH[i,j] = ph[j] — удобно для построения поверхностей
TH, PH = np.meshgrid(th, ph)

for V in [0.4, 0.2, 0.1, -0.4, -0.2, -0.1]:   # разные значения потенциала
    # уравнение эквипотенциалы диполя: V = p*cosθ / r²  ⇒  r² = p*cosθ / V
    r2 = p * np.cos(TH) / V
    # np.where(условие, a, b) — поэлементный выбор: где условие True, берём a, иначе b
    # здесь: где r² > 0 оставляем как есть, а где нет — заменяем на NaN
    # (такие точки matplotlib просто не будет рисовать)
    r2 = np.where(r2 > 0, r2, np.nan)
    r = np.sqrt(r2)                   # поэлементный корень
    # перевод в декартовы координаты для каждой точки поверхности
    Xs = r * np.sin(TH) * np.cos(PH)
    Ys = r * np.sin(TH) * np.sin(PH)
    Zs = r * np.cos(TH)
    color = 'crimson' if V > 0 else 'royalblue'   # плюс — красный, минус — синий
    # plot_surface рисует поверхность по трём матрицам X, Y, Z
    ax2.plot_surface(Xs, Ys, Zs, color=color, alpha=0.35, linewidth=0)

ax2.scatter([0, 0], [0, 0], [+0.5, -0.5],
            c=['red', 'blue'], s=80, depthshade=False)
ax2.set(xlabel='x, см', ylabel='y, см', zlabel='z, см',
        title='3D эквипотенциалы',
        xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))
ax2.set_box_aspect((1, 1, 1))

ax3 = fig.add_subplot(223)

# Сетка для поля
x = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)
X, Z = np.meshgrid(x, z)

# Заряды: +1 справа (0.5, 0), -1 слева (-0.5, 0)
q1, x1, z1 = +1, 0.5, 1
q2, x2, z2 = -1, -0.5, 0

def field(q, x0, z0, X, Z):
    dx = X - x0
    dz = Z - z0
    r2 = dx**2 + dz**2
    r3 = r2**(3/2)
    r3 = np.where(r3 == 0, 1e-9, r3)  # защита от деления на ноль
    Ex = q * dx / r3
    Ez = q * dz / r3
    return Ex, Ez

Ex1, Ez1 = field(q1, x1, z1, X, Z)
Ex2, Ez2 = field(q2, x2, z2, X, Z)
Ex = Ex1 + Ex2
Ez = Ez1 + Ez2

# Силовые линии
ax3.streamplot(X, Z, Ex, Ez, color='steelblue', linewidth=1,
               arrowsize=1, density=1.5)

# Заряды
ax3.scatter([-0.5, 0.5], [0, 1], c=['blue', 'red'], s=120, zorder=5)


ax3.set(xlabel='x, см', ylabel='z, см',
        title='Силовые линии диполя (y=0)')
ax3.set_xlim(-5, 5)
ax3.set_ylim(-5, 5)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')




# Проекция эквипотенциалей
ax4 = fig.add_subplot(224)

th_2d = np.linspace(0.01, np.pi - 0.01, 400)
for V in [0.4, 0.2, 0.1, -0.4, -0.2, -0.1]:
    r2 = p * np.cos(th_2d) / V
    r2 = np.where(r2 > 0, r2, np.nan)   # где r² отрицательно — ставим NaN
    r = np.sqrt(r2)
    x = r * np.sin(th_2d)
    z = r * np.cos(th_2d)
    
    color = 'crimson' if V > 0 else 'royalblue'
    ax4.plot(x, z, color=color, lw=1.5, alpha=0.8)   # phi = 0
    ax4.plot(-x, z, color=color, lw=1.5, alpha=0.8)  # phi = pi

# Рисуем заряды
ax4.scatter([0, 0], [+0.5, -0.5], c=['red', 'blue'], s=80, zorder=5)

ax4.set(xlabel='x, см', ylabel='z, см', 
        title='Проекция эквипотенциалей (y=0)')
ax4.set_xlim(-5, 5)
ax4.set_ylim(-5, 5)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

plt.tight_layout()                    # автоматически подгоняет отступы между графиками
plt.show()

print("Сравнение с теорией (p = 1):\n")
print("На оси z (θ = 0):  E_z = +2p / r³")
for r in [1, 2, 3, 4, 5]:
    # f-строка: {выражение:.3e} значит вывести число в формате 3 знака после точки в экспоненциальной записи
    print(f"   r = {r} см  →  E_z = {2*p/r**3:.3e}")

print("\nВ плоскости z = 0 (θ = 90°):  |E| = p / r³")
for r in [1, 2, 3, 4, 5]:
    print(f"   r = {r} см  →  |E| = {p/r**3:.3e}")