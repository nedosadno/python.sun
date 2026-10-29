import numpy as np                      # подключаем numpy, чтобы работать с массивами и математикой
import matplotlib.pyplot as plt         # matplotlib нужен для графиков
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


N = input("Введи N для chargesN.dat: ").strip()   # спрашиваем номер файла
charges = np.atleast_2d(np.loadtxt("charges" + N + ".dat"))   # столбцы: Q x y z
# np.loadtxt — читает текстовый файл и возвращает массив numpy (числа по строкам файла)
# np.atleast_2d — делает массив двумерным: если в файле одна строка, без этого получился бы
#                  одномерный массив, а нам нужны "строки-заряды" и "столбцы-координаты"

# границы параллелепипеда (из условия задачи)
xmin, xmax = 2.0, 3.0
ymin, ymax = 1.0, 4.0
zmin, zmax = 0.5, 1.0

# параметры сферы вводим с клавиатуры
xc, yc, zc, R = map(float, input("xs ys zs R: ").split())

eps0 = 8.854e-12                        # электрическая постоянная
k = 1 / (4 * np.pi * eps0)              # коэффициент в законе Кулона
# np.pi — число пи с машинной точностью

# поле в точке
def E_point(x, y, z):
    Ex = Ey = Ez = 0.0
    for Q, xq, yq, zq in charges:
        dx, dy, dz = x - xq, y - yq, z - zq
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < 1e-18:
            continue
        r = np.sqrt(r2)                 # np.sqrt — поэлементный квадратный корень
        Ex += k * Q * dx / r**3
        Ey += k * Q * dy / r**3
        Ez += k * Q * dz / r**3
    return Ex, Ey, Ez

def flux_box(n):
    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    dz = (zmax - zmin) / n
    flux = 0.0

    # Грани x = xmin, x = xmax
    # np.linspace(a, b, n) — массив из n равномерно распределённых чисел от a до b
    for y in np.linspace(ymin+dy/2, ymax-dy/2, n):
        for z in np.linspace(zmin+dz/2, zmax-dz/2, n):
            Ex,_,_ = E_point(xmin, y, z); flux -= Ex*dy*dz
            Ex,_,_ = E_point(xmax, y, z); flux += Ex*dy*dz

    # Грани y = ymin, y = ymax
    for x in np.linspace(xmin+dx/2, xmax-dx/2, n):
        for z in np.linspace(zmin+dz/2, zmax-dz/2, n):
            _,Ey,_ = E_point(x, ymin, z); flux -= Ey*dx*dz
            _,Ey,_ = E_point(x, ymax, z); flux += Ey*dx*dz

    # Грани z = zmin, z = zmax
    for x in np.linspace(xmin+dx/2, xmax-dx/2, n):
        for y in np.linspace(ymin+dy/2, ymax-dy/2, n):
            _,_,Ez = E_point(x, y, zmin); flux -= Ez*dx*dy
            _,_,Ez = E_point(x, y, zmax); flux += Ez*dx*dy

    return flux


#Теорема Гаусса
Q_box = 0.0
inside_box_flags = []
for Q, xq, yq, zq in charges:
    inside = (xmin <= xq <= xmax and ymin <= yq <= ymax and zmin <= zq <= zmax)
    inside_box_flags.append(inside)
    if inside:
        Q_box += Q
Phi_box_theor = Q_box / eps0

Q_sph = 0.0
inside_sph_flags = []
for Q, xq, yq, zq in charges:
    d = np.sqrt((xq-xc)**2 + (yq-yc)**2 + (zq-zc)**2)   # np.sqrt — корень из числа
    inside = d <= R
    inside_sph_flags.append(inside)
    if inside:
        Q_sph += Q
Phi_sph_theor = Q_sph / eps0

print("Теорема Гаусса")
print(f"Параллелепипед: Q_внутри = {Q_box:.3e} Кл, Φ_теор = {Phi_box_theor:.4e} В·м")
print(f"Сфера:          Q_внутри = {Q_sph:.3e} Кл, Φ_теор = {Phi_sph_theor:.4e} В·м\n")


# поток через параллелепипед
def flux_box(n):
    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    dz = (zmax - zmin) / n
    flux = 0.0

    # Грани x = xmin, x = xmax
    # np.linspace(a, b, n) — массив из n равномерно распределённых чисел от a до b
    for y in np.linspace(ymin+dy/2, ymax-dy/2, n):
        for z in np.linspace(zmin+dz/2, zmax-dz/2, n):
            Ex,_,_ = E_point(xmin, y, z); flux -= Ex*dy*dz
            Ex,_,_ = E_point(xmax, y, z); flux += Ex*dy*dz

    # Грани y = ymin, y = ymax
    for x in np.linspace(xmin+dx/2, xmax-dx/2, n):
        for z in np.linspace(zmin+dz/2, zmax-dz/2, n):
            _,Ey,_ = E_point(x, ymin, z); flux -= Ey*dx*dz
            _,Ey,_ = E_point(x, ymax, z); flux += Ey*dx*dz

    # Грани z = zmin, z = zmax
    for x in np.linspace(xmin+dx/2, xmax-dx/2, n):
        for y in np.linspace(ymin+dy/2, ymax-dy/2, n):
            _,_,Ez = E_point(x, y, zmin); flux -= Ez*dx*dy
            _,_,Ez = E_point(x, y, zmax); flux += Ez*dx*dy

    return flux


# поток через сферу
def flux_sphere(n):
    dth = np.pi / n
    dph = 2 * np.pi / n
    flux = 0.0
    for i in range(n):
        th = (i + 0.5) * dth
        st, ct = np.sin(th), np.cos(th)   # np.sin, np.cos — тригонометрические функции
        for j in range(n):
            ph = (j + 0.5) * dph
            x = xc + R * st * np.cos(ph)
            y = yc + R * st * np.sin(ph)
            z = zc + R * ct
            nx, ny, nz = st * np.cos(ph), st * np.sin(ph), ct
            Ex, Ey, Ez = E_point(x, y, z)
            dS = R**2 * st * dth * dph
            flux += (Ex*nx + Ey*ny + Ez*nz) * dS
    return flux


# сходимость
ns = np.array([4, 8, 16, 32, 64, 128])    # np.array — создаёт массив numpy из списка
err_box, err_sph = [], []
Phi_box_num, Phi_sph_num = [], []

print("поток через параллелепипед")
for n in ns:
    Phi = flux_box(n)
    Phi_box_num.append(Phi)
    e = abs(Phi - Phi_box_theor)
    err_box.append(e)
    print(f"n={n:4d}  Φ_числ = {Phi:+.4e}  |ошибка| = {e:.3e}")

print("\nпоток через сферу")
for n in ns:
    Phi = flux_sphere(n)
    Phi_sph_num.append(Phi)
    e = abs(Phi - Phi_sph_theor)
    err_sph.append(e)
    print(f"n={n:4d}  Φ_числ = {Phi:+.4e}  |ошибка| = {e:.3e}")


# график ошибки
plt.figure(figsize=(9, 5))
plt.plot(ns, err_box, 'o-', color='steelblue', label='параллелепипед')
plt.plot(ns, err_sph, 's-', color='crimson',   label='сфера')
plt.xlabel('n — число разбиений сетки')
plt.ylabel('|Φ_числ − Φ_теор|,  В·м')
plt.title('Зависимость ошибки от плотности сетки')
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# заряды в поверхности
fig2 = plt.figure(figsize=(14, 6))

ax1 = fig2.add_subplot(121, projection='3d')
def draw_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, color='gray', alpha=0.10):
    # np.array — создаём массив из списка списков (координаты углов параллелепипеда)
    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]]
    pc = Poly3DCollection([corners[f] for f in faces],
                          facecolor=color, edgecolor='k',
                          linewidths=0.8, alpha=alpha)
    ax.add_collection3d(pc)

draw_box(ax1, xmin, xmax, ymin, ymax, zmin, zmax)

# Заряды: красный — внутри, серый — снаружи
for (Q, xq, yq, zq), ins in zip(charges, inside_box_flags):
    color = ('red' if Q > 0 else 'blue') if ins else 'gray'
    marker = 'o' if ins else 'x'
    ax1.scatter(xq, yq, zq, c=color, marker=marker, s=90, depthshade=False)

# Подписи
for i, (Q, xq, yq, zq) in enumerate(charges):
    ax1.text(xq, yq, zq, f"q{i+1}", color='black')

ax1.set(xlabel='x', ylabel='y', zlabel='z',
        title='Параллелепипед и заряды\n',
        xlim=(-6, 6), ylim=(-4, 7), zlim=(-11, 3))
ax1.set_box_aspect((1, 1, 2))

# ---------- Сфера + заряды ----------
ax2 = fig2.add_subplot(122, projection='3d')

# np.linspace — массив равномерных значений. Здесь: 40 значений u от 0 до 2π и 20 значений v от 0 до π
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
# np.outer(a, b) — внешнее произведение: матрица, где M[i,j] = a[i]*b[j]
# np.ones_like(u) — массив из единиц той же формы, что и u
X_s = xc + R * np.outer(np.cos(u), np.sin(v))
Y_s = yc + R * np.outer(np.sin(u), np.sin(v))
Z_s = zc + R * np.outer(np.ones_like(u), np.cos(v))
ax2.plot_wireframe(X_s, Y_s, Z_s, color='gray', lw=0.4, alpha=0.5)

for (Q, xq, yq, zq), ins in zip(charges, inside_sph_flags):
    color = ('red' if Q > 0 else 'blue') if ins else 'gray'
    marker = 'o' if ins else 'x'
    ax2.scatter(xq, yq, zq, c=color, marker=marker, s=90, depthshade=False)
    ax2.text(xq, yq, zq, f"q{i+1}")

lim = R * 1.6
ax2.set(xlabel='x', ylabel='y', zlabel='z',
        title=f'Сфера R={R}, центр=({xc},{yc},{zc}) и заряды\n',
        xlim=(xc-lim, xc+lim),
        ylim=(yc-lim, yc+lim),
        zlim=(zc-lim, zc+lim))
ax2.set_box_aspect((1, 1, 1))

plt.tight_layout()
plt.show()