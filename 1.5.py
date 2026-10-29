import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open('data.txt', 'r')  # открываем файл для чтения
lines = f.readlines()  # читаем все строки в список
f.close() 

def parse(line):  # функция: из строки в список чисел
    clean = line.split('#')[0]  # отрезаем комментарий после #
    parts = clean.split()  # разбиваем по пробелам
    return [float(p) for p in parts]  # каждую часть в число

plane_pt   = parse(lines[0])  # точка на плоскости [x0,y0,z0]
plane_norm = parse(lines[1])  # вектор нормали [nx,ny,nz]
charge_line = parse(lines[2])  # строка заряда [Q,x,y,z]
Q_real = charge_line[0]  # величина заряда
x = charge_line[1]  # координата x
y = charge_line[2]  # координата y
z = charge_line[3]  # координата z

length = (plane_norm[0]**2 + plane_norm[1]**2 + plane_norm[2]**2) ** 0.5  # длина нормали
plane_norm[0] /= length  # нормируем x
plane_norm[1] /= length  # нормируем y
plane_norm[2] /= length  # нормируем z
dt =  (x - plane_norm[0])  +  (y - plane_norm[1]) + (z - plane_norm[2])
if dt < 0:
    plane_norm[0] *= -1  
    plane_norm[1] *= -1  
    plane_norm[2] *= -1 


eps0 = 8.854e-12  # электрическая постоянная
k = 1 / (4 * np.pi * eps0)  # коэффициент Кулона

#заряд-изображение
dx = x - plane_pt[0]  # вектор от точки плоскости до заряда (x)
dy = y - plane_pt[1]  # то же (y)
dz = z - plane_pt[2]  # то же (z)
d = dx*plane_norm[0] + dy*plane_norm[1] + dz*plane_norm[2]  # расстояние до плоскости

x_img = x - 2*d*plane_norm[0]  # зеркальное отражение (x)
y_img = y - 2*d*plane_norm[1]  # зеркальное отражение (y)
z_img = z - 2*d*plane_norm[2]  # зеркальное отражение (z)
Q_img = -Q_real  # заряд-изображение противоположен по знаку

print("Реальный заряд:", Q_real, "в точке", x, y, z)
print(f"Заряд-изображение: {Q_img:.1e}, в точке {x_img:.3f}, {y_img:.3f}, {z_img:.3f}")


def E_field(px, py, pz):  # поле в точке (px,py,pz)
    dx1, dy1, dz1 = px-x, py-y, pz-z  # вектор от реального заряда
    r1 = (dx1**2 + dy1**2 + dz1**2) ** 0.5  # расстояние
    Ex = k*Q_real*dx1 / r1**3  # поле от реального (x)
    Ey = k*Q_real*dy1 / r1**3  # поле от реального (y)
    Ez = k*Q_real*dz1 / r1**3  # поле от реального (z)
    dx2, dy2, dz2 = px-x_img, py-y_img, pz-z_img  # вектор от изображения
    r2 = (dx2**2 + dy2**2 + dz2**2) ** 0.5  # расстояние
    Ex += k*Q_img*dx2 / r2**3  # прибавляем поле от изображения (x)
    Ey += k*Q_img*dy2 / r2**3  # прибавляем (y)
    Ez += k*Q_img*dz2 / r2**3  # прибавляем (z)
    return Ex, Ey, Ez  # возвращаем вектор поля (суперпозиция)

#  локальные оси u, v на плоскости (переходим в СК наклоненной системы)
# Плоскость может быть наклонена в пространстве как угодно
# Чтобы интегрировать по ней, нам нужны две оси, лежащие внутри плоскости
# Ось X и ось Y могут быть не параллельны плоскости
# Поэтому строим два перпендикулярных вектора u и v, которые:
#   1) лежат в плоскости (перпендикулярны нормали n),
#   2) перпендикулярны друг другу
# Тогда любую точку плоскости можно задать как: r = r0 + u*u_axis + v*v_axis

ax_, ay, az = 1.0, 0.0, 0.0  # берём произвольный вектор a = (1,0,0)
if abs(plane_norm[0]) > 0.9:  # если нормаль почти параллельна оси X, вектор a=(1,0,0) почти параллелен n
    ax_, ay, az = 0.0, 1.0, 0.0  # тогда берём a = (0,1,0), иначе векторное произведение na даст ноль

# u_axis = n × a. Векторное произведение двух векторов даёт вектор, перпендикулярный обоим
# Значит u_axis перпендикулярен нормали, т е лежит в плоскости. Это наша первая локальная ось
ux = plane_norm[1]*az - plane_norm[2]*ay  
uy = plane_norm[2]*ax_ - plane_norm[0]*az 
uz = plane_norm[0]*ay - plane_norm[1]*ax_  
len_u = (ux**2 + uy**2 + uz**2) ** 0.5  # длина получившегося вектора 
ux, uy, uz = ux/len_u, uy/len_u, uz/len_u  #получаем единичный вектор u_axis

# v_axis = n × u_axis. результат перпендикулярен и нормали, и u_axis.
# Значит v_axis тоже лежит в плоскости и перпендикулярен u_axis. Это вторая локальная ось
# Вместе u_axis и v_axis образуют прямоугольную систему координат на плоскости
# В этой системе удобно задавать область интегрирования (квадрат u_min..u_max, v_min..v_max).
vx = plane_norm[1]*uz - plane_norm[2]*uy 
vy = plane_norm[2]*ux - plane_norm[0]*uz  
vz = plane_norm[0]*uy - plane_norm[1]*ux  # длина уже 1, т.к. n и u единичные и перпендикулярны

#интегрирование - тут мы задаем область на плоскости
u_min, u_max = -30.0, 30.0  # границы по u
v_min, v_max = -30.0, 30.0  # границы по v
N = 100  # размер сетки N×N

du = (u_max - u_min) / N  
dv = (v_max - v_min) / N 
dS = du * dv 
Q_induced = 0.0  # суммарный заряд

for i in range(N):  # цикл по u
    u = u_min + (i + 0.5) * du  # центр i-й клетки (смещаемся в зав-ти от номера клетки)
    for j in range(N):  # цикл по v
        v = v_min + (j + 0.5) * dv  # центр j-й клетки
        px = plane_pt[0] + u*ux + v*vx  # 2D-координата (u,v) в 3D-точка на плоскости
        py = plane_pt[1] + u*uy + v*vy  # (y)
        pz = plane_pt[2] + u*uz + v*vz  # (z)
        Ex, Ey, Ez = E_field(px, py, pz)  # поле в точке
        En = Ex*plane_norm[0] + Ey*plane_norm[1] + Ez*plane_norm[2]  # проекция на нормаль - считаем поток через верхнюю крышку

        sigma = eps0 * En  # плотность заряда (теорема Гаусса)
        Q_induced += sigma * dS  # накапливаем заряд

print(f"Индуцированный заряд: {Q_induced:.3e} Кл")

# визуализация
fig = plt.figure(figsize=(10, 8))  # окно графика 10×8
ax = fig.add_subplot(111, projection='3d')  # 3D-оси

for i in range(10):  # 10 линий сетки плоскости
    v_fix = v_min + (v_max - v_min) * i / 9  # фиксированное v
    lx, ly, lz = [], [], []  # координаты линии
    for j in range(10):  # 10 точек на линии
        u_var = u_min + (u_max - u_min) * j / 9  # текущее u
        lx.append(plane_pt[0] + u_var*ux + v_fix*vx)  # x точки
        ly.append(plane_pt[1] + u_var*uy + v_fix*vy)  # y точки
        lz.append(plane_pt[2] + u_var*uz + v_fix*vz)  # z точки
    ax.plot(lx, ly, lz, color='gray')  # рисуем линию

    u_fix = u_min + (u_max - u_min) * i / 9  # фиксированное u
    lx, ly, lz = [], [], []  # снова списки
    for j in range(10):  # 10 точек
        v_var = v_min + (v_max - v_min) * j / 9  # текущее v
        lx.append(plane_pt[0] + u_fix*ux + v_var*vx)  # x
        ly.append(plane_pt[1] + u_fix*uy + v_var*vy)  # y
        lz.append(plane_pt[2] + u_fix*uz + v_var*vz)  # z
    ax.plot(lx, ly, lz, color='gray')  # рисуем

ax.scatter(x, y, z, color='pink', s=150, label='Реальный заряд')  # точка реального заряда
ax.scatter(x_img, y_img, z_img, color='darkblue', s=150, label='Заряд-изображение')  # точка изображения



ax.set_xlabel('Ось X')  # подпись X
ax.set_ylabel('Ось Y')  # подпись Y
ax.set_zlabel('Ось Z')  # подпись Z
ax.set_title('3D: плоскость, заряды, поле E')  # заголовок
ax.legend()  # легенда
plt.show()  # показать график