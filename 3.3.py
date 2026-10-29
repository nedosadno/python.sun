import numpy as np
import matplotlib.pyplot as plt

# Фиксируем зерно генератора, чтобы результаты были воспроизводимы
# (при каждом запуске одна и та же псевдослучайная последовательность).
np.random.seed(0)

dh      = 1.0                 # максимальный шаг одной частицы
n_max   = 1000                # максимальное число шагов
N_list  = [100, 1000, 10000]  # три разных числа частиц для сравнения

# Теоретические значения коэффициентов прямой y = a*n + b,
# которой должна подчиняться зависимость <x^2>(n):
#     <X_n^2> = (dh^2/3) * n + 0
#   =>  a2_theor = dh^2/3,   b_theor = 0.
a1_theor = dh*(41/1600)
a2_theor = dh**2 / 3.0
b_theor = 0.0

def simulate(N, n_max, dh):
    # Возвращает массивы <x>(n) и <x^2>(n) для n = 1..n_max.

    # Как считаем:
    #   1) Генерируем матрицу шагов (n_max, N): steps[k, i] - смещение i-й частицы на (k+1)-м шаге. Каждый элемент U[-dh, dh].
    #   2) cumsum вдоль оси времени -> traj[k-1, i] - координата i-й частицы после k шагов.
    #   3) Усредняем по частицам (axis=1) -> <x>_n и <x^2>_n.

    #   < <x>_n > = 0,
    #   < <x^2>_n > = n * dh^2 / 3.

    steps = np.random.uniform(-dh, dh*1.05, size=(n_max, N))
    traj  = np.cumsum(steps, axis=0)      # траектории всех частиц
    mean_x  = traj.mean(axis=1)           # <x> для каждого n
    mean_x2 = ((traj - mean_x[:, np.newaxis])**2).mean(axis=1)      # <x^2> для каждого n

    return mean_x, mean_x2


def fit_coeffs(y):
    # Для каждого n = 2..len(y) делаем МНК прямую по первым n точкам.
    # имитируем ситуацию "у нас пока только n шагов", смотрим,
    # каким получился бы наклон a и сдвиг b, и следим, как ониA
    # сходятся к своим теоретическим значениям с ростом n.

    # Возвращает:
    #   a_arr, b_arr - массивы длины len(y)-1 (соответствуют n = 2..len(y)).

    n_vals = np.arange(1, len(y) + 1)
    a_arr = np.zeros(len(y) - 1)
    b_arr = np.zeros(len(y) - 1)
    for k in range(2, len(y) + 1):
        n_use = n_vals[:k]      # n = 1..k
        y_use = y[:k]           # первые k значений y
        # np.polyfit степени 1 - это и есть МНК для прямой.
        a, b = np.polyfit(n_use, y_use, 1)
        a_arr[k-2] = a
        b_arr[k-2] = b
    return a_arr, b_arr


def stabilization_step(coeff_arr, tol):
    # Наименьшее n, начиная с которого коэффициент отличается от
    # своего финального значения меньше, чем на tol 

    #   final = coeff_arr[-1]          предельное значение
    #   rel   = |coeff_arr - final| / |final|  относительное отклонение
    #   Ищем первое i, начиная с которого ВСЕ последующие rel < tol.

    final = coeff_arr[-1]

    if abs(final) < 1e-15:
        for i in range(len(coeff_arr)):
            if np.all(np.abs(coeff_arr[i:] - final) < tol):
                return i + 2      # +2, потому что массив начинается с n=2
        return None
    else:
        rel = np.abs(coeff_arr - final) / abs(final)
        for i in range(len(rel)):
            if np.all(rel[i:] < tol):
                return i + 2
        return None


#   строка 0: <x>(n)         - должно быть - 0 (шум)
#   строка 1: <x^2>(n)       - должно быть - (dh^2/3)*n (прямая)
#   строка 2: коэффициенты a(n), b(n) МНК-аппроксимации для <x^2>
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for col, N in enumerate(N_list):
    # моделирование 
    mean_x, mean_x2 = simulate(N, n_max, dh)
    n_vals = np.arange(1, n_max + 1)

    # МНК-коэффициенты для <x> и <x^2> 
    a_x,  b_x  = fit_coeffs(mean_x)
    a_x2, b_x2 = fit_coeffs(mean_x2)
    n_fit = np.arange(2, n_max + 1)   # коэффициенты определены с n=2

    # Теория: <X_n> = 0. На графике должно быть случайное
    # колебание около нуля с амплитудой - sigma_n / sqrt(N).
    ax = axes[0, col]
    ax.plot(n_vals, mean_x, lw=1, label=r'$\langle x\rangle$')

    ax.plot(n_vals[1:], a_x * n_vals[1:] + b_x, 'r-', lw=1)
    ax.set_title(f'N = {N}')
    ax.set_xlabel('n'); ax.set_ylabel(r'$\langle x^2\rangle$')

    ax.set_title(f'N = {N}')
    ax.set_xlabel('n'); ax.set_ylabel(r'$\langle x\rangle$')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Теория: <X_n^2> = (dh^2/3)*n. Прямая с нулевым сдвигом.
    ax = axes[1, col]
    ax.plot(n_vals, mean_x2, lw=1, label=r'$\langle x^2\rangle$')
    ax.plot(n_vals, a2_theor * n_vals + b_theor, 'r-', lw=1,
            label=f'теория: {a2_theor:.3f} n')
    ax.set_title(f'N = {N}')
    ax.set_xlabel('n'); ax.set_ylabel(r'$\langle x^2\rangle$')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Смотрим, как наклон a и сдвиг b МНК-прямой для <x^2>
    # сходятся к теоретическим a = dh^2/3 и b = 0.
    ax = axes[2, col]
    ax.plot(n_fit, a_x2, lw=1, label='a (наклон)')
    ax.plot(n_fit, b_x2, lw=1, label='b (сдвиг)')
    ax.axhline(a2_theor, color='r', ls='-', lw=1,
               label=f'теория a = {a2_theor:.3f}')
    ax.axhline(b_theor, color='g', ls='-', lw=1, label='теория b = 0')
    ax.set_title(f'Коэффициенты МНК для $\\langle x^2\\rangle$, N = {N}')
    ax.set_xlabel('n'); ax.set_ylabel('коэффициент')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Для трёх уровней точности: 1%, 0.1%, 0.01%.
    # Смотрим, начиная с какого n коэффициент перестаёт
    # отклоняться от финального значения больше, чем на tol.
    print(f"\nN = {N}")
    for tol in [0.01, 0.001, 0.0001]:
        n_a = stabilization_step(a_x2, tol)
        n_b = stabilization_step(b_x2, tol)
        print(f"  точность {tol*100:>5.2f}%:  "
              f"a стабилизируется с n = {n_a},  "
              f"b стабилизируется с n = {n_b}")

plt.tight_layout()
plt.show()