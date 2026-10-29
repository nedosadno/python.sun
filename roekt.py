import math
import tkinter as tk
from tkinter import ttk
import numpy as np


f = 400e3              # Частота, Гц (400 кГц)
U1p = 300.0            # Амплитудное напряжение, В
I1p_total = 15.0       # Пиковый ток всей системы, А

# Параметры сердечника RM14 материал N41
Ae = 200e-6            # Эффективное сечение, м^2
le = 70e-3             # Средняя длина магнитной линии, м
Bsat = 0.39            # Индукция насыщения, Тл
mu_i = 2500            # Начальная магнитная проницаемость


mu0 = 4 * math.pi * 1e-7
rho_cu = 1.72e-8       # Удельное сопротивление меди, Ом*м

# Конструктивные параметры каркаса (заданы жестко)
lw = (150e-3)*2*np.pi            # Средняя длина витка, м
hw = 19e-3             # Высота намотки, м
d_eq = 0.1e-3          # Толщина изоляции между обмотками, м
J_max = 8e6            # Максимально допустимая плотность тока, А/м^2 (8 А/мм^2)

# Ограничения из ТЗ по индуктивности рассеяния
L_sys_min = 0.5e-6     # Нижняя граница, Гн
L_sys_max = 25e-6      # Верхняя граница, Гн

# Доступные варианты зазоров в сердечнике 
available_gaps = [0.0, 1.0e-3, 2.0e-3] 

# Доступные провода 
wires = {
    "МВ-4 (0.5 мм²)": {"S": 0.5e-6},
    "ЛЭПКО 49x0.10": {"S": 49 * math.pi * (0.1)**2},
    "ЛЭПКО 175x0.10": {"S": 175 * math.pi * (0.1e-3/2)**2}
}

def calculate_transformer():
    results = []
    
    # Считаем минимально возможное число витков по напряжению
    N1_min_voltage = math.ceil(U1p / (2 * math.pi * f * Bsat * Ae))
    print(f"Минимальное N1 по напряжению: {N1_min_voltage} витков")

    # Внешний цикл: перебираем количество параллельных трансформаторов
    for m in range(1, 5):
        I1p_unit = I1p_total / m               # Ток на один блок
        I1rms_unit = I1p_unit / math.sqrt(2)   # Действующий ток на блок
        
        # Коэффициент трансформации целый: от 1 до 10
        for k in range(1, 11):
            # Перебираем целое число витков вторички
            for N2 in range(1, 100):
                N1 = k * N2 # Первичка должна быть целой и кратной k
                
                if N1 < N1_min_voltage:
                    continue # Если витков мало, сердечник насытится
                    
                # Считаем индуктивность рассеяния одного блока
                L_unit = (mu0 * N1**2 * lw * d_eq) / hw
                
                # При параллельном соединении общая индуктивность делится на m
                L_sys = L_unit / m
                
                # Проверяем, попали ли в диапазон ТЗ (0.5 - 25 мкГн)
                if not (L_sys_min <= L_sys <= L_sys_max):
                    continue
                    
                # Считаем требуемый зазор, чтобы не было насыщения по току
                lg_req = (mu0 * N1 * I1p_unit) / Bsat - (le / mu_i)
                
                # Ищем минимальный подходящий стандартный зазор
                chosen_gap = None
                for gap in available_gaps:
                    if gap >= lg_req:
                        chosen_gap = gap
                        break
                
                if chosen_gap is None:
                    continue # Если нужен зазор больше 2 мм, этот вариант не подходит
                
                                # Проверяем, подходит ли провод по току (с учетом скин-эффекта)
                delta_s = math.sqrt(rho_cu / (math.pi * f * mu0)) # Глубина скин-слоя (0.104 мм)
                
                for wire_name, wire_data in wires.items():
                    S_wire = wire_data["S"]
                    
                    # Для обычного провода МВ-4 учитываем скин-эффект
                    if "МВ-4" in wire_name:
                        D_wire = math.sqrt(4 * S_wire / math.pi) # Диаметр провода (м)
                        S_eff = math.pi * D_wire * delta_s        # Эффективная площадь сечения (м^2)
                        if S_eff > S_wire: # Защита от дурака (если частота низкая)
                            S_eff = S_wire
                    else:
                        # Для литцендрата (жилы 0.1 мм < 2 * delta_s) скин-эффект отсутствует
                        S_eff = S_wire 
                        
                    J = I1rms_unit / S_eff # Плотность тока по эффективному сечению
                    
                    if J > J_max:
                        continue # Провод сгорит
                    
                    # Считаем сопротивление (для ВЧ правильнее считать с учетом R_ac, 
                    # но для проверки студенту достаточно R_dc по S_eff)
                    R_unit = (rho_cu * lw * N1) / S_eff
                    
                    if J > J_max:
                        continue # Провод сгорит
                    
                    # Считаем сопротивление
                    R_unit = (rho_cu * lw * N1) / S_wire
                    
                    # Если все проверки пройдены, сохраняем вариант
                    results.append({
                        'm': m,
                        'k': k,
                        'N1': N1,
                        'N2': N2,
                        'L_unit': L_unit * 1e6,
                        'L_sys': L_sys * 1e6,
                        'l_g': chosen_gap * 1e3,
                        'wire': wire_name,
                        'J': J / 1e6,
                        'R': R_unit
                    })
    return results

# 3. ОКНО С ТАБЛИЦЕЙ И ЛЕГЕНДОЙ
def show_table(data):
    root = tk.Tk()
    root.title("Таблица возможных конфигураций системы")
    root.geometry("1300x700")

    # --- Блок с легендой ---
    legend_frame = tk.Frame(root, bd=2, relief="groove", bg="#FFFAFA")
    legend_frame.pack(side="top", fill="x", padx=10, pady=10)
    
    tk.Label(legend_frame, text="Значения столбцов таблицы", 
             font=("Arial", 11, "bold"), bg="#e9e9e9").pack(anchor="w", padx=10, pady=5)
    
    legend_text = (
        "m — Количество одинаковых трансформаторов, включенных параллельно.\n"
        "k — Коэффициент трансформации (N1/N2), от 1 до 10.\n"
        "N1, N2 — Количество витков первичной и вторичной обмоток соответственно.\n"
        "L_системы — Итоговая индуктивность рассеяния ( от 0.5 до 25 мкГн).\n"
        "Зазор — Выбранный зазор в сердечнике, чтобы при токе 15А феррит не насыщался.\n"
        "Тип провода — Провод, который выдерживает ток без перегрева (плотность тока J не более 8 А/мм²).\n"
        "Плотн. тока — Показывает, насколько сильно нагружен провод.\n"
        "R обмотки — Активное сопротивление обмотки"
    )
    tk.Label(legend_frame, text=legend_text, justify="left", bg="#dbd2d2", font=("Arial", 10)).pack(anchor="w", padx=20, pady=(0, 10))


    table_frame = tk.Frame(root)
    table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    columns = ("m", "k", "N1", "N2", "L_sys", "l_g", "wire", "J", "R")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    
    headers = {
        "m": "Кол-во блоков (m)", "k": "Коэф. k", 
        "N1": "Витки N1", "N2": "Витки N2",
        "L_sys": "L_системы (мкГн)", 
        "l_g": "Зазор (мм)", "wire": "Тип провода", 
        "J": "Плотн. тока (А/мм²)", "R": "R обмотки (Ом)"
    }
    
    for col, header in headers.items():
        tree.heading(col, text=header)
        tree.column(col, anchor="center", width=120)

    # Сортируем, чтобы красиво выводилось (сначала 1 блок, потом 2 и т.д.)
    data.sort(key=lambda x: (x['m'], x['k'], x['N1']))

    # Заполняем строки таблицы
    for row in data:
        tree.insert("", tk.END, values=(
            row['m'], row['k'], row['N1'], row['N2'], 
            f"{row['L_sys']:.2f}", f"{row['l_g']:.1f}", 
            row['wire'], f"{row['J']:.2f}", f"{row['R']:.3f}"
        ))

    # Скроллбар, если строк будет много
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

# Точка входа в программу
if __name__ == "__main__":
    all_results = calculate_transformer()
    if not all_results:
        print("Ошибка: не найдено ни одного варианта, удовлетворяющего ТЗ.")
        print("Возможно, нужно изменить материал сердечника или увеличить допустимый ток.")
    else:
        show_table(all_results)
        print(len(all_results))