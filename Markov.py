import random
import re
from collections import defaultdict


def read_text(path):
    # Пробуем открыть файл в разных кодировках
    for enc in ("utf-8", "cp1251", "koi8-r"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise SystemExit("Не удалось определить кодировку файла")


# Определения символов, которые считаются частью слова
letters = "A-Za-zА-Яа-яЁё"       # все буквы: латиница + кириллица (включая Ё/ё)
inner_symbols = "-'"             # дефис и апостроф — допустимы только внутри слова
punctuation = ".,!?;:—–-"        # знаки препинания — считаем отдельными токенами

split_pattern = re.compile(
    rf"[{letters}]+(?:[{inner_symbols}][{letters}]+)*"   # слово: буква, потом (дефис|апостроф + буквы)*
    rf"|[{punctuation}]")                                # или одиночный знак препинания


def split_into_tokens(text):
    # findall возвращает список слов и знаков препинания по порядку
    return split_pattern.findall(text)

def build_chain(tokens, order):
    # order — сколько предыдущих токенов используем для предсказания
    # chain: ключ (кортеж из order)
    # чем чаще токен шёл после контекста,тем больше его копий в списке 
    # starts = ключи, которые в исходном тексте стояли в начале предложения
    chain = defaultdict(list)
    starts = []

    # Идём окном по токенам. Останавливаемся там, где есть ещё хотя бы один токен для предсказания
    for i in range(len(tokens) - order):
        # Ключ — order подряд идущих токенов
        # Именно tuple, тк списки нельзя использовать как ключи словаря
        key = tuple(tokens[i:i + order])

        # Токен, который идёт сразу после контекста — его модель предсказывает
        next_token = tokens[i + order]

        # Добавляем это продолжение в список для данного ключа
        # Если такое продолжение уже было — оно появится второй раз
        chain[key].append(next_token)

        # Проверяем, начинается ли здесь новое предложение:
        #   - либо это самое начало текста (i == 0),
        #   - либо предыдущий токен — точка, ! или ?
        # Такие контексты сохраняем отдельно — с них будем стартовать
        if i == 0 or tokens[i - 1] in ".!?":
            starts.append(key)

    return chain, starts

def generate(chain, starts, order, length):
    # Страховка: если стартовых контекстов не нашлось, берём все ключи
    if not starts:
        starts = list(chain.keys())

    # Выбираем случайный стартовый контекст, текст начнётся не с середины фразы.
    key = random.choice(starts)
    # Результат изначально — слова стартового контекста
    result = list(key)
    while len(result) < length:
        # Смотрим, какие продолжения были у текущего контекста
        next_options = chain.get(key)
        # Если у этого контекста в исходнике не было продолжений, то
        # Начинаем новое предложение со случайного старта
        if not next_options:
            key = random.choice(starts)
            result.extend(key)
            continue

        # Так как в списке слова повторяются пропорционально частоте,
        # частые продолжения выпадают чаще = марковская вероятность
        next_token = random.choice(next_options)
        result.append(next_token)

        # Сдвигаем окно
        key = tuple(result[-order:])

    return result[:length]

def format_tokens(tokens):
    # Пробел ставим перед словом, но не перед знаком препинания
    out = []

    # Множество для быстрой проверки является ли токен знаком пунктуации
    punct = set(punctuation)

    for t in tokens:
        if out and t not in punct:
            out.append(" ")
        out.append(t)

    # Склеиваем и убираем возможные пробелы по краям.
    return "".join(out).strip()

def ask_int(prompt, allowed=None):
    try:
        value = int(input(prompt))
    except ValueError:
        print("Введите целое число.")
        return ask_int(prompt, allowed)   # повторяем

    if allowed and value not in allowed:
        print(f"Допустимо: {allowed}")
        return ask_int(prompt, allowed)

    return value

def main():
    # Спрашиваем имя файла. .strip() убирает случайные пробелы по краям
    filename = input("Имя файла: ").strip()

    try:
        text = read_text(filename)
    except FileNotFoundError:
        print("Файл не найден")
        return

    # Разбиваем текст на токены (слова и знаки препинания)
    tokens = split_into_tokens(text)

    if not tokens:
        print("Файл пуст или не содержит текста.")
        return

    print(f"Токенов в тексте: {len(tokens)}")

    # length — сколько токенов хотим получить на выходе
    length = ask_int("Сколько слов сгенерировать: ")

    # chain_size — размер связки: 2 или 3 слова
    # allowed=(2, 3) не даст ввести что-то другое
    chain_size = ask_int("Размер связки (2 или 3): ", allowed=(2, 3))

    # Размер связки — это сколько слов участвует в предсказании
    # Одно из них предсказывается, остальные образуют контекст
    # Значит длина ключа = chain_size - 1
    order = chain_size - 1

    # Строим модель
    chain, starts = build_chain(tokens, order)

    # Показываем, сколько уникальных контекстов получилось 
    print(f"Уникальных контекстов: {len(chain)}")
 
    # Генерируем последовательность токенов заданной длины
    result = generate(chain, starts, order, length)

    # Превращаем список токенов в читаемую строку
    print(format_tokens(result))

main()