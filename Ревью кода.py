import datetime as dt


class Record:
    """Класс для хранения информации о записях."""

    def __init__(self, amount, comment, date=None):
        self.amount = amount
        self.comment = comment
        if date is None:
            self.date = dt.datetime.now().date()
        else:
            self.date = dt.datetime.strptime(date, '%d.%m.%Y').date()


class Calculator:
    """Базовый класс для калькуляторов."""

    def __init__(self, limit):
        self.limit = limit
        self.records = []

    def add_record(self, record):
        """Добавляет новую запись в список."""
        self.records.append(record)

    def get_today_stats(self):
        """Возвращает сумму значений за сегодняшний день."""
        today = dt.datetime.now().date()
        return sum(
            rec.amount for rec in self.records if rec.date == today
        )

    def get_week_stats(self):
        """Возвращает сумму значений за последние 7 дней."""
        today = dt.datetime.now().date()
        week_ago = today - dt.timedelta(days=7)
        return sum(
            rec.amount
            for rec in self.records
            if week_ago <= rec.date <= today
        )


class CaloriesCalculator(Calculator):
    """Калькулятор для подсчета калорий."""

    def get_calories_remained(self):
        """Возвращает сообщение об остатке калорий на сегодня."""
        calories_remaining = self.limit - self.get_today_stats()
        if calories_remaining > 0:
            return (
                'Сегодня можно съесть что-нибудь ещё, но с общей '
                f'калорийностью не более {calories_remaining} кКал'
            )
        return 'Хватит есть!'


class CashCalculator(Calculator):
    """Калькулятор для подсчета денег."""

    USD_RATE = 60.0
    EURO_RATE = 70.0
    RUB_RATE = 1.0

    def get_today_cash_remained(self, currency):
        """Возвращает сообщение об остатке денег на сегодня."""
        cash_remaining = self.limit - self.get_today_stats()
        rates = {
            'usd': (self.USD_RATE, 'USD'),
            'eur': (self.EURO_RATE, 'Euro'),
            'rub': (self.RUB_RATE, 'руб'),
        }
        if currency not in rates:
            return f'Неизвестная валюта {currency}'

        rate, currency_name = rates[currency]
        cash_remaining /= rate
        rounded_cash = round(cash_remaining, 2)

        if cash_remaining > 0:
            return f'На сегодня осталось {rounded_cash} {currency_name}'
        if cash_remaining == 0:
            return 'Денег нет, держись'
        
        debt = abs(rounded_cash)
        return f'Денег нет, держись: твой долг - {debt} {currency_name}'

    def get_week_stats(self):
        """Возвращает сумму затрат за неделю."""
        return super().get_week_stats()