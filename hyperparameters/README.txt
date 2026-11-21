Лабораторная работа №3 — Titanic / Decision Tree

Открыть lab03.ipynb: код, пояснения, выполненные ячейки и графики.
report.pdf: краткий отчёт по всем пунктам ТЗ.
lab03.py: тот же эксперимент как обычный Python-скрипт.
data/train.csv: исходные данные из предоставленного репозитория.
verify.py: независимая проверка метрик, исходного файла и разбиения.
results/: полные результаты обоих поисков, метрики по folds, графики,
разбиение по Id, финальные предсказания и зафиксированный до test выбор.

Рекомендуется Python 3.13 (проверенный запуск: Python 3.13).
Запуск из папки lab03:
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python lab03.py
  python verify.py

Для ноутбука: открыть lab03.ipynb в Jupyter/VS Code, выбрать это окружение,
рабочая папка — lab03, затем Restart Kernel / Run All.
Время поиска зависит от компьютера. Seed фиксирован, но версии sklearn
могут менять разбиение и реализацию алгоритма. Используйте requirements.txt.

Источник: https://github.com/batsuperflash/ml_course_mai
Commit: 6f7704934e19192d2e79281fdfd54961a0b16c29
Предыдущая работа: classification/lab02_titanic.ipynb, ячейки 7, 9, 12, 18.
Данные: classification/data/train.csv.
Baseline: DecisionTreeClassifier(random_state=42), остальные defaults.

В лабораторной №2 test уже использовался при сравнении моделей.
В лабораторной №3 test не участвует в выборе параметров. Тем не менее
это повторная контрольная оценка, а не новый независимый внешний тест.
Ответов к classification/data/test.csv нет, его F1 не вычисляется.
