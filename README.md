# Введение в машинное обучение

Репозиторий практических и лабораторных работ по курсу «Введение в машинное обучение».
**Студент:** Музаффар Мустафаев (Muzaffar Mustafayev)

---

## 📁 Структура репозитория

```text
machine-learning-tsu/
├── README.md
├── .gitignore
├── Titanic.csv                 # Датасет Titanic
├── vgsales.csv                 # Датасет Video Game Sales
│
├── Laboratory_1/               # Практическая работа №1 (95 заданий)
│   ├── main.py                 # Скрипт автоматического запуска всех заданий
│   ├── 01_Python_Basics.ipynb  # Задания 1–36: Основы языка Python
│   ├── 02_NumPy.ipynb          # Задания 1–29: Библиотека NumPy
│   ├── 03_Pandas_Titanic.ipynb # Задания 1–15: Анализ набора данных Titanic
│   ├── 04_Pandas_Video_Games.ipynb # Задания 1–15: Анализ Video Game Sales
│   ├── Laboratory_1_complete.ipynb # Полный единый блокнот
│   ├── outputs/                # 39 файлов графиков, таблиц и отчетов
│   ├── Muzaffar_Mustafayev/    # Папка сдачи для преподавателя
│   └── Muzaffar_Mustafayev_Laboratory_1.zip # Архив для сдачи в Moodle
│
└── Laboratory_2/               # Лабораторная работа №2 (Классификация ФКГ и сенсоров)
    ├── main.py                 # Скрипт автоматического запуска всех этапов
    ├── 01_EDA_and_Preprocessing.ipynb   # Раздел 1.1: Исследование и предобработка
    ├── 02_Machine_Learning_Models.ipynb # Раздел 1.2: Модели Scikit-learn и CatBoost
    ├── Laboratory_2_complete.ipynb      # Полный единый блокнот
    ├── outputs/                # Графики сигналов, CV, метрик и ROC-AUC
    ├── Muzaffar_Mustafayev/    # Папка сдачи для преподавателя
    └── Muzaffar_Mustafayev_Laboratory_2.zip # Архив для сдачи в Moodle
```

---

## 🚀 Лабораторная работа №1 (Практика №1)

### Открытие в Google Colab

| Раздел | Ссылка на Colab |
| :--- | :--- |
| **01. Python: Базовые навыки (1–36)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_1/01_Python_Basics.ipynb) |
| **02. NumPy (1–29)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_1/02_NumPy.ipynb) |
| **03. Pandas: Titanic (1–15)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_1/03_Pandas_Titanic.ipynb) |
| **04. Pandas: Video Game Sales (1–15)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_1/04_Pandas_Video_Games.ipynb) |
| **Полный блокнот (Laboratory 1 Complete)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_1/Laboratory_1_complete.ipynb) |

*Для автоматического выполнения всех заданий:*
```bash
cd Laboratory_1
python main.py
```

---

## 🚀 Лабораторная работа №2 (Классификация ФКГ и модуля позиционирования)

### Открытие в Google Colab

| Раздел | Ссылка на Colab |
| :--- | :--- |
| **01. EDA и предобработка данных (1.1.1–1.1.13)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_2/01_EDA_and_Preprocessing.ipynb) |
| **02. Модели машинного обучения (1.2.1–1.2.10)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_2/02_Machine_Learning_Models.ipynb) |
| **Полный блокнот (Laboratory 2 Complete)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muzaffarbekmustafayev/machine-learning-tsu/blob/main/Laboratory_2/Laboratory_2_complete.ipynb) |

*Для автоматического выполнения всех этапов:*
```bash
cd Laboratory_2
pip install catboost openpyxl
python main.py
```
