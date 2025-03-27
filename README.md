# street_tree

Построить DL-модель для классификации состояния дерева (Good/Fair/Poor) по данным из NY 2015 Street Tree Census.

## Описание проекта

Данный проект включает разработку и обучение модели глубокого обучения для классификации состояния деревьев на основе данных о деревьях, собранных в рамках переписи деревьев Нью-Йорка в 2015 году. Модель предсказывает одно из трех состояний дерева:
- **Good** (Хорошее)
- **Fair** (Среднее)
- **Poor** (Плохое)

## API для предсказания здоровья деревьев

### Эндпоинт: `/predict_health/`

API предоставляет возможность отправить данные о деревьях в формате JSON и получить предсказания состояния деревьев.

- **Метод:** `POST`
- **URL:** `http://127.0.0.1:8000/predict_health/`
- **Описание:** Эндпоинт для получения предсказаний здоровья деревьев на основе JSON данных.

### Пример тела запроса (Request Body):
```json
[
  {
    "tree_id": 536325, 
    "block_id": 415591, 
    "tree_dbh": 56, 
    "curb_loc": 0, 
    "spc_latin": "Fraxinus", 
    "steward": "None", 
    "guards": "Unsure", 
    "sidewalk": 0, 
    "user_type": "NYC Parks Staff", 
    "problems": "NoProblem", 
    "root_stone": 0, 
    "root_grate": 0, 
    "root_other": 0, 
    "trunk_wire": 0, 
    "trnk_light": 0, 
    "trnk_other": 0, 
    "brch_light": 0, 
    "brch_shoe": 0, 
    "brch_other": 0, 
    "postcode": 10306, 
    "borough": "Staten Island", 
    "cncldist": 50, 
    "st_assem": 62, 
    "st_senate": 24, 
    "nta": "SI24", 
    "boro_ct": 5027900,             
    "latitude": 40.571808, 
    "longitude": -74.14425163
  }
]

Ответ сервиса в виде 
Ответ сервиса в виде:
```json
[
  {
    "tree_id": 536325,
    "predictions": "Good",
    "probably": [0.1, 0.3, 0.6],
    "class_labels": "Good",
    "name_model": "TreeHealthModel"
  }
]
Поля ответа:
tree_id: Уникальный идентификатор дерева.
predictions: Предсказанное состояние дерева (Good, Fair, Poor).
probably: Вероятности для каждого класса в формате [Poor, Fair, Good].
class_labels: Название класса, соответствующее предсказанию.
name_model: Название модели, использованной для предсказания.
Пример интерпретации:
Для дерева с tree_id = 536325 модель предсказала состояние Good с вероятностями:
Poor: 10%
Fair: 30%
Good: 60%


