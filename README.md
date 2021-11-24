# Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy 
API: flask 
Данные: с archive.ics.uci.edu - https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Задача: предсказать, к какому классу относится банкнота - настоящая (1) или подделка (0). Бинарная классификация

Используемые признаки:

1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer) 


Преобразования признаков: StandardScaler()

Модель: Gradient Boosting Classifier

## Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/TatMillion/ML_in_business.git
$ git checkout course_project
$ docker build -t tatiana/project .
```
Запускаем контейнер
Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)

```
$ docker run -d -p 8180:8180 tatiana/project
```
## Переходим на localhost:8180
