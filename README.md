### <ins>Соревнования (competitions):</ins>

- **<ins>[Всероссийский чемпионат "Цифровой прорыв 2022"](https://github.com/mikhail-rozov/Digital_Breakthrough_2022)</ins>**  
Индивидуальное соревнование - задача регрессии, где основным признаком был набор текстов.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Занял призовое 8 место из 114 участников.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Решал данную задачу ещё до углублённого погружения в NLP, поэтому сейчас этот этап делал бы &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;по-другому. Старое решение не меняю, т.к. нужно оставлять воспроизводимость результата. Исходя &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;из особенностей датасета и целевой метрики, поставил задачу построить очень стабильную &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;модель, которая на приватном лидерборде не "провалится". В итоге это себя оправдало, метрика &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;вообще не снизилась, тогда как многие оппоненты "провалились" вниз. Подробный ход своих &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;мыслей изложил в презентации к решению.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<ins>Стек:</ins> Word2Vec, Sklearn, nltk, pymorphy2, CatBoost, LightGBM, Pandas.  

### <ins>Classic machine learning (ML):</ins>

- **<ins>[Модель прогнозирования спроса на услуги в телекоме](https://github.com/mikhail-rozov/Megafon_project)</ins>**  
По обезличенным данным телеком-провайдера разработана модель,
предсказывающая, согласится ли абонент на предложенную ему услугу. Также создан
механизм индивидуального предложения услуг каждому абоненту.  
<ins>Стек:</ins> Sklearn, Pandas, CatBoost, XGBoost.  
    <br>
- **<ins>[Модель предсказания цены на недвижимость](https://github.com/mikhail-rozov/Data_science_modules_1/tree/master/Course_project)</ins>**  
Частное соревнование на kaggle.com. Модель регрессии. Предсказывает стоимость квартиры в Москве по различным признакам.  
<ins>Стек:</ins> NumPy, Pandas, Sklearn, Matplotlib, Seaborn.  
    <br>
- **<ins>[Модель кредитного скоринга](https://github.com/mikhail-rozov/-Data_science_modules_2/tree/master/Course_project)</ins>**  
Частное соревнование на kaggle.com. Модель бинарной классификации. Предсказывает факт невыполнения заёмщиком кредитных обязательств.  
<ins>Стек:</ins> NumPy, Pandas, CatBoost, Matplotlib, Seaborn.  
    <br>
- **<ins>[API для доступа к ML-модели](https://github.com/mikhail-rozov/ML_in_business/tree/master/Project)</ins>**  
Создание API-сервиса для коммуникации с ML-моделью (по схеме клиент-сервер). Пользователь вводит значения признаков, а сервер возвращает предсказания модели. Есть возможность разворачивания сервиса из docker-контейнера.  
<ins>Стек:</ins> Flask, requests, Docker, NumPy, Pandas, Sklearn.

### <ins>Парсинг (скрапинг) данных:</ins>

- **<ins>[Интернет-парсер](https://github.com/mikhail-rozov/Data_collection_methods/tree/master/Lesson_8)</ins>**  
Приложение для автоматического сбора информации о подписчиках и подписках заданных
пользователей Instagram с дальнейшей загрузкой данных в БД MongoDB.  
<ins>Стек:</ins> Python, Scrapy, MongoDB.

### <ins>Recommendation systems (RecSys):</ins>

- **<ins>[Двухэтапная система рекомендаций для ритейла](https://github.com/mikhail-rozov/Recommendation_systems/tree/master/Lesson_8)</ins>**  
Система на основе коллаборативной фильтрации, где для подбора кандидатов используется модель ALS, а для их ранжирования - модель CatBoostRanker.  
<ins>Стек:</ins> NumPy, Pandas, Scipy, implicit, CatBoost.

### <ins>SQL:</ins>

- **<ins>[База данных MySQL](https://github.com/mikhail-rozov/MySQL_project)</ins>**  
Cоздание и заполнение базы данных для лобби компьютерной игры.
Отработка запросов, представлений, триггеров и процедур.  
<ins>Стек:</ins> SQL.

### <ins>Natural language processing (NLP):</ins>

- **<ins>[Чат-бот в телеграм](https://github.com/mikhail-rozov/nlp-introduction/tree/master/Course_project)</ins>**  
Чат-бот в Telegram на 3 интента, в основе которого лежат языковые модели,
основанные на нейросетях (одна модель была мной дообучена на диалогах). Может сообщать
текущую погоду, отвечать на вопросы из базы и поддерживать беседу. В зависимости от
сообщения пользователя происходит автоматический выбор соответствующего интента
(модели).  
<ins>Стек:</ins> Python, transformers, PyTorch, spaCy, annoy, requests.  
    <br>
- **<ins>[Модель генерации текста](https://github.com/mikhail-rozov/nlp-introduction/blob/master/Lesson_14/Lesson_14_task_1.ipynb)</ins>**  
Обученная с нуля нейросеть GPT-2 (small) на цитатах bash.im. Модель генерирует забавные
цитаты и диалоги.  
<ins>Стек:</ins> PyTorch, transformers.

### <ins>Computer vision (CV):</ins>

- **<ins>[Приложение с распознаванием жестов](https://github.com/mikhail-rozov/GB-Pytorch/tree/master/Course_project)</ins>**  
Приложение, которое считывает изображение с веб-камеры и производит детектирование лица. При 
обнаружении лица в кадре производится детектирование жеста пользователя. В
зависимости от жеста, на экран выводится соответствующее сообщение.  
<ins>Стек:</ins> OpenCV, mediapipe, Sklearn, Pandas.  
    <br>
- **<ins>[Модель сегментации изображений](https://github.com/mikhail-rozov/GB-Pytorch/blob/master/Lesson_5/Lesson_5_tasks.ipynb)</ins>**  
Модель сегментации губ на основе UNet, где в качестве энкодера использовалась предобученная модель ResNet, а декодер обучался с нуля.  
<ins>Стек:</ins> PyTorch, Pandas.

### <ins>Generative adversarial networks (GANs):</ins>

- **<ins>[Модель, генерирующая точки, лежащие на заданном графике](https://github.com/mikhail-rozov/GB-Pytorch/blob/master/Lesson_8/Lesson_8_task.ipynb)</ins>**  
Создание и обучение модели на приципах GAN, которая генерирует из случайных данных (шума) точки, лежащие на графике заданной функции.  
<ins>Стек:</ins> Python, PyTorch, Matplotlib.
