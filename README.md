

---
# Face, Emotion, Gender, Age, Race Detection with DeepFace and Streamlit

Детектор эмоций, пола, возраста и расы на видео на основе библиотеки DeepFace и с веб-интрефейсом на Streamlit 

---
## 🚀 Функционал
- Детекция лиц и распознавание эмоций, пола, возраста и расы с отрисовкой результата на видео
- Отображение прогресс-бара детекции видео
- Регулировка порога уверенности модели в том что на изображении есть лицо
- Сохранение результатов детекции видео в файл `csv` с последующей отрисовкой графиков для анализа видео
- Отрисовка всех распознанных ранее результатов 
- Возможность скачать результаты детекции (видео)


---
## 🛠 Стек

- **3.8** <= [python](https://www.python.org/)  <= **3.11**
- [deepface](https://github.com/serengil/deepface) для детекции лиц и распознавания эмоций, пола, возраста и расы
- [Streamlit](https://github.com/streamlit/streamlit) для написания веб-интерфейса
- [ffmpeg](https://ffmpeg.org/) для конвертации видео в отображаемый в браузере формат
- [altair](hhttps://docs.streamlit.io/develop/api-reference/charts/st.altair_chart) для отрисовки графиков результатов детекции видео

Работоспособность приложения проверялась на WSL Ubuntu 22.04 (python 3.10)  
[Документация](https://www.tensorflow.org/install/pip) с командами установки TensorFlow для Windows


---
## 🐍 Установка и запуск на Linux через Python

**1) Клонирование репозитория**  

```
git clone https://github.com/sergey21000/face-detector-deepface.git
cd face-detector-deepface
```

**2) Создание и активация виртуального окружения (опционально)**

```
python3 -m venv env
source env/bin/activate
```

**3) Установка зависимостей**  

- *С поддержкой CPU*
  ```
  pip install -r requirements-cpu.txt
  ```

- *С поддержкой CUDA*
  ```
  pip install -r requirements.txt
  ```

**4) Запуск сервера Streamlit**  
```
streamlit run main_page.py
```

После запуска сервера перейти в браузере по адресу http://localhost:8501/  
