import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
from collections import Counter
from ffmpeg import FFmpeg  # Для конвертации видео с использованием ffmpeg

# ===================== Настройка моделей DeepFace и TensorFlow =====================
MODELS_DIR = Path('models')
os.environ['DEEPFACE_HOME'] = str(MODELS_DIR)

import tensorflow as tf
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as ex:
        print(ex)

def get_face_matrics(face_result):
    """
    Формирует словарь метрик для данного лица.
    
    Параметры:
      face_result (dict): Словарь, полученный из анализа лица библиотекой DeepFace.
      
    Возвращает:
      dict: Словарь с ключами 'age', 'gender', 'race', 'emotion' и координатами региона ('x', 'y', 'w', 'h').
    """
    metrics = {
        'age': face_result.get('age'),
        'gender': face_result.get('dominant_gender'),
        'race': face_result.get('dominant_race'),
        'emotion': face_result.get('dominant_emotion'),
        'x': face_result.get('region').get('x'),
        'y': face_result.get('region').get('y'),
        'w': face_result.get('region').get('w'),
        'h': face_result.get('region').get('h')
    }
    return metrics

def match_face(face_img, db_path):
    """
    Ищет совпадения для переданного изображения лица в базе изображений.
    
    Параметры:
      face_img (numpy.ndarray): Изображение лица в виде numpy массива.
      db_path (str): Путь к папке с базой изображений лиц.
      
    Возвращает:
      str или None: Идентификатор найденного лица или None, если совпадений нет.
    """
    try:
        df = DeepFace.find(
            img_path=face_img,
            db_path=db_path,
            model_name="Facenet",
            enforce_detection=False,
            silent=True
        )
        if not df[0].empty:
            return df[0].iloc[0]['identity']
    except Exception as e:
        print(f"Ошибка при поиске совпадения для лица: {e}")
    return None

class FaceMetrics:
    """
    Класс для хранения и обновления метрик для уникального лица.
    Позволяет аккумулировать историю распознаваний и получать усреднённые значения метрик.
    """
    def __init__(self, identifier):
        """
        Инициализация объекта.
        
        Параметры:
          identifier (str): Уникальный идентификатор лица (например, путь к изображению).
        """
        self.id = identifier
        self.metrics_history = []  # список с историей распознавания

    def update(self, metrics):
        """
        Добавляет новые метрики в историю.
        
        Параметры:
          metrics (dict): Словарь с результатами анализа.
        """
        self.metrics_history.append(metrics)

    def get_dominant_gender(self):
        """
        Определяет доминирующий пол на основе истории.
        
        Возвращает:
          str: Наиболее частое значение ('Man' или 'Woman').
        """
        genders_list = [m['gender'] for m in self.metrics_history]
        gender_count = Counter(genders_list)
        return "Man" if gender_count.get("Man", 0) > gender_count.get("Woman", 0) else "Woman"

    def get_dominant_race(self):
        """
        Определяет доминирующую расу по количеству вхождений.
        
        Возвращает:
          str: Название расы или 'Unknown', если данных нет.
        """
        races_list = [m['race'] for m in self.metrics_history]
        race_count = Counter(races_list)
        if not race_count:
            return "Unknown"
        max_count = max(race_count.values())
        dominant_races = [race for race, count in race_count.items() if count == max_count]
        return dominant_races[0]

    def get_dominant_age(self):
        """
        Рассчитывает средний возраст на основе истории.
        
        Возвращает:
          int: Усреднённый возраст.
        """
        total_age = sum(m['age'] for m in self.metrics_history)
        avg_age = total_age / len(self.metrics_history)
        return int(avg_age)

    def get_emotion(self):
        """
        Возвращает последнюю распознанную эмоцию.
        
        Возвращает:
          str: Эмоция из последней записи.
        """
        return self.metrics_history[-1].get('emotion')

    def get_average_metrics(self):
        """
        Формирует словарь усреднённых метрик для данного лица.
        
        Возвращает:
          dict: Словарь с ключами 'face_id', 'age', 'gender', 'race', 'emotion'.
        """
        return {
            'face_id': self.id,
            'age': self.get_dominant_age(),
            'gender': self.get_dominant_gender(),
            'race': self.get_dominant_race(),
            'emotion': self.get_emotion()
        }

    def count_emotions(self):
        """
        Подсчитывает количество каждого типа эмоций.
        
        Возвращает:
          dict: Словарь с подсчитанными значениями для каждого типа эмоций.
        """
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_counts = {emotion: 0 for emotion in emotions}
        for record in self.metrics_history:
            emotion = record.get('emotion', '').lower()
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
        return emotion_counts

def process_video_one_cell(video_path, faces_dir, output_video_path, face_conf_threshold=0.7, align=False, progress_callback=None, csv_output_path="video_results.csv"):
    """
    Обрабатывает видео: анализирует каждый кадр, выполняет аннотацию, сохраняет обработанное видео
    и записывает результаты распознавания лиц в CSV.
    
    Параметры:
      video_path (str): Путь к исходному видеофайлу.
      faces_dir (str): Путь для сохранения изображений лиц.
      output_video_path (str): Путь для сохранения обработанного видео.
      face_conf_threshold (float): Порог уверенности для аннотации лица.
      align (bool): Флаг использования дополнительного выравнивания.
      progress_callback (function): Функция для обновления прогресса обработки.
      csv_output_path (str): Путь для сохранения CSV с результатами.
      
    Возвращает:
      dict: Словарь объектов FaceMetrics для каждого уникального лица.
    """
    tracked_faces = {}
    os.makedirs(faces_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame_image = cap.read()
        if not ret:
            print("Видео закончено")
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        try:
            analysis_result = DeepFace.analyze(
                img_path=np.array(pil_image),
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend='centerface',
                enforce_detection=False,
                align=align,
                silent=True
            )
        except Exception as e:
            print(f"Ошибка при обработке кадра {frame_count}: {e}")
            analysis_result = None
        
        face_results = analysis_result if analysis_result is not None and isinstance(analysis_result, list) else ([analysis_result] if analysis_result is not None else [])
        
        if face_results:
            region = face_results[0]['region']
            w_face, h_face = region['w'], region['h']
            frame_w, frame_h = pil_image.size
            if not (w_face == frame_w - 1 and h_face == frame_h - 1):
                draw = ImageDraw.Draw(pil_image)
                font_size = pil_image.size[1] // 40  # динамический размер шрифта
                try:
                    font = ImageFont.truetype("fonts/LiberationMono-Regular.ttf", size=font_size)
                except Exception:
                    font = ImageFont.load_default()
                box_color = "red"
                text_color = "yellow"
                fill_color = "black"
                
                for number_face, face_result in enumerate(face_results):
                    metrics = get_face_matrics(face_result)
                    x, y, w_face, h_face = metrics['x'], metrics['y'], metrics['w'], metrics['h']
                    face_img = pil_image.crop((x, y, x + w_face, y + h_face))
                    face_id = match_face(np.array(face_img), faces_dir)
                    if face_id is None:
                        face_name = f'fr{frame_count}_fc{number_face}'
                        face_id = os.path.join(faces_dir, f"{face_name}.jpg")
                        face_img.save(face_id)
                    if face_id in tracked_faces:
                        tracked_faces[face_id].update(metrics)
                    else:
                        tracker = FaceMetrics(face_id)
                        tracker.update(metrics)
                        tracked_faces[face_id] = tracker
                    
                    gender_text = tracked_faces[face_id].get_dominant_gender()
                    age_text = tracked_faces[face_id].get_dominant_age()
                    race_text = tracked_faces[face_id].get_dominant_race()
                    emotion_text = tracked_faces[face_id].get_emotion()
                    text = f"{gender_text}, {age_text}, {race_text}, {emotion_text}"
                    
                    draw.rectangle([(x, y), (x + w_face, y + h_face)], outline=box_color, width=2)
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = x if (x + text_width) <= pil_image.size[0] else x - text_width // 2
                    draw.rectangle([(text_x, y - text_height), (text_x + text_width, y)], fill=fill_color)
                    draw.text((text_x, y - text_height), text, font=font, fill=text_color)
                
                annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(annotated_frame)
            else:
                out.write(frame_image)
        else:
            out.write(frame_image)
        
        if progress_callback is not None:
            progress_callback(frame_count, total_frames)
    
    cap.release()
    out.release()
    
    # Сохранение результатов в CSV с использованием переданного пути
    conver_and_save_detected_faces(tracked_faces, csv_output_path)
    return tracked_faces

def conver_and_save_detected_faces(tracked_faces, csv_output_path):
    """
    Формирует итоговый CSV с информацией о каждом распознанном лице и сохраняет его.
    
    Параметры:
      tracked_faces (dict): Словарь с объектами FaceMetrics.
      csv_output_path (str): Путь для сохранения итогового CSV.
    """
    result_list = []
    for tr in tracked_faces.values():
        df1 = pd.DataFrame({'age': [tr.get_dominant_age()]})
        df2 = pd.DataFrame({'gender': [tr.get_dominant_gender()]})
        df3 = pd.DataFrame({'race': [tr.get_dominant_race()]})
        df4 = pd.DataFrame(tr.count_emotions(), index=[0])
        df_combined = pd.concat([df1, df2, df3, df4], axis=1)
        result_list.append(df_combined)
    df_results = pd.concat(result_list, axis=0, ignore_index=True)
    df_results.to_csv(csv_output_path, index=False)
    print(f"Результаты сохранены в {csv_output_path}")

def convert_video(input_video_path: str, output_video_path: str) -> None:
    """
    Конвертирует видео с использованием ffmpeg для корректного отображения в браузере.
    
    Параметры:
      input_video_path (str): Путь к исходному видеофайлу.
      output_video_path (str): Путь для сохранения конвертированного видео.
    """
    ffmpeg = FFmpeg().option('y').input(input_video_path).output(output_video_path)
    ffmpeg.execute()
