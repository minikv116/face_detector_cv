# ---
# page_title: "Обработка видео_"
# page_icon: "📹"
# layout: wide
# ---

import os
import tempfile
import streamlit as st
from video_handler import process_video_one_cell, convert_video
import results_display

# ===================== Настройка страницы обработки видео =====================
st.set_page_config(page_title="Обработка видео", layout="wide")
st.title("Обработка видео")

# ===================== Настройки детекции на боковой панели =====================
face_conf_threshold = st.sidebar.slider(
    label='Порог уверенности для детекции лиц',
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
)
align = False
#align = st.sidebar.checkbox(label='Align', value=False)

# ===================== Загрузка видео пользователем =====================
st.subheader("Загрузка видео")
st_video = st.file_uploader(label='Выберите видео для обработки', type=["mp4", "avi", "mov"])

# Задаём размеры для центровки видео (аналогично main_page.py)
video_width = 60
video_side = 100

if st_video:
    if st.button('Начать обработку видео'):
        with st.spinner("Обработка видео..."):
            # Сохраняем загруженное видео во временный файл
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(st_video.read())
            video_file = temp_file.name
            
            # ===================== Определение новой папки в results_folder =====================
            base_results_folder = "results_folder"
            os.makedirs(base_results_folder, exist_ok=True)
            # Получаем список существующих папок с числовыми названиями
            existing = [int(d) for d in os.listdir(base_results_folder) if d.isdigit()]
            next_num = max(existing) + 1 if existing else 1
            new_folder = os.path.join(base_results_folder, str(next_num))
            os.makedirs(new_folder, exist_ok=True)
            
            # Задаём пути для сохранения: обработанного видео, сконвертированного видео, CSV и изображений лиц
            output_video_path = os.path.join(new_folder, "result_video.mp4")
            convert_video_path = os.path.join(new_folder, "result_video_convert.mp4")
            csv_output_path = os.path.join(new_folder, "video_results.csv")
            faces_dir = os.path.join(new_folder, "faces")
            os.makedirs(faces_dir, exist_ok=True)
            
            # ===================== Создание прогресс-бара для отображения хода обработки =====================
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def update_progress(current, total):
                """
                Функция обновления прогресса обработки видео.
                
                Параметры:
                  current (int): текущий номер обработанного кадра.
                  total (int): общее количество кадров.
                """
                progress = current / total
                progress_bar.progress(progress)
                progress_text.text(f"Обработка кадра {current} из {total}")
            
            # Запуск обработки видео с сохранением CSV в новой папке
            tracked_faces = process_video_one_cell(
                video_path=video_file,
                faces_dir=faces_dir,
                output_video_path=output_video_path,
                face_conf_threshold=face_conf_threshold,
                align=align,
                progress_callback=update_progress,
                csv_output_path=csv_output_path
            )
        st.success("Обработка видео завершена!")
        
        # ===================== Конвертация видео для отображения в браузере =====================
        with st.spinner('Идет конвертация видео ...'):
            convert_video(output_video_path, convert_video_path)
        
        with open(convert_video_path, 'rb') as file:
            video_bytes = file.read()
        
        st.subheader("Результат обработки видео")
        _, container, _ = st.columns([video_side, video_width, video_side])
        container.video(data=video_bytes)
        
        st.download_button(
            label='Скачать видео',
            data=video_bytes,
            file_name=os.path.basename(convert_video_path)
        )
        
        # ===================== Отображение графиков результатов после обработки =====================
        st.header("Отображение результатов")
        results_display.display_results(csv_path=csv_output_path)
