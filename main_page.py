import os
import streamlit as st
import results_display

# ===================== Настройка страницы Streamlit =====================
st.set_page_config(
    layout='wide',
    page_title='Приложение распознавания эмоций, пола, возраста, расы',
    page_icon='📹'
)
st.title("Приложение распознавания эмоций, пола, возраста, расы")

# ===================== Определение путей к файлам в папке media =====================
# Путь к обработанному видео
MAIN_VIDEO_PATH = os.path.join("media", "result_video.mp4")
# Путь к CSV с результатами (для главной страницы файлы расположены в media)
CSV_PATH = os.path.join("media", "video_results.csv")

# ===================== Отображение предпросмотра видео =====================
video_width = 280
video_side = 100

if os.path.exists(MAIN_VIDEO_PATH):
    with open(MAIN_VIDEO_PATH, 'rb') as file:
        video_bytes = file.read()
    st.subheader("Пример обработанного видео")
    # Центрирование видео с помощью колонок
    _, container, _ = st.columns([video_side, video_width, video_side])
    container.video(data=video_bytes)
else:
    st.info("Обработанное видео не найдено в папке media.")

# ===================== Отображение результатов в виде графиков =====================
st.header("Отображение результатов")
# Функция display_results принимает параметр csv_path, по которому строятся графики
results_display.display_results(csv_path=CSV_PATH)
