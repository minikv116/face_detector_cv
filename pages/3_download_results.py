# ---
# page_title: "Скачать результаты"
# page_icon: "📥"
# layout: wide
# ---

import os
import streamlit as st

# ===================== Настройка страницы =====================
st.set_page_config(page_title="Скачать результаты", layout="wide")
st.title("Скачать результаты")

# ===================== Определение пути к папке с результатами =====================
results_folder = "results_folder"

if not os.path.exists(results_folder):
    st.info("Папка results_folder не найдена.")
else:
    # Получаем список подкаталогов, названия которых являются числами (порядковые номера)
    folders = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d)) and d.isdigit()]
    # Сортировка по порядку (по числовому значению)
    folders = sorted(folders, key=lambda x: int(x))
    
    if not folders:
        st.info("Нет результатов для скачивания.")
    else:
        st.subheader("Найденные результаты:")
        # Перебор всех найденных папок с результатами
        for folder in folders:
            folder_path = os.path.join(results_folder, folder)
            st.markdown(f"### Результаты {folder}")
            
            # Определяем пути к файлам: сконвертированное видео и CSV с результатами
            converted_video_path = os.path.join(folder_path, "result_video_convert.mp4")
            csv_file_path = os.path.join(folder_path, "video_results.csv")
            
            # Разбиваем область на две колонки: одна для видео, другая для CSV
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Сконвертированное видео:**")
                if os.path.exists(converted_video_path):
                    with open(converted_video_path, "rb") as f:
                        video_data = f.read()
                    st.download_button(
                        label="Скачать видео",
                        data=video_data,
                        file_name=f"result_video_convert_{folder}.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.info("Сконвертированное видео не найдено.")
                    
            with col2:
                st.markdown("**CSV с результатами:**")
                if os.path.exists(csv_file_path):
                    with open(csv_file_path, "rb") as f:
                        csv_data = f.read()
                    st.download_button(
                        label="Скачать CSV",
                        data=csv_data,
                        file_name=f"video_results_{folder}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("CSV файл не найден.")
