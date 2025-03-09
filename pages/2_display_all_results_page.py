import os
import pandas as pd
import streamlit as st
import altair as alt
import itertools

# ===================== Настройка страницы общего отображения результатов =====================
st.set_page_config(page_title="Общее отображение результатов", layout="wide")
st.title("Общее отображение результатов")

# ===================== Поиск CSV файлов в папке results_folder =====================
base_results_folder = "results_folder"
csv_files = []

if os.path.exists(base_results_folder):
    for d in os.listdir(base_results_folder):
        folder_path = os.path.join(base_results_folder, d)
        if os.path.isdir(folder_path) and d.isdigit():
            csv_path = os.path.join(folder_path, "video_results.csv")
            if os.path.exists(csv_path):
                csv_files.append(csv_path)

if not csv_files:
    st.info("CSV файлы не найдены в папке results_folder.")
else:
    # ===================== Панель выбора CSV файлов =====================
    st.sidebar.header("Выбор CSV файлов для отображения")
    selected_files = []
    # Для каждого найденного CSV создаётся галочка для его выбора
    for csv_file in csv_files:
        if st.sidebar.checkbox(f"{csv_file}", value=True):
            selected_files.append(csv_file)
    
    if selected_files:
        # Объединяем выбранные CSV файлы в один DataFrame
        combined_df = pd.concat([pd.read_csv(f) for f in selected_files], ignore_index=True)
        
        st.subheader("Объединенная таблица результатов")
        st.dataframe(combined_df)
        
        # ===================== Подготовка данных для построения графиков =====================
        # Преобразуем значения пола для отображения (с англ. на рус.)
        gender_map = {"Man": "Мужчина", "Woman": "Женщина"}
        if "gender" in combined_df.columns:
            combined_df['gender'] = combined_df['gender'].map(gender_map).fillna(combined_df['gender'])
        
        # Определяем соответствие эмоций: ключи – значения из CSV, значения – для отображения
        emotion_map = {
            'angry': 'злость',
            'disgust': 'отвращение',
            'fear': 'страх',
            'happy': 'радость',
            'sad': 'грусть',
            'surprise': 'удивление'
        }
        emotions = list(emotion_map.keys())
        for emo in emotions:
            if emo not in combined_df.columns:
                combined_df[emo] = 0
        
        # ===================== Построение простых распределений =====================
        st.subheader("Простые распределения")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Распределение по полу**")
            gender_counts = combined_df['gender'].value_counts().reindex(["Мужчина", "Женщина"]).fillna(0)
            gender_chart = alt.Chart(pd.DataFrame({
                'Пол': gender_counts.index,
                'Количество': gender_counts.values
            })).mark_bar().encode(
                x=alt.X('Пол:N', title='Пол'),
                y=alt.Y('Количество:Q', title='Количество')
            ).properties(width=300, height=300)
            st.altair_chart(gender_chart, use_container_width=True)
        with col2:
            st.markdown("**Распределение по эмоциям**")
            emotion_sums = combined_df[emotions].sum()
            # Преобразуем названия эмоций согласно emotion_map
            emotion_sums.index = [emotion_map.get(x, x) for x in emotion_sums.index]
            emotion_chart = alt.Chart(pd.DataFrame({
                'Эмоция': emotion_sums.index,
                'Количество': emotion_sums.values
            })).mark_bar().encode(
                x=alt.X('Эмоция:N', title='Эмоция'),
                y=alt.Y('Количество:Q', title='Количество')
            ).properties(width=300, height=300)
            st.altair_chart(emotion_chart, use_container_width=True)
        with col3:
            st.markdown("**Распределение по возрасту (по годам)**")
            df_age = combined_df[(combined_df['age'] >= 1) & (combined_df['age'] <= 99)]
            all_years = list(range(1, 100))
            age_counts = df_age['age'].value_counts().reindex(all_years, fill_value=0).sort_index()
            age_data = pd.DataFrame({
                'Возраст': age_counts.index,
                'Количество': age_counts.values
            })
            age_chart = alt.Chart(age_data).mark_bar().encode(
                x=alt.X('Возраст:O', title='Возраст', sort=[str(year) for year in all_years]),
                y=alt.Y('Количество:Q', title='Количество')
            ).properties(width=300, height=300)
            st.altair_chart(age_chart, use_container_width=True)
        
        # ===================== Построение группированных столбчатых графиков =====================
        st.subheader("Группированные столбчатые графики")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("**Распределение эмоций по полу**")
            df_gender = combined_df[['gender'] + emotions].copy()
            df_gender = df_gender.melt(id_vars='gender', value_vars=emotions,
                                       var_name='emotion', value_name='count')
            df_gender = df_gender.groupby(['gender', 'emotion'], as_index=False)['count'].sum()
            df_gender['emotion'] = df_gender['emotion'].map(emotion_map)
            chart_gender = alt.Chart(df_gender).mark_bar().encode(
                x=alt.X('emotion:N', title='Эмоция'),
                y=alt.Y('count:Q', title='Количество'),
                color=alt.Color('gender:N', title='Пол', scale=alt.Scale(domain=["Мужчина", "Женщина"]))
            ).properties(width=300, height=300)
            st.altair_chart(chart_gender, use_container_width=True)
        with col5:
            st.markdown("**Распределение эмоций по возрастным группам**")
            # Создаем возрастные группы
            combined_df['age_group'] = combined_df['age'].apply(
                lambda x: 'до 18' if x < 18 else ('18–25' if 18 <= x <= 25 else ('26–40' if 26 <= x <= 40 else ('41–60' if 41 <= x <= 60 else '60+')))
            )
            df_age_group = combined_df[['age_group'] + emotions].copy()
            df_age_group = df_age_group.melt(id_vars='age_group', value_vars=emotions,
                                             var_name='emotion', value_name='count')
            df_age_group = df_age_group.groupby(['age_group', 'emotion'], as_index=False)['count'].sum()
            df_age_group['emotion'] = df_age_group['emotion'].map(emotion_map)
            age_groups = ['до 18', '18–25', '26–40', '41–60', '60+']
            all_combinations = pd.DataFrame(list(itertools.product(age_groups, list(emotion_map.values()))),
                                            columns=['age_group','emotion'])
            df_age_group = all_combinations.merge(df_age_group, on=['age_group','emotion'], how='left').fillna(0)
            chart_age_group = alt.Chart(df_age_group).mark_bar().encode(
                x=alt.X('emotion:N', title='Эмоция'),
                y=alt.Y('count:Q', title='Количество'),
                color=alt.Color('age_group:N', title='Возрастная группа')
            ).properties(width=300, height=300)
            st.altair_chart(chart_age_group, use_container_width=True)
        
        # ===================== Построение 100% накопленных столбчатых графиков =====================
        st.subheader("100% накопленные столбчатые графики")
        col6, col7 = st.columns(2)
        with col6:
            st.markdown("**100% распределение эмоций по полу**")
            df_gender_pct = df_gender.copy()
            total_by_gender = df_gender_pct.groupby('gender')['count'].transform('sum')
            df_gender_pct['percent'] = df_gender_pct['count'] / total_by_gender * 100
            chart_gender_pct = alt.Chart(df_gender_pct).mark_bar().encode(
                x=alt.X('emotion:N', title='Эмоция'),
                y=alt.Y('percent:Q', title='Процент', stack="normalize"),
                color=alt.Color('gender:N', title='Пол', scale=alt.Scale(domain=["Мужчина", "Женщина"]))
            ).properties(width=300, height=300)
            st.altair_chart(chart_gender_pct, use_container_width=True)
        with col7:
            st.markdown("**100% распределение эмоций по возрастным группам**")
            df_age_pct = df_age_group.copy()
            total_by_age = df_age_pct.groupby('age_group')['count'].transform('sum')
            df_age_pct['percent'] = df_age_pct['count'] / total_by_age * 100
            chart_age_pct = alt.Chart(df_age_pct).mark_bar().encode(
                x=alt.X('emotion:N', title='Эмоция'),
                y=alt.Y('percent:Q', title='Процент', stack="normalize"),
                color=alt.Color('age_group:N', title='Возрастная группа')
            ).properties(width=300, height=300)
            st.altair_chart(chart_age_pct, use_container_width=True)
    else:
        st.info("Не выбраны CSV файлы для отображения.")
