import os
import itertools
import pandas as pd
import streamlit as st
import altair as alt

def categorize_age(age):
    """
    Преобразует числовой возраст в возрастную группу.
    
    Параметры:
      age (int): Числовое значение возраста.
      
    Возвращает:
      str: Возрастная группа.
    """
    if age < 18:
        return 'до 18'
    elif 18 <= age <= 25:
        return '18–25'
    elif 26 <= age <= 40:
        return '26–40'
    elif 41 <= age <= 60:
        return '41–60'
    else:
        return '60+'

def display_results(csv_path="media/video_results.csv"):
    """
    Отображает результаты распознавания лиц: загружает CSV, строит таблицу и графики.
    
    Параметры:
      csv_path (str): Путь к CSV файлу с результатами.
      
    Выполняет:
      - Загрузку CSV с результатами и вывод таблицы.
      - Построение распределений по полу, эмоциям и возрасту.
      - Построение группированных и 100% накопленных столбчатых графиков.
    
    Документация по Altair: https://altair-viz.github.io/
    """
    # Проверяем, существует ли CSV файл
    if not os.path.exists(csv_path):
        st.error("CSV с результатами не найден. Сначала выполните обработку видео.")
        return

    # Загружаем данные из CSV
    df = pd.read_csv(csv_path)
    
    # Преобразование значений пола для отображения (англ. -> рус.)
    gender_map = {"Man": "Мужчина", "Woman": "Женщина"}
    df['gender'] = df['gender'].map(gender_map).fillna(df['gender'])
    
    # Определение списка эмоций и их русских названий
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
        if emo not in df.columns:
            df[emo] = 0

    # Отображение таблицы результатов
    st.subheader("Таблица результатов")
    st.dataframe(df)

    # Построение простых распределений
    st.subheader("Простые распределения")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Распределение по полу**")
        gender_counts = df['gender'].value_counts().reindex(["Мужчина", "Женщина"]).fillna(0)
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
        emotion_sums = df[emotions].sum()
        # Преобразуем ключи эмоций в русские названия
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
        df_age = df[(df['age'] >= 1) & (df['age'] <= 99)]
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

    # Построение группированных столбчатых графиков
    st.subheader("Группированные столбчатые графики")
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**Распределение эмоций по полу**")
        # Преобразуем данные в длинный формат для группировки по полу и эмоциям
        df_gender = df[['gender'] + emotions].copy()
        df_gender = df_gender.melt(id_vars='gender', value_vars=emotions,
                                   var_name='emotion', value_name='count')
        df_gender = df_gender.groupby(['gender', 'emotion'], as_index=False)['count'].sum()
        # Преобразуем названия эмоций в русские
        df_gender['emotion'] = df_gender['emotion'].map(emotion_map)
        
        # Построение группированного графика с использованием xOffset для смещения баров по полу
        chart_gender = alt.Chart(df_gender).mark_bar().encode(
            x=alt.X('emotion:N', title='Эмоция'),
            xOffset=alt.X('gender:N', title='Пол'),
            y=alt.Y('count:Q', title='Количество'),
            color=alt.Color('gender:N', title='Пол', scale=alt.Scale(domain=["Мужчина", "Женщина"]))
        ).properties(width=300, height=300)
        st.altair_chart(chart_gender, use_container_width=True)
    
    with col5:
        st.markdown("**Распределение эмоций по возрастным группам**")
        # Преобразуем возраст в возрастные группы
        df['age_group'] = df['age'].apply(categorize_age)
        df_age_group = df[['age_group'] + emotions].copy()
        df_age_group = df_age_group.melt(id_vars='age_group', value_vars=emotions,
                                         var_name='emotion', value_name='count')
        df_age_group = df_age_group.groupby(['age_group', 'emotion'], as_index=False)['count'].sum()
        df_age_group['emotion'] = df_age_group['emotion'].map(emotion_map)
        
        # Для корректного отображения всех комбинаций создаём DataFrame с полным набором возрастных групп и эмоций
        age_groups = ['до 18', '18–25', '26–40', '41–60', '60+']
        all_combinations = pd.DataFrame(list(itertools.product(age_groups, list(emotion_map.values()))),
                                        columns=['age_group','emotion'])
        df_age_group = all_combinations.merge(df_age_group, on=['age_group','emotion'], how='left').fillna(0)
        
        # Построение группированного графика с использованием xOffset для смещения баров по возрастным группам
        chart_age_group = alt.Chart(df_age_group).mark_bar().encode(
            x=alt.X('emotion:N', title='Эмоция'),
            xOffset=alt.X('age_group:N', title='Возрастная группа'),
            y=alt.Y('count:Q', title='Количество'),
            color=alt.Color('age_group:N', title='Возрастная группа')
        ).properties(width=300, height=300)
        st.altair_chart(chart_age_group, use_container_width=True)
    
    # Построение 100% накопленных столбчатых графиков
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
