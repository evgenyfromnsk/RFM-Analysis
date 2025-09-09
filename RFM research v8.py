#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import psycopg2                     # модуль для подключения к базе Postgres
import getpass                      # модуль для более безопасного ввода пароля (без локального хранения на ПК)
import pandas.io.sql as sqlio       # функции pandas для приёма SQL запросов
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  
from matplotlib.patches import Rectangle
from matplotlib.sankey import Sankey
from collections import Counter
from itertools import groupby
from itertools import tee
from collections import defaultdict


# In[3]:


query = """
select * from sandbox.rfm_new_may20250930
"""


# In[4]:


df = sqlio.read_sql_query(query,conn)
conn.close()


# In[5]:


# 1. Подсчёт уникальных email по сегментам
pivot = df.groupby(['rfm_segment']).agg(count=('email', 'nunique')).reset_index()

# 2. Подсчёт процентов и округление до целого
total = pivot['count'].sum()
pivot['percent'] = (pivot['count'] / total * 100).round(0).astype(int)

# 3. Метки для treemap — название + процент
pivot['label'] = pivot['rfm_segment'].astype(str) + '\n' + pivot['percent'].astype(str) + '%'

# 4. Mapping RFM -> группы
group_map = {
    'Best Customers': 'Best & Big Spenders',
    'Big Spenders': 'Best & Big Spenders',
    'Loyal Customers': 'Best & Big Spenders',
    'New Customers': 'New & Potential',
    'Potential Loyalists': 'New & Potential',
    'About To Sleep': 'Sleeping & Lost',
    'At Risk': 'Sleeping & Lost',
    'Lost': 'Sleeping & Lost',
    'Need Attention': 'Sleeping & Lost',
    'Others': 'Sleeping & Lost'
}
pivot['group'] = pivot['rfm_segment'].map(group_map)

# 5. Pastel colors — более мягкие варианты
pastel_colors = {
    'Best & Big Spenders': ['#e6f9e6', '#ccf2cc', '#b3ecb3'],  # зелёные
    'New & Potential': ['#fff7b3', '#ffe680'],                 # жёлтые
    'Sleeping & Lost': ['#ffe6e6', '#ffd6d6', '#ffc6c6', '#ffe0e0', '#fff0f0']  # красные/розовые
}

# 6. Точный порядок сегментов
segment_order = [
    'Best Customers', 'Big Spenders', 'Loyal Customers',
    'New Customers', 'Potential Loyalists',
    'About To Sleep', 'At Risk', 'Lost', 'Need Attention', 'Others'
]

# 7. Сортировка pivot по порядку сегментов
pivot['seg_order'] = pivot['rfm_segment'].apply(lambda x: segment_order.index(x))
pivot = pivot.sort_values('seg_order').reset_index(drop=True)

# 8. Присвоение цветов корректно по группам
used_color_idx = {group: 0 for group in pastel_colors.keys()}
colors = []
for seg in segment_order:
    group = group_map[seg]
    palette = pastel_colors[group]
    
    # Берём следующий цвет для этой группы
    color = palette[used_color_idx[group] % len(palette)]
    colors.append(color)
    
    # Увеличиваем счётчик для этой группы
    used_color_idx[group] += 1

pivot['color'] = colors

# 9. Построение treemap
plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=pivot['count'],
    label=pivot['label'],  # название + округлённый процент
    color=pivot['color'],
    alpha=.95,
    text_kwargs={'fontsize':11, 'weight':'bold', 'color':'black'}
)
plt.axis('off')
plt.title("RFM Segments 07 2025", fontsize=16)
plt.show()


# In[6]:


df_ppc = df[df['medium_group'] == 'PPC']
# Группировка
pivot = df_ppc.groupby(['rfm_segment']).agg(
    count=('email', 'nunique')
).reset_index()

# Проценты и подписи
total = pivot['count'].sum()
pivot['percent'] = (pivot['count'] / total * 100).round(0).astype(int)
pivot['label'] = pivot['rfm_segment'] + '\n' + pivot['count'].astype(str) + ' (' + pivot['percent'].astype(str) + '%)'

# 3. Метки для treemap — название + процент
pivot['label'] = pivot['rfm_segment'].astype(str) + '\n' + pivot['percent'].astype(str) + '%'

# 4. Mapping RFM -> группы
group_map = {
    'Best Customers': 'Best & Big Spenders',
    'Big Spenders': 'Best & Big Spenders',
    'Loyal Customers': 'Best & Big Spenders',
    'New Customers': 'New & Potential',
    'Potential Loyalists': 'New & Potential',
    'About To Sleep': 'Sleeping & Lost',
    'At Risk': 'Sleeping & Lost',
    'Lost': 'Sleeping & Lost',
    'Need Attention': 'Sleeping & Lost',
    'Others': 'Sleeping & Lost'
}
pivot['group'] = pivot['rfm_segment'].map(group_map)

# 5. Pastel colors — более мягкие варианты
pastel_colors = {
    'Best & Big Spenders': ['#e6f9e6', '#ccf2cc', '#b3ecb3'],  # зелёные
    'New & Potential': ['#fff7b3', '#ffe680'],                 # жёлтые
    'Sleeping & Lost': ['#ffe6e6', '#ffd6d6', '#ffc6c6', '#ffe0e0', '#fff0f0']  # красные/розовые
}

# 6. Точный порядок сегментов
segment_order = [
    'Best Customers', 'Big Spenders', 'Loyal Customers',
    'New Customers', 'Potential Loyalists',
    'About To Sleep', 'At Risk', 'Lost', 'Need Attention', 'Others'
]

# 7. Сортировка pivot по порядку сегментов
pivot['seg_order'] = pivot['rfm_segment'].apply(lambda x: segment_order.index(x))
pivot = pivot.sort_values('seg_order').reset_index(drop=True)

# 8. Присвоение цветов корректно по группам
used_color_idx = {group: 0 for group in pastel_colors.keys()}
colors = []
for seg in segment_order:
    group = group_map[seg]
    palette = pastel_colors[group]
    
    # Берём следующий цвет для этой группы
    color = palette[used_color_idx[group] % len(palette)]
    colors.append(color)
    
    # Увеличиваем счётчик для этой группы
    used_color_idx[group] += 1

pivot['color'] = colors

# 9. Построение treemap
plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=pivot['count'],
    label=pivot['label'],  # название + округлённый процент
    color=pivot['color'],
    alpha=.95,
    text_kwargs={'fontsize':11, 'weight':'bold', 'color':'black'}
)
plt.axis('off')
plt.title("PPC RFM Segments 07 2025", fontsize=16)
plt.show()


# In[7]:


df['productgroup'] = df['productgroup'].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
)


segments = df['rfm_segment'].dropna().unique()

# Подсчёт продуктов по сегментам
segment_product_counts = {}

for segment in segments:
    segment_df = df[df['rfm_segment'] == segment]
    all_products = [item for sublist in segment_df['productgroup'].dropna() for item in sublist if item is not None]
    product_counter = Counter(all_products)
    segment_product_counts[segment] = product_counter

# уникальные продукты (без None)
all_products = sorted(
    set(p for counter in segment_product_counts.values() for p in counter if p is not None)
)

# таблица для хитмапы
heatmap_data = pd.DataFrame(index=all_products, columns=segments).fillna(0)

for segment in segments:
    total = sum(segment_product_counts[segment].values())
    for product in all_products:
        count = segment_product_counts[segment].get(product, 0)
        heatmap_data.loc[product, segment] = count / total if total > 0 else 0

# беру топ 20
top_n = 20
top_products = heatmap_data.sum(axis=1).sort_values(ascending=False).head(top_n).index
heatmap_trimmed = heatmap_data.loc[top_products]

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_trimmed, annot=True, fmt=".2%", cmap="YlGnBu", linewidths=0.5)
plt.title("Распределение RFM-сегментов по продуктам")
plt.xlabel("RFM-сегмент")
plt.ylabel("Продукт")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[8]:


df_clean = df.dropna(subset=['country', 'rfm_segment', 'email'])

country_segment_counts = (
    df_clean.groupby(['country', 'rfm_segment'])['email']
    .nunique()
    .unstack(fill_value=0)
)

# Переводим в доли — делим по колонкам (внутри каждого сегмента)
country_segment_pct = country_segment_counts.div(country_segment_counts.sum(axis=0), axis=1)

top_countries = country_segment_counts.sum(axis=1).sort_values(ascending=False).head(20).index

heatmap_data = country_segment_pct.loc[top_countries]

plt.figure(figsize=(16, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1%",
    cmap='YlGnBu',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'format': '%.0f%%'}
)

plt.title("Доля уникальных пользователей из стран внутри RFM-сегментов", fontsize=16)
plt.xlabel("RFM-сегмент")
plt.ylabel("Страна")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[9]:


# check=df[df['rfm_segment']=='New Customers']


# In[10]:


# check['email'].nunique()


# In[11]:


# check.groupby('country')['email'].nunique().sort_values(ascending=False)


# In[12]:



# # Удалим строки, где country или segment отсутствуют
# df_clean = df.dropna(subset=['medium_group', 'rfm_segment'])

# # Считаем частоту сегментов по странам
# medium_segment_counts = df_clean.groupby(['medium_group', 'rfm_segment']).size().unstack(fill_value=0)

# # Переводим в проценты по строкам (странам)
# medium_segment_pct = medium_segment_counts.div(medium_segment_counts.sum(axis=1), axis=0)

# heatmap_data = medium_segment_pct


# plt.figure(figsize=(14, 8))

# sns.heatmap(
#     heatmap_data,         # твой финальный датафрейм
#     annot=True,           # показать значения
#     fmt=".2%",            # формат процентов (0.1234 → 12.34%)
#     cmap='YlGnBu',        # цветовая палитра
#     linewidths=0.5,
#     linecolor='white',
#     cbar_kws={'format': '%.0f%%'}  # шкала цвета
# )

# plt.title("Доля RFM-сегментов по Каналам", fontsize=16)
# plt.xlabel("RFM-сегмент")
# plt.ylabel("Страна")
# plt.tight_layout()
# plt.show()


# In[13]:


df_clean = df.dropna(subset=['medium_group', 'rfm_segment'])

medium_segment_counts = (
    df_clean.groupby(['medium_group', 'rfm_segment'])['email']
    .nunique()
    .unstack(fill_value=0)
)

# Делим по колонкам (по сегменту): какую долю формирует каждый канал
medium_segment_pct = medium_segment_counts.div(medium_segment_counts.sum(axis=0), axis=1)

# Оставим топ-30 каналов по общему количеству уникальных email
top_channels = medium_segment_counts.sum(axis=1).sort_values(ascending=False).head(30).index
heatmap_data = medium_segment_pct.loc[top_channels]


plt.figure(figsize=(16, 8))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1%",
    cmap='YlGnBu',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'format': '%.0f%%'}
)

plt.title("Вклад каналов в формирование RFM-сегментов", fontsize=16)
plt.xlabel("RFM-сегмент")
plt.ylabel("Канал (medium_group)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Динамика RFM сегментов по месяцам



# In[15]:


df_agg = sqlio.read_sql_query(query_agg,conn)
conn.close()


# In[16]:


df_agg['mth'] = pd.to_datetime(df_agg['mth'])

# Переход от long → wide формат для stacked area chart
pivot_df = df_agg.pivot(index='mth', columns='rfm_segment', values='emails')
pivot_df = pivot_df.fillna(0)

# Строим график
plt.figure(figsize=(14, 8))
pivot_df.plot(kind='area', stacked=True, figsize=(14, 8), cmap='tab20')

plt.title("Динамика RFM-сегментов по месяцам", fontsize=16)
plt.xlabel("Месяц")
plt.ylabel("Количество email'ов")
plt.legend(title="RFM сегмент", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(alpha=0.3)

plt.show()


# In[17]:


df_agg['mth'] = pd.to_datetime(df_agg['mth'])

# Pivot для wide-формата
pivot_df1 = df_agg.pivot(index='mth', columns='rfm_segment', values='emails')
pivot_df1 = pivot_df1.fillna(0)

# Нормируем строки на 100%
pivot_pct = pivot_df1.div(pivot_df1.sum(axis=1), axis=0) * 100

# Визуализация в %
plt.figure(figsize=(14, 8))
pivot_pct.plot(kind='area', stacked=True, figsize=(14, 8), cmap='tab20')

plt.title("Доля RFM-сегментов по месяцам (%)", fontsize=16)
plt.xlabel("Месяц")
plt.ylabel("Доля сегмента (%)")
plt.legend(title="RFM сегмент", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(alpha=0.3)

plt.show()


# # Transitions

# In[18]:


query4 = """
WITH rfm_data AS (
    SELECT 
        email,
        mth,
        rfm_segment,
        CASE 
            WHEN mth BETWEEN '2025-01-01' AND '2025-03-31' THEN 'Q1'
            WHEN mth BETWEEN '2025-04-01' AND '2025-06-30' THEN 'Q2'
        END AS quarter
    FROM sandbox.rfm_new_20250930
    WHERE mth BETWEEN '2025-01-01' AND '2025-06-30'
),

last_segment_per_q AS (
    SELECT DISTINCT 
        quarter,
        email,
        first_value(rfm_segment) over (partition by quarter, email order by  mth desc) rfm_segment
    FROM rfm_data
),


paired_segments AS (
    SELECT 
        q1.email,
        q1.rfm_segment AS prev_segment,
        q2.rfm_segment AS current_segment
    FROM last_segment_per_q q1
    JOIN last_segment_per_q q2 ON q1.email = q2.email
    WHERE q1.quarter = 'Q1' AND q2.quarter = 'Q2'
),


transitions AS (
    SELECT 
        prev_segment,
        current_segment,
        COUNT(distinct email) AS cnt
    FROM paired_segments
    WHERE prev_segment IS NOT NULL AND current_segment IS NOT NULL
    GROUP BY prev_segment, current_segment
),


totals AS (
    SELECT 
        prev_segment,
        SUM(cnt) AS total
    FROM transitions
    GROUP BY prev_segment
)


SELECT 
    t.prev_segment,
    t.current_segment,
    t.cnt,
    tot.total,
    ROUND(t.cnt::NUMERIC / tot.total * 100, 1) AS percentage
FROM transitions t
JOIN totals tot ON t.prev_segment = tot.prev_segment
ORDER BY t.prev_segment, percentage DESC

"""


# In[19]:


# In[20]:


# --- HEATMAP ---
heatmap_data = df_4.pivot(index='prev_segment', columns='current_segment', values='percentage').fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='Blues')
plt.title('Процент переходов между RFM сегментами Q1 → Q2')
plt.xlabel('Сегмент в Q2')
plt.ylabel('Сегмент в Q1')
plt.tight_layout()
plt.show()


# In[21]:


# Цветовая схема сегментов
SEGMENT_COLORS = {
    'About To Sleep': '#1f77b4',
    'Need Attention': '#e377c2',
    'At Risk': '#ff7f0e',
    'Loyal Customers': '#8c564b',
    'Best Customers': '#2ca02c',
    'Potential Loyalists': '#bcbd22',
    'Cannot Lose Them': '#9467bd',
    'Big Spenders': '#d62728',
    'Others': '#7f7f7f'
}

# Фиксированный порядок всех сегментов
ALL_SEGMENTS = list(SEGMENT_COLORS.keys())

# Создаем узлы для обеих колонок (дублируем все сегменты)
left_nodes = ALL_SEGMENTS
right_nodes = ALL_SEGMENTS
all_nodes = left_nodes + right_nodes

# Словарь индексов с уникальными именами для правых узлов
node_indices = {}
for i, node in enumerate(left_nodes):
    node_indices[f"left_{node}"] = i
for i, node in enumerate(right_nodes, len(left_nodes)):
    node_indices[f"right_{node}"] = i

# Подготовка данных связей с явными подписями
sources, targets, values, link_colors = [], [], [], []
link_labels = []  # Для хранения описаний переходов

for _, row in df_4.iterrows():
    source_key = f"left_{row['prev_segment']}"
    target_key = f"right_{row['current_segment']}"
    
    if source_key in node_indices and target_key in node_indices:
        sources.append(node_indices[source_key])
        targets.append(node_indices[target_key])
        values.append(row['cnt'])
        link_colors.append(SEGMENT_COLORS[row['prev_segment']])
        link_labels.append(f"{row['prev_segment']} → {row['current_segment']}")

# Распределение узлов
left_y = np.linspace(0.05, 0.95, len(left_nodes))
right_y = np.linspace(0.05, 0.95, len(right_nodes))
y_positions = np.concatenate([left_y, right_y]).tolist()
x_positions = [0]*len(left_nodes) + [1]*len(right_nodes)

# Создаем явные подписи для всех узлов
labels = left_nodes + right_nodes

# Создаем диаграмму Sankey
fig = go.Figure(go.Sankey(
    arrangement="perpendicular",
    node=dict(
        pad=40,
        thickness=25,
        line=dict(color="black", width=1.2),
        label=labels,  # Явные подписи для всех узлов
        x=x_positions,
        y=y_positions,
        color=[SEGMENT_COLORS[n.split('_')[-1]] if '_' in n else SEGMENT_COLORS[n] for n in labels],
        hovertemplate='<b>%{label}</b><extra></extra>'
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors,
        customdata=link_labels,
        hovertemplate='<b>%{customdata}</b><br>Количество: %{value:,}<extra></extra>'
    )
))

# Настройки оформления
fig.update_layout(
    title_text="Переходы между RFM-сегментами за 2 месяца",
    title_font=dict(size=22),
    font_size=14,
    height=800,
    width=900,
    margin=dict(l=80, r=80, t=100, b=80),
    plot_bgcolor="white"
)

# Яркие подписи колонок
fig.add_annotation(
    x=0, y=1.05,
    text="<b>Q1 2025</b>",
    showarrow=False,
    xref="paper", yref="paper",
    font=dict(size=16, color="#333"),
    bgcolor="#f0f0f0"
)

fig.add_annotation(
    x=1, y=1.05,
    text="<b>Q2 2025</b>",
    showarrow=False,
    xref="paper", yref="paper",
    font=dict(size=16, color="#333"),
    bgcolor="#f0f0f0"
)

fig.show()


# # Как становятся Best Customers


query5 = """

with best as  (
 SELECT distinct email
    FROM sandbox.rfm_new_20250930
    WHERE rfm_segment = 'Best Customers'
    and mth::date>'2024-01-01'
)

, rfm_clean AS (
    SELECT 
        email,
        mth::date AS mth,
        rfm_segment
    FROM sandbox.rfm_new_20250930 s join best b using(email)
),

rfm_ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY email ORDER BY mth) AS step
    FROM rfm_clean
)

SELECT *
FROM rfm_ranked

"""


# In[23]:


df_5 = sqlio.read_sql_query(query5,conn)
conn.close()


# In[24]:


paths = []

# Сортировка и группировка по email
for email, group in df_5.sort_values(['email', 'mth']).groupby('email'):
    segments = list(group['rfm_segment'])
    
    if 'Best Customers' in segments:
        idx = segments.index('Best Customers')  # индекс первого появления Best
        path = segments[:idx + 1]

        # Удаляем подряд идущие дубликаты
        deduped_path = [key for key, _ in groupby(path)]

        # Сохраняем, только если путь длинее 1 шага
        if len(deduped_path) > 1:
            paths.append(tuple(deduped_path))

# Подсчёт частот
path_counts = Counter(paths)


for path, count in path_counts.most_common(10):
    print(f"{' → '.join(path)}: {count}")


# In[25]:


top_paths = path_counts.most_common(10)
df_top = pd.DataFrame(top_paths, columns=["path", "count"])
df_top["path_str"] = df_top["path"].apply(lambda x: " → ".join(x))

# Словарь сокращений
segment_short = {
    "New Customers": "New", "Potential Loyalists": "Potential", "Best Customers": "Best",
    "Big Spenders": "Big", "Loyal Customers": "Loyal", "At Risk": "Risk",
    "Need Attention": "Attention", "About To Sleep": "Sleep", "Lost": "Lost", "Others": "Other"
}

df_top["path_str_short"] = df_top["path"].apply(
    lambda x: " → ".join([segment_short.get(seg, seg) for seg in x])
)

# Строим график
plt.figure(figsize=(12, 8))
plt.barh(df_top["path_str_short"], df_top["count"], color="skyblue")
plt.xlabel("Количество пользователей")
plt.title("Топ путей к сегменту Best Customers")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# 
# # Доля Best Customers з?

# # Количество месяцев до перехода New -> Best


query6 = """
WITH segment_data AS (
    SELECT
        email,
        mth,
        rfm_segment
    FROM sandbox.rfm_new_20250930
),

first_new AS (
    SELECT
        email,
        MIN(mth) AS first_new_month
    FROM segment_data
    WHERE rfm_segment = 'New Customers'
    GROUP BY email
),

first_best AS (
    SELECT
        email,
        MIN(mth) AS first_best_month
    FROM segment_data
    WHERE rfm_segment = 'Best Customers'
    GROUP BY email
),

growth_cohort AS (
    SELECT
        n.email,
        n.first_new_month,
        b.first_best_month,
        (b.first_best_month - n.first_new_month)::int / 30 months_to_best
    FROM first_new n
    JOIN first_best b ON n.email = b.email
    WHERE b.first_best_month >= n.first_new_month
)

SELECT 
    months_to_best,
    COUNT(distinct email) AS users_count
FROM growth_cohort
GROUP BY months_to_best
ORDER BY months_to_best
 """


# In[40]:


df_6 = sqlio.read_sql_query(query6,conn)
conn.close()


# In[42]:


df_6_filtered = df_6[df_6['months_to_best'] < 12]

plt.figure(figsize=(8.4, 4.2))
bars = plt.bar(df_6_filtered['months_to_best'], df_6_filtered['users_count'], color='skyblue')

plt.title('Рост клиентов от New до Best Customers', fontsize=14)
plt.xlabel('Месяцы до перехода в Best Customers', fontsize=12)
plt.ylabel('Количество пользователей', fontsize=12)
plt.xticks(df_6_filtered['months_to_best'])  # отметки на оси X
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавляем значения над столбцами
max_height = df_6_filtered['users_count'].max()
for bar in bars:
    height = bar.get_height()
    label = f'{height/1000:.1f}K'
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height + max_height * 0.03,  # небольшой отступ сверху
        label, 
        ha='center', 
        va='bottom',
        fontsize=10
    )

# Увеличиваем верхнюю границу оси Y для размещения подписей
plt.ylim(0, max_height * 1.15)

plt.tight_layout()
plt.show()


# In[29]:


# plt.figure(figsize=(12,6))
# plt.figure(figsize=(8.4, 4.2))
# bars = plt.bar(df_6['months_to_best'], df_6['users_count'], color='skyblue')
# plt.title('Рост клиентов от New до Best Customers', fontsize=14)
# plt.xlabel('Месяцы до перехода в Best Customers', fontsize=12)
# plt.ylabel('Количество пользователей', fontsize=12)
# plt.xticks(df_6['months_to_best'])  # отметки на оси X
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Добавляем значения над столбцами
# for bar in bars:
#     height = bar.get_height()
#     label = f'{height/1000:.1f}K'
#     plt.text(
#         bar.get_x() + bar.get_width() / 2, 
#         height + 1200,  # чуть выше столбца
#         label, 
#         ha='center', 
#         va='bottom',
#         fontsize=10
#     )

# plt.tight_layout()
# plt.show()


# # Конверсия из New -> Best по когортам

# In[30]:

query8 = """
WITH source AS (
    SELECT 
        email,
        mth::date AS mth,
        rfm_segment
    FROM sandbox.rfm_new_20250930
    WHERE mth::date >= '2024-01-01'
),

-- 1. Для каждого email определяем, когда он впервые стал New Customers
new_customers AS (
    SELECT 
        email,
        MIN(mth) AS cohort_date
    FROM source
    WHERE rfm_segment = 'New Customers'
    GROUP BY email
),

-- 2. Дальнейшая история пользователя после когорты
user_journey AS (
    SELECT 
        s.email,
        n.cohort_date,
        s.mth,
        s.rfm_segment,
        (DATE_PART('year', s.mth) * 12 + DATE_PART('month', s.mth) - DATE_PART('year', n.cohort_date) * 12 - 
        DATE_PART('month', n.cohort_date))::int AS period_number
--        (s.mth - n.cohort_date)::int / 30 period_number
    FROM source s
    JOIN new_customers n ON s.email = n.email
    WHERE s.mth >= n.cohort_date
),

-- 3. Отметим только случаи, когда пользователь в этом месяце стал Best
converted AS (
    SELECT DISTINCT ON (email)
        email,
        cohort_date,
        period_number
    FROM user_journey
    WHERE rfm_segment = 'Best Customers'
    ORDER BY email, period_number
)
,

-- 4. Размер когорты
cohort_sizes AS (
    SELECT 
        cohort_date,
        COUNT(DISTINCT email) AS cohort_size
    FROM new_customers
    GROUP BY cohort_date
),

-- 5. Кол-во сконвертировавшихся в Best в каждый месяц
conversion_by_period AS (
    SELECT 
        cohort_date,
        period_number,
        COUNT(DISTINCT email) AS conversions
    FROM converted
    GROUP BY cohort_date, period_number
)

-- 6. Финальная таблица
SELECT 
    s.cohort_date,
    c.period_number,
    c.conversions,
    s.cohort_size,
    ROUND(c.conversions::NUMERIC / s.cohort_size, 4) AS conversion_rate
FROM cohort_sizes s
LEFT JOIN conversion_by_period c ON s.cohort_date = c.cohort_date
ORDER BY cohort_date, period_number

 """


# In[31]:


df_8 = sqlio.read_sql_query(query8,conn)
conn.close()


# In[32]:


# Преобразуем дату когорты в формат "YYYY-MM"
df_8['cohort_month'] = pd.to_datetime(df_8['cohort_date']).dt.to_period('M').astype(str)

# Строим сводную таблицу: строки — когорты, столбцы — месяцы после когорты
pivot = df_8.pivot(index='cohort_month', columns='period_number', values='conversion_rate')

# Рисуем тепловую карту
plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True)
plt.title("Конверсия в Best Customers по когортам")
plt.xlabel("Месяц после когорты")
plt.ylabel("Когорта (месяц)")
plt.tight_layout()
plt.show()


# ## PPC Cohort to Best


query8_ppc = """

WITH source AS (
    SELECT 
        email,
        mth::date AS mth,
        rfm_segment
    FROM sandbox.rfm_new_20250930
    WHERE mth::date >= '2024-01-01'
    and medium_group = 'PPC'
),

-- 1. Для каждого email определяем, когда он впервые стал New Customers
new_customers AS (
    SELECT 
        email,
        MIN(mth) AS cohort_date
    FROM source
    WHERE rfm_segment = 'New Customers'
    GROUP BY email
),

-- 2. Дальнейшая история пользователя после когорты
user_journey AS (
    SELECT 
        s.email,
        n.cohort_date,
        s.mth,
        s.rfm_segment,
        (DATE_PART('year', s.mth) * 12 + DATE_PART('month', s.mth) - DATE_PART('year', n.cohort_date) * 12 - 
        DATE_PART('month', n.cohort_date))::int AS period_number
--        (s.mth - n.cohort_date)::int / 30 period_number
    FROM source s
    JOIN new_customers n ON s.email = n.email
    WHERE s.mth >= n.cohort_date
),

-- 3. Отметим только случаи, когда пользователь в этом месяце стал Best
converted AS (
    SELECT DISTINCT ON (email)
        email,
        cohort_date,
        period_number
    FROM user_journey
    WHERE rfm_segment = 'Best Customers'
    ORDER BY email, period_number
)
,

-- 4. Размер когорты
cohort_sizes AS (
    SELECT 
        cohort_date,
        COUNT(DISTINCT email) AS cohort_size
    FROM new_customers
    GROUP BY cohort_date
),

-- 5. Кол-во сконвертировавшихся в Best в каждый месяц
conversion_by_period AS (
    SELECT 
        cohort_date,
        period_number,
        COUNT(DISTINCT email) AS conversions
    FROM converted
    GROUP BY cohort_date, period_number
)

-- 6. Финальная таблица
SELECT 
    s.cohort_date,
    c.period_number,
    c.conversions,
    s.cohort_size,
    ROUND(c.conversions::NUMERIC / s.cohort_size, 4) AS conversion_rate
FROM cohort_sizes s
LEFT JOIN conversion_by_period c ON s.cohort_date = c.cohort_date
ORDER BY cohort_date, period_number


 """


# In[34]:


df_8_ppc = sqlio.read_sql_query(query8_ppc,conn)
conn.close()


# In[35]:


# Преобразуем дату когорты в формат "YYYY-MM"
df_8_ppc['cohort_month'] = pd.to_datetime(df_8_ppc['cohort_date']).dt.to_period('M').astype(str)

# Строим сводную таблицу: строки — когорты, столбцы — месяцы после когорты
pivot = df_8_ppc.pivot(index='cohort_month', columns='period_number', values='conversion_rate')

# Рисуем тепловую карту
plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True)
plt.title("Конверсия в Best Customers по когортам")
plt.xlabel("Месяц после когорты")
plt.ylabel("Когорта (месяц)")
plt.tight_layout()
plt.show()



query10 = """
WITH first_start AS (
    SELECT DISTINCT ON (email)
        email AS entity_id,
        mth AS start_time,
        medium_group AS group_name
    FROM sandbox.rfm_new_20250930
    ORDER BY email, mth
),
first_event AS (
    SELECT
        email AS entity_id,
        MIN(mth) AS event_time
    FROM sandbox.rfm_new_20250930
    WHERE rfm_segment IN ('About To Sleep', 'At Risk', 'Lost')
    GROUP BY email
),
last_obs AS (
    SELECT
        email AS entity_id,
        MAX(mth) AS last_time
    FROM sandbox.rfm_new_20250930
    GROUP BY email
)
SELECT
    fs.entity_id,
    fs.group_name AS "group",
    fs.start_time,
    fe.event_time,
    lo.last_time,
    -- duration в месяцах
    COALESCE(
        (EXTRACT(YEAR FROM fe.event_time) - EXTRACT(YEAR FROM fs.start_time)) * 12 +
        (EXTRACT(MONTH FROM fe.event_time) - EXTRACT(MONTH FROM fs.start_time)),
        (EXTRACT(YEAR FROM lo.last_time) - EXTRACT(YEAR FROM fs.start_time)) * 12 +
        (EXTRACT(MONTH FROM lo.last_time) - EXTRACT(MONTH FROM fs.start_time)) + 1
    ) AS duration,
    -- событие 1/0
    CASE WHEN fe.event_time IS NOT NULL THEN 1 ELSE 0 END AS event
FROM first_start fs
LEFT JOIN first_event fe ON fs.entity_id = fe.entity_id
LEFT JOIN last_obs lo ON fs.entity_id = lo.entity_id
"""


# In[37]:


df_10 = sqlio.read_sql_query(query10,conn)
conn.close()


# In[38]:


# Ограничиваем первые 12 месяцев и основные каналы
main_channels = ["Direct", "PPC", "SEO", "Other", "NAG",'EMAIL']
df_12 = df_10[(df_10["duration"] <= 12) & (df_10["group"].isin(main_channels))]

plt.figure(figsize=(8,5))

for g, sub in df_12.groupby("group"):
    # Считаем события и число находящихся под наблюдением по месяцам
    survival = sub.groupby("duration")["event"].agg(["count", "sum"]).sort_index()
    survival["at_risk"] = survival["count"].iloc[::-1].cumsum()
    survival["surv_prob"] = (1 - survival["sum"] / survival["at_risk"]).cumprod()
    
    plt.step(survival.index, survival["surv_prob"], where="post", label=str(g))

plt.xlabel("Месяцы")
plt.ylabel("Вероятность оставаться активным (S(t))")
plt.title("Kaplan–Meier кривые по основным каналам (первые 12 месяцев)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




