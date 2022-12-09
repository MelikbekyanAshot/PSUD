import streamlit as st
import pandas as pd
from analysis.analytics import Analytics
from data.constants import START_DATE, END_DATE
import seaborn as sns
import datetime

excel_table = pd.read_excel('data/База данных.xlsx',
                            sheet_name=[0, 1, 2, 3, 4, 5])
analytics = Analytics(excel_table)

start_date_col, end_date_col = st.columns(2)
with start_date_col:
    start_date = pd.to_datetime(st.date_input('Начало',
                                              value=START_DATE,
                                              min_value=START_DATE,
                                              max_value=END_DATE))
with end_date_col:
    end_date = pd.to_datetime(st.date_input('Конец',
                                            value=END_DATE,
                                            min_value=START_DATE,
                                            max_value=END_DATE))

abc_tab, metrics_tab, predict_tab = st.tabs(['ABC анализ', 'Метрики', 'Прогнозирование'])
with abc_tab:
    col1, col2 = st.columns(2)
    with col1:
        fig = analytics.plot_pareto_chart(start_date, end_date)
        st.pyplot(fig)

    with col2:
        fig = analytics.brief_pie_chart(start_date, end_date)
        st.plotly_chart(fig, use_container_width=True)

with metrics_tab:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        summary_profit = analytics.summary_metrics(start_date, end_date)
        st.metric(label='Суммарная прибыль', value=summary_profit)

    with col2:
        sales_amount = analytics.sales_number(start_date, end_date)
        st.metric(label='Количество заказов', value=sales_amount)

    with col3:
        st.metric(label='Потрачено на доставку',
                  value=analytics.spent_on_delivery(start_date, end_date))

    with col4:
        st.metric(label='Количество клиентов',
                  value=analytics.customer_number(start_date, end_date))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(analytics.rank_managers(start_date, end_date), use_container_width=True)
    with col2:
        st.plotly_chart(analytics.sales_by_customer_segment(start_date, end_date), use_container_width=True)

    st.plotly_chart(analytics.rank_customers(start_date, end_date))

with predict_tab:
    profit_plot = analytics.plot_sale_amount()
    st.bokeh_chart(profit_plot, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        level = st.selectbox('Выберите уровень представления',
                             ['Категория продукта', 'Подкатегория продукта', 'Продукт'])
    with col2:
        items = st.multiselect('Выберите позиции',
                               ['Technology', 'Office Supplies', 'Furniture'])
    # fig = analytics.sales_by_days(level=level, items=items)
    # st.plotly_chart(fig, use_container_width=True)
