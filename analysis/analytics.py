import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from abc_classification.abc_classifier import ABCClassifier
from abc_classification.abc_visualiser import pareto_chart
from bokeh.plotting import figure
from bokeh.palettes import Spectral4

import plotly.express as px


class Analytics:
    def __init__(self, excel_data: dict[int, pd.DataFrame]):
        self.sales_df = excel_data[0]
        self.delivery_df = excel_data[1]
        self.dates_df = excel_data[2]
        self.products_df = excel_data[3]
        self.managers_df = excel_data[5]
        self.preprocess_dataframe()

    def preprocess_dataframe(self):
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].str.replace(',', '.')
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].str.replace(' ', '')
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].astype(float)
        self.dates_df['Дата'] = pd.to_datetime(self.dates_df['Дата'])

    def rank_managers(self, start_date, end_date):
        df = self.__join(self.sales_df, self.managers_df, 'Регион')
        df = self.__join(df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        df = df.groupby('Менеджер').agg(
            total_profit=('Прибыль', 'sum'),
            sales_amount=('Продукт', 'count')
        )
        df.columns = ['Суммарная прибыль', 'Количество продаж']
        df.index.name = 'Менеджер'
        return df

    def spent_on_delivery(self, start_date, end_date):
        df = self.__join(self.sales_df, self.delivery_df, 'Метод доставки')
        df = self.__join(df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        spent = df['Стоимость доставки'].sum()
        return spent

    def brief_pie_chart(self, start_date, end_date):
        _, brief = self.__abc_analysis(start_date, end_date)
        brief = brief
        fig = px.pie(brief,
                     values='Прибыль', names='class',
                     title='Краткая сводка', )
        fig.update_layout(title_x=0.5)
        return fig

    def plot_pareto_chart(self):
        pass

    def __abc_analysis(self, start_date, end_date):
        df = self.sales_df \
            .set_index('Order_ID') \
            .join(self.dates_df.set_index('Order_ID'))
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        abc_clf = ABCClassifier(df)
        classified = abc_clf.classify(abc_column='Продукт',
                                      criterion='Прибыль')
        brief = abc_clf.brief_abc(classified)
        return classified, brief

    def plot_profit(self):
        df = self.sales_df \
            .set_index('Order_ID') \
            .join(self.dates_df.set_index('Order_ID'))
        df = df.groupby(by='Дата').sum()
        p = figure(title='Прибыль',
                   x_axis_label='Дата',
                   y_axis_label='Дневная прибыль',
                   x_axis_type='datetime')
        p.line(df.index, df['Прибыль'], line_width=2)
        return p

    def summary_metrics(self, start_date, end_date):
        df = self.sales_df \
            .set_index('Order_ID') \
            .join(self.dates_df.set_index('Order_ID'))
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        return round(sum(df['Прибыль']))

    def sales_number(self, start_date, end_date):
        df = self.__join(self.sales_df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        return len(df)

    @st.cache
    def sales_by_days(self, level: str, items: str):
        df = self.__join(self.sales_df, self.products_df, 'Продукт')
        df = self.__join(df, self.dates_df, 'Order_ID')
        df = df[['Дата', level, 'Количество']]
        df = df[df[level].isin(items)]
        fig = px.line(data_frame=df, x='Дата', y='Количество', title='A')
        fig.update_layout(title_x=0.5)
        return fig

    def __join(self, df1: pd.DataFrame, df2: pd.DataFrame, by: str):
        return df1 \
            .set_index(by) \
            .join(df2.set_index(by))
