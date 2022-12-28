import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from abc_classification.abc_classifier import ABCClassifier
from abc_classification.abc_visualiser import pareto_chart
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
from matplotlib.ticker import PercentFormatter
import numpy as np

import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Analytics:
    def __init__(self, excel_data: dict[int, pd.DataFrame]):
        self.sales_df = excel_data[0]
        self.delivery_df = excel_data[1]
        self.dates_df = excel_data[2]
        self.products_df = excel_data[3]
        self.customers_df = excel_data[4]
        self.managers_df = excel_data[5]
        self.preprocess_dataframe()

    def preprocess_dataframe(self):
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].str.replace(',', '.')
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].str.replace(' ', '')
        self.sales_df['Прибыль'] = self.sales_df['Прибыль'].astype(float)
        self.sales_df['Выручка'] = self.sales_df['Выручка'].str.replace(',', '.')
        self.sales_df['Выручка'] = self.sales_df['Выручка'].str.replace(' ', '')
        self.sales_df['Выручка'] = self.sales_df['Выручка'].astype(float)
        self.dates_df['Дата'] = pd.to_datetime(self.dates_df['Дата'], format="%Y%W-%w")

    def rank_managers(self, start_date, end_date):
        df = self.__join(self.sales_df, self.managers_df, 'Регион')
        df = self.__join(df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        df = df.groupby('Менеджер').agg(
            total_profit=('Прибыль', 'sum'),
            sales_amount=('Продукт', 'count')
        )
        df.columns = ['Прибыль', 'Количество продаж']
        df.index.name = 'Менеджер'
        fig = px.bar(df, x=df.index, y='Прибыль', title='Прибыль по менеджерам')
        fig.update_layout(title_x=0.5)
        return fig

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
        fig.update_layout(title_x=0.5,
                          font=dict(   size=36))
        fig.update_layout(legend=dict(font=dict(size=24)),
                          legend_title=dict(font=dict(size=24)))
        return fig

    def plot_pareto_chart(self, start_date, end_date):
        df = self.sales_df \
            .set_index('Order_ID') \
            .join(self.dates_df.set_index('Order_ID'))
        df = self.__join(df, self.products_df, 'Продукт')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        df = df[df['Прибыль'] > 0]
        group_by = 'Подкатегория продукта'
        column = 'Прибыль'
        df = df.groupby(group_by)[column].sum().reset_index()
        df = df.sort_values(by=column, ascending=False)

        df["cumpercentage"] = df[column].cumsum() / df[column].sum() * 100

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.bar(df[group_by], df[column], color="C0")
        ax2 = ax.twinx()
        ax2.plot(df[group_by], df["cumpercentage"], color="C1", marker="D", ms=7)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        ax.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        # plt.show()

        return fig

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

    def plot_sale_amount(self, category: str):
        df = self.sales_df \
            .set_index('Order_ID') \
            .join(self.dates_df.set_index('Order_ID'))
        df = self.__join(df, self.products_df, 'Продукт')
        # st.write(df['Категория продукта'].unique())
        df = df[df['Категория продукта'] == category]
        df = df.groupby(df['Дата']).sum()
        fig = px.scatter(
            df, x=df.index, y='Прибыль', opacity=0.65,
            trendline='ols', trendline_color_override='darkblue'
        )
        fig.show()
        # st.dataframe(df)
        #
        # p = figure(title='Прибыль',
        #            x_axis_label='Дата',
        #            y_axis_label='Дневная прибыль',
        #            x_axis_type='datetime')
        # p.line(df.index, df['Количество'], line_width=2)
        # return p

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

    def customer_number(self, start_date, end_date):
        df = self.__join(self.sales_df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        return df['Покупатель'].nunique()

    def rank_customers(self, start_date, end_date):
        df = self.__join(self.sales_df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        df = df.groupby('Покупатель').sum().sort_values('Прибыль', ascending=True)['Прибыль']
        # st.dataframe(df)
        fig = go.Figure(go.Bar(
            x=df,
            y=df.index,
            orientation='h',
            textfont=dict(
                size=24,
            )
        ))
        fig.update_xaxes(type="log")
        fig.update_layout(
            xaxis_title="Логарифм прибыльности клиентов",
            yaxis_title="Клиенты",
            title="Топ клиентов",
            title_x=0.5
        )
        return fig


    def sales_by_customer_segment(self, start_date, end_date):
        df = self.__join(self.sales_df, self.customers_df, 'Покупатель')
        df = self.__join(df, self.dates_df, 'Order_ID')
        df = df[(start_date <= df['Дата']) & (df['Дата'] <= end_date)]
        df = df.groupby('Сегмент покупателя').agg(
            total_profit=('Прибыль', 'sum'),
            sales_amount=('Продукт', 'count')
        )
        fig = px.bar(df, x=df.index, y=[df.total_profit, df.sales_amount], title='Прибыль по сегментам покупателей',
                     labels={'y': 'Прибыль'})
        fig.update_layout(title_x=0.5)
        return fig

    def __join(self, df1: pd.DataFrame, df2: pd.DataFrame, by: str):
        return df1 \
            .set_index(by) \
            .join(df2.set_index(by))
