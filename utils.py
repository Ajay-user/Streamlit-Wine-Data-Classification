import pathlib

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import scikitplot as skplot


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import streamlit as st


@st.cache_data
def get_data_description():
    desc = pathlib.Path(
        './data/wine-data-decription.txt').read_text(encoding='utf-8')
    return desc


@st.cache_data
def load_data():
    df = pd.read_csv('./data/wine-data.csv')
    x = df.drop(columns=['wine-type'])
    y = df['wine-type']
    return df, x, y


#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PLOT UTILS

def descriptive_stats(df, stat, title):
    with plt.style.context(style='fivethirtyeight'):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        data_stat = df.loc[stat]
        data_stat.plot(
            kind='barh', title=f"{title} ingredient content", ax=ax)
        for col in df.columns:
            ax.scatter(x=data_stat[col], y=col)
            ax.scatter(
                x=data_stat[col], y=col, marker=f'$--->{data_stat[col]:0.2f}$', s=2000, c='k')
    return fig


def distribution_plot(df, feature):
    with plt.style.context(style='fivethirtyeight'):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        df[feature].plot(
            kind='density', ax=ax,
            title=f'Density plot of {feature}')
    return fig


def feature_stats_plot(df, stacked=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    df.plot(kind='bar', ax=ax, stacked=stacked)
    if not stacked:
        for i, row in enumerate(df.values):
            ax.scatter(x=float(i-0.15), y=row[0]+0.3,
                       c='k', marker=f'${row[0]:0.2f}$', s=600)
            ax.scatter(x=i, y=row[1]+0.3, c='k',
                       marker=f'${row[1]:0.2f}$', s=600)
            ax.scatter(x=float(i+0.15), y=row[2]+0.3,
                       c='k', marker=f'${row[2]:0.2f}$', s=600)
    ax.legend(bbox_to_anchor=(1.05, 1))
    return fig


def feature_correlation_heatmap(X):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    sns.heatmap(X.corr(), annot=True, ax=ax)
    return fig


def feature_scatter_plot(x, y, xaxis, yaxis, color_encode):
    with plt.style.context(style="fivethirtyeight"):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        color_dict = {'class_0': 'red',
                      'class_1': 'blue', 'class_2': 'green'}
        x.plot(x=xaxis, y=yaxis, kind='scatter', alpha=0.5,
               ax=ax, edgecolor='k', color=list(y.map(color_dict)) if color_encode else 'k')

    c0 = lines.Line2D([], [], color='red', marker='.', ls='', label='class_0')
    c1 = lines.Line2D([], [], color='blue', marker='.', ls='', label='class_1')
    c2 = lines.Line2D([], [], color='green', marker='.',
                      ls='', label='class_2')
    ax.legend(handles=[c0, c1, c2])
    return fig


def hexbin_plot(df, xaxis, yaxis, mincnt):
    class_dict = {
        'class_0': 0,
        'class_1': 1,
        'class_2': 2
    }

    fig, ax = plt.subplots(nrows=1, ncols=1)
    df['wine-type'] = df['wine-type'].map(class_dict).astype(int)
    df.plot(kind='hexbin', x=xaxis, y=yaxis, gridsize=(15, 16),
            ax=ax, edgecolor='k', C='wine-type',
            cmap='winter', mincnt=mincnt, vmin=mincnt,
            title=f'Hexbin plot {xaxis} vs {yaxis}',
            )
    df['wine-type'] = df['wine-type'].map(
        {v: k for k, v in class_dict.items()}).astype(str)
    return fig


def feature_aggregation_plot(df, feature, fn, pie_chart=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if not pie_chart:
        df.plot(kind='bar', color=[
            'salmon', 'lightblue', 'seagreen'], ax=ax)

        df.plot(
            linestyle='', marker='o', markersize=50, alpha=0.5, ax=ax)

        for k, v in df.items():
            ax.scatter(k, v, marker=f'${v :0.2f}$', s=500, c='k')

        ax.set(
            title=f'{fn.capitalize()} of {feature} content across different wine class')
    else:
        df.plot(
            kind='pie', subplots=True, autopct='%1.1f%%',
            labeldistance=1.2, ax=ax, ylabel='',
            explode=[0.01, 0.01, 0.01], colors=['salmon', 'lightblue', 'seagreen'])
    return fig


def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skplot.metrics.plot_confusion_matrix(
        y_true=y_test, y_pred=y_pred, title='Confusion Matrix', ax=ax)
    return fig


def plot_feature_importance(model, feature_names):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skplot.estimators.plot_feature_importances(
        clf=model, feature_names=feature_names, ax=ax, x_tick_rotation=90)
    return fig


def get_classification_report(y_test, y_pred):
    return classification_report(y_true=y_test, y_pred=y_pred)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
