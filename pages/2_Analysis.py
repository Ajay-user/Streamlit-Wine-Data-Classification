
import streamlit as st

from utils import load_data
from utils import feature_correlation_heatmap, feature_scatter_plot, feature_aggregation_plot, hexbin_plot

df, x, y = load_data()
ingredients = [col for col in df.columns if col != 'wine-type']


tab_1, tab_2, tab_3 = st.tabs(['Correlation', 'Scatter plot', 'Aggregation'])

with tab_1:
    st.subheader('Correlation between features')
    st.write(x.corr())

    st.divider()
    corr_fig = feature_correlation_heatmap(x)
    st.pyplot(fig=corr_fig, use_container_width=True)


with tab_2:
    st.markdown('Explore relationship between ingredients 🔎')

    xaxis = st.selectbox(label='X-axis', options=ingredients, index=0)
    yaxis = st.selectbox(label='Y-axis', options=ingredients, index=1)

    color_encode = st.checkbox(label='Color encode 🌈')

    scatter_fig = feature_scatter_plot(x, y, xaxis, yaxis, color_encode)
    st.pyplot(fig=scatter_fig, use_container_width=True)

    st.divider()

    mincnt = st.number_input(
        label='minimum count', min_value=0, step=1, value=0)

    hexbin_fig = hexbin_plot(df, xaxis, yaxis, mincnt)
    st.pyplot(fig=hexbin_fig, use_container_width=True)


with tab_3:
    st.subheader('Aggregate of ingredients across different wine class')
    agg_col1, agg_col2 = st.columns(2)
    with agg_col1:
        agg_fn = st.radio(
            label="Choose the aggregation function",
            options=['min', 'max', 'mean', 'median', 'sum', 'count', 'var', 'std', 'sem'], index=2)
        feature = st.selectbox(label='Feature you want to aggregate',
                               options=ingredients, index=0)
        agg_df = df[[f'{feature}', 'wine-type']
                    ].groupby(by='wine-type').agg(agg_fn)[feature]
    with agg_col2:
        pie_fig = feature_aggregation_plot(agg_df, feature, agg_fn, True)
        st.pyplot(fig=pie_fig, use_container_width=True)

    st.divider()
    agg_fig = feature_aggregation_plot(agg_df, feature, agg_fn)
    st.pyplot(fig=agg_fig, use_container_width=True)
