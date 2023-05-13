import streamlit as st


from utils import load_data
from utils import descriptive_stats, distribution_plot, feature_stats_plot


df, x, y = load_data()
feature_names = [col for col in df.columns if col != 'wine-type']

st.header('Dataset stats 📊')
data_stats = df.describe().loc[['min', 'mean', 'max', 'std']]
st.write(data_stats)

tab_min, tab_mean, tab_max, tab_std = st.tabs(["min", "mean", 'max', 'std'])


with tab_min:
    fig_min = descriptive_stats(data_stats, stat='min', title="Minimum")
    st.pyplot(fig=fig_min, use_container_width=True)

with tab_mean:
    fig_mean = descriptive_stats(data_stats, stat='mean', title="Average")
    st.pyplot(fig=fig_mean, use_container_width=True)

with tab_max:
    fig_max = descriptive_stats(data_stats, stat='max', title="Maximum")
    st.pyplot(fig=fig_max, use_container_width=True)

with tab_std:
    fig_std = descriptive_stats(
        data_stats, stat='std', title="Standard deviation of")
    st.pyplot(fig=fig_std, use_container_width=True)


st.header('Distribution 📊')
selected_feature = st.selectbox(
    label="Select the ingredient to see the distribution",
    options=feature_names)
distribution_fig = distribution_plot(df, selected_feature)
st.pyplot(fig=distribution_fig, use_container_width=True)

feat_stats = (
    df[[f'{selected_feature}', 'wine-type']]
    .groupby(by=['wine-type']).agg(['min', 'mean', 'max'])
)
feature_col1, feature_col2 = st.columns(2)
with feature_col1:
    feat_fig_1 = feature_stats_plot(feat_stats)
    st.pyplot(fig=feat_fig_1)
with feature_col2:
    feat_fig_2 = feature_stats_plot(feat_stats, stacked=True)
    st.pyplot(fig=feat_fig_2)

st.header('Wine Dataset 📑')
st.write(df)
