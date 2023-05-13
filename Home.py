import streamlit as st

from utils import get_data_description

desc = get_data_description()


st.title(":violet[Wine Data] :green[Analysis] :bar_chart:")
st.markdown('Wine type classification problem : ')
st.caption('Go to Data and understand different statistics about dataset')
st.caption(
    'Go to Analysis and explore relationship between different ingredients used to create three different types of wine')
st.caption(
    'Go to Inference and make prediction about wine type using different ingredients values')
st.caption(
    'On Inference page you can also how model make its decision')
st.write(desc)
