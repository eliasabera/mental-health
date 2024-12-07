import streamlit as st

st.title('A Random App')
st.write('Look at the pretty waves')

with open('css.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)