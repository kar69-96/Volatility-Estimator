"""
Simple test app to verify Streamlit interactivity works.
"""
import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("Streamlit Interactivity Test")

if st.button("Click Me!"):
    st.success("✓ Button works! Streamlit is interactive.")
    st.balloons()

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

option = st.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {option}")

