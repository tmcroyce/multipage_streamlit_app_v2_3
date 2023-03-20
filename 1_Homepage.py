import streamlit as st
import pandas as pd
import numpy as np

st.title('Welcome!')
st.sidebar.success('Select a page to get started!')

st.write('''
The [NBA Game Predictions [Simple]](https://tmcroyce-multipage-streamlit-app-v2-3-1-homepage-jp4gxf.streamlit.app/NBA_Game_Predictions_[Simple]) app is a simple app that uses a machine learning model to predict the outcome of NBA games. 
The model is trained on data from the 2016-present NBA seasons, and is retrained daily using the latest data.
The simple app is just meant to view the predictions of the day's games.

The [NBA Game Analyzer Tool](https://tmcroyce-multipage-streamlit-app-v2-3-1-homepage-jp4gxf.streamlit.app/NBA_Game_Analyzer_Tool) is meant to view the Machine Learning model's predictions as well as various metrics and exploratory data analysis for the night's games. 

The [NBA Player Analyzer Tool](https://tmcroyce-multipage-streamlit-app-v2-3-1-homepage-jp4gxf.streamlit.app/NBA_Player_Analyzer_Tool) allows the user to analyze specific players playing at specific positions.
''')


st.write('Contact: [LinkedIn](https://www.linkedin.com/in/travis-royce/) | [GitHub](https://github.com/tmcroyce) | traviscroyce@gmail.com')

