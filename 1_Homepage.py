import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(layout='wide')

st.sidebar.success('Select a page to get started!')

# add centered title
st.markdown("""
    <style>
    .custom-title {
        text-align: center;
        color: white;
        font-size: 60px;
        font-weight: bold;
        background: linear-gradient(90deg, #2c3333 30%, #00072b 100%);
        padding: 20px;
        border-radius: 20px;
        margin: 0 auto;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-title">Dunkstradamus</div>', unsafe_allow_html=True)
st.write('')

# add photo
st.image('images/dunkstradamus.png', use_column_width=True)

#st.title('Dunkstradamus Predictions')


st.write('''
Welcome to the home of Dunkstradamus! \n

Dunkstradamus comes in two apps -- there is the [Dunsktradamus [Simple]](https://tmcroyce-multipage-streamlit-app-v2-3-1-homepage-jp4gxf.streamlit.app/Dunkstradamus_Simple) app --  a user-friendly application powered by a machine learning model to forecast NBA game outcomes. Continuously updated, the model is trained on data from the 2016 season up until the present and undergoes daily retraining to ensure accurate predictions. The Simple app is perfect for users seeking a quick glance at the day's game predictions.

For those who desire a more comprehensive analysis, the [Dunkstradamus [Extended]](https://tmcroyce-multipage-streamlit-app-v2-3-1-homepage-jp4gxf.streamlit.app/Dunkstradamus_Extended) app offers a deeper dive into the machine learning model's predictions. In addition to forecasts, the Extended app provides valuable metrics and insightful exploratory data analysis for each game, offering a richer understanding of the night's matchups.


Select an app to get started!
''')


st.write('Contact: [LinkedIn](https://www.linkedin.com/in/travis-royce/) | [GitHub](https://github.com/tmcroyce) | traviscroyce@gmail.com')

