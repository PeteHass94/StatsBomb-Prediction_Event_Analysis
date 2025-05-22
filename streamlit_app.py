# import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )


"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).

conda create --name streamlit_env
conda activate streamlit_env
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

# Library imports
import traceback
import copy

import streamlit as st


from utils.page_components import (
        add_common_page_elements,
    )
    
sidebar_container = add_common_page_elements()

# displaytext = """## Welcome to Football Analysis on Big Chance Prediction! \n\n"""

# st.markdown(displaytext)

# displaytext = """Big Chances are described as xG Values over 0.38 in a game. See here for more information: [What Are Expected Goals (xG)?](https://statsbomb.com/soccer-metrics/expected-goals-xg-explained/). 

# On here you can see what free data StatsBomb have to offer. 
# And you can use that data to see if big chances in a football match can be predicted using xgboost. 

# """

# st.markdown(displaytext)


# Add a title and introductory text
st.title("⚽ Football Analysis: Predicting Big Chances with Machine Learning")
st.divider()
# Add a brief description
st.markdown("""
Welcome to the **Football Analysis App**! This platform leverages **StatsBomb's free football data** to explore and predict **Big Chances** in football matches using advanced machine learning techniques like **XGBoost**.

---

### What Are Big Chances?
Big Chances are defined as moments in a match with an **Expected Goals (xG) value greater than 0.38**. These are high-probability scoring opportunities that can often determine the outcome of a game.

For more details, check out this resource: [What Are Expected Goals (xG)?](https://statsbomb.com/soccer-metrics/expected-goals-xg-explained/).

---

### What Can You Do Here?
- Explore **StatsBomb's free football data**.
- Analyze match events and visualize key metrics.
- Use machine learning to predict **Big Chances** in football matches.

---

### How to Get Started
1. Use the sidebar to navigate through the app.
2. Explore the data and visualizations.
3. Dive into the machine learning predictions to see how Big Chances are modeled.

---

Enjoy exploring football data and uncovering insights into the game!
""")


