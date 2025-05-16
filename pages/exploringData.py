#Setup code
#python -m venv env -> Create a virutal environment
#source env/bin/activate -> Activate the virtual environment
#pip install -r requirements.txt -> Install the packages
#streamlit run app.py -> Run the app
import streamlit as st
import pandas as pd
import numpy as np
import time
from statsbombpy import sb
from requests.exceptions import HTTPError
import modules.functions as function
from requests_cache import install_cache

install_cache("statsbomb_cache", backend="sqlite", expire_after=3600)  # Cache expires after 1 hour

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Statsbomb Chart App", page_icon=":soccer:")
st.title('Data Basics Masterclass test')
st.caption('Made by Ana Beatriz Macedo')

def fetch_with_retries(fetch_function, retries=5, **kwargs):
    for i in range(retries):
        try:
            return fetch_function(**kwargs)
        except HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** i  # Exponential backoff
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
    st.error(f"Failed to fetch data after {retries} retries.")
    return pd.DataFrame()

# Cached API calls
@st.cache_data
def get_competitions():
    return fetch_with_retries(sb.competitions)

@st.cache_data
def get_matches(competition_id, season_id):
    return fetch_with_retries(sb.matches, competition_id=competition_id, season_id=season_id)

@st.cache_data
def get_match_events(match_id):
    return fetch_with_retries(sb.events, match_id=match_id)

# Main function
def main():
    # Initialize session state variables
    for key in ["selected_player_filter", "selected_match_name", "selected_team_filter"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Step 1: Get competitions data
    competitions = get_competitions()
    st.sidebar.header("Statsbomb Open Source Data")
    st.subheader("Competitions Table")
    st.dataframe(competitions)

    selected_competition_name = st.sidebar.selectbox("Competition", competitions["competition_name"].unique())
    comp = competitions.query('competition_name == @selected_competition_name')
    comp_id = comp.competition_id.iloc[0]

    selected_year_name = st.sidebar.selectbox("Season", comp["season_name"])
    season = comp.query('season_name == @selected_year_name')
    season_id = season.season_id.iloc[0]

    # Step 2: Get matches data
    matches = get_matches(competition_id=comp_id, season_id=season_id)
    st.subheader("Matches Table")
    st.dataframe(matches)

    st.sidebar.subheader("Matches available")
    matches['Game'] = matches['home_team'] + ' x ' + matches['away_team']
    selected_match_name = st.sidebar.selectbox("Game", matches["Game"])
    game = matches.query('Game == @selected_match_name')
    match_id = game.match_id.iloc[0]

    # Step 3: Get and process match events
    match_events = function.process_match_events(get_match_events(match_id))
    st.subheader("Match Events")
    st.dataframe(match_events)
    st.text(match_events.columns)
    st.text(match_events['type'].unique())
        
    # Step 4: Display charts
    st.subheader('Charts Area')
    st.write('With the "Plot Function" on the left, please select the chart of your choice to appear on the screen.')

    st.session_state.selected_competitions = competitions
    st.session_state.selected_matches = matches
    st.session_state.selected_match_events = match_events

    function_names = ['Team Shot Chart', 'Player Heatmap', 'Passing Network', 'xG Evolution']
    selected_function = st.sidebar.selectbox("Plot Function", function_names)

    if selected_function == 'Player Heatmap':
        selected_player = st.selectbox("Select Player", match_events["player"].unique())
        st.session_state.selected_player_filter = selected_player
        fig = function.heatmap(match_events, selected_player)
        st.pyplot(fig)

    if selected_function == 'Team Shot Chart':
        selected_team = st.selectbox("Select Team", match_events["team"].unique())
        st.session_state.selected_player_filter = selected_team
        fig1, fig2, fig3 = function.teamShotChart(match_events, selected_team)
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)

    if selected_function == 'xG Evolution':
        fig = function.xGEvolution(match_events)
        st.pyplot(fig)

    if selected_function == 'Passing Network':
        selected_team = st.selectbox("Select Team", match_events["team"].unique())
        st.session_state.selected_player_filter = selected_team
        fig = function.pass_network(selected_team, match_events)
        st.pyplot(fig)
        st.table(function.ball_passer_receiver(selected_team, match_events))

if __name__ == "__main__":
    main()
