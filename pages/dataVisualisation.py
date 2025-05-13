# Library imports
import traceback
import copy

import streamlit as st
import pandas as pd

import utils.data_fetcher as dataFetcher


from utils.page_components import (
    add_common_page_elements,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.header("StatsBomb py", divider=True)
st.text("Getting the data")



def main():
     # Initialize session state variables
    for key in ["selected_player_filter", "selected_match_name", "selected_team_filter"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # # Step 1: Get competitions data
    # competitions = dataFetcher.get_competitions()
    # st.sidebar.header("Statsbomb Open Source Data")
    # st.subheader("Competitions Table")
    # st.dataframe(competitions)

    # selected_competition_name = st.sidebar.selectbox("Competition", competitions["competition_name"].unique())
    # comp = competitions.query('competition_name == @selected_competition_name')
    # comp_id = comp.competition_id.iloc[0]

    # selected_year_name = st.sidebar.selectbox("Season", comp["season_name"])
    # season = comp.query('season_name == @selected_year_name')
    # season_id = season.season_id.iloc[0]

    # # Step 2: Get matches data
    # matches = dataFetcher.get_matches(competition_id=comp_id, season_id=season_id)
    # st.subheader("Matches Table")
    # st.dataframe(matches)

    # st.sidebar.subheader("Matches available")
    # matches['Game'] = matches['home_team'] + ' x ' + matches['away_team']
    # selected_match_name = st.sidebar.selectbox("Game", matches["Game"])
    # game = matches.query('Game == @selected_match_name')
    # match_id = game.match_id.iloc[0]

    # lineups = dataFetcher.get_match_lineups(game)
    
    # st.dataframe(lineups, column_config={"cards": st.column_config.JsonColumn(width="large")})
    
    competitions = dataFetcher.get_competition_teams_matches()
    st.dataframe(competitions[["competition_name", "season_name","competition_id", "season_id", "matches", "teams count", "teams"]])

if __name__ == "__main__":
    main()