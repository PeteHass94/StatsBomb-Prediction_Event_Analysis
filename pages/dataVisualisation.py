# Library imports
import traceback
import copy

import streamlit as st
import pandas as pd

import utils.data_fetcher as dataFetcher
import modules.visuals as visuals

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
    st.text(f"Total Competitions: {len(competitions)}, Total Teams: {competitions['teams count'].sum()}, Total Matches: {competitions['matches'].sum()}")

    # Step 1: User selects competition
    selected_competition_name = st.sidebar.selectbox("Competition", competitions["competition_name"].unique())
    comp = competitions.query('competition_name == @selected_competition_name')

    # Step 2: Combine team lists and remove duplicates
    # Flatten the list of lists
    
    
    all_teams = [team for sublist in comp["teams"] for team in sublist]
    # unique_teams = list(all_teams)
    

    # Step 3: Team selectbox
    selected_team_name = st.sidebar.selectbox("Team", sorted(all_teams))  # sort for easier UX
    team_comp_rows = comp[comp["teams"].apply(lambda team_list: selected_team_name in team_list)]
    team_comp_rows_str = ", ".join(
                            [f"{comp_row['competition_name']} {comp_row['season_name']}" for _, comp_row in team_comp_rows.iterrows()]
                            ) if not team_comp_rows.empty else ""
    st.subheader(f"Selected Team: {selected_team_name}")
    st.text(f"In competitions: {team_comp_rows_str}")

    matches = dataFetcher.get_teams_matches(comp.competition_id.iloc[0], selected_team_name)
    st.dataframe(matches[["match_id", "season", "opponent", "match_date", "team_score", "opponent_score", "venue",
                          "match_time", "goals_scored", "goals_conceded", "shots_attempted", "shots_conceded",
                          "binned_xg_for", "binned_xg_against", 
                          "binned_final_third_passes_for", "binned_final_third_passes_against", "binned_box_passes_for", "binned_box_passes_against",
                          "binned_final_third_carries_for", "binned_final_third_carries_against", "binned_box_carries_for", "binned_box_carries_against",
                          "rolling_xg_for", "rolling_xg_against", 
                          "rolling_final_third_passes_for", "rolling_final_third_passes_against", "rolling_box_passes_for", "rolling_box_passes_against",
                          "rolling_final_third_carries_for", "rolling_final_third_carries_against", "rolling_box_carries_for", "rolling_box_carries_against",
                          "goal_next_10"]]
                    , column_config={"goals_scored": st.column_config.JsonColumn(width="large"),
                                     "goals_conceded": st.column_config.JsonColumn(width="large"),
                                    "shots_attempted": st.column_config.JsonColumn(width="large"),
                                    "shots_conceded": st.column_config.JsonColumn(width="large")
                                    }
                    , hide_index=True)
    st.text(f"Total Matches: {len(matches)}, Total goals scored: {matches['team_score'].sum()}, Total Goals Conceded: {matches['opponent_score'].sum()}, Minutes Played: {matches['match_time'].sum()}")

    st.subheader("Goals Scored Histogram")
    selected_match = matches.iloc[0]  # or user-selected

    selected_match.index.name = 'bin'

    goals_scored_df = pd.DataFrame(selected_match['goals_scored'])
    goals_conceded_df = pd.DataFrame(selected_match['goals_conceded'])

    visuals.plot_xg_histograms(selected_match['binned_xg_for'], selected_match['binned_xg_against'], goals_scored_df, goals_conceded_df)
    visuals.plot_rolling_xg(selected_match['rolling_xg_for'], selected_match['rolling_xg_against'], goals_scored_df, goals_conceded_df)
    
    
if __name__ == "__main__":
    main()