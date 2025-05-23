# Library imports
import traceback
import copy

import streamlit as st
import pandas as pd
import os

import utils.data_fetcher as dataFetcher
import modules.visuals as visuals
import modules.machine_learning as ml


from utils.page_components import (
    add_common_page_elements,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.header("Predicting xG Chance with XGBoost", divider=True)
st.text("Getting the data")

# def get_cached_matches(competition_id, big_chance_threshold, data_dir='data'):
#     filename = f"{data_dir}/matches_comp_{competition_id}_thresh_{big_chance_threshold}.csv"
#     if os.path.exists(filename):
#         return pd.read_csv(filename)
#     else:
#         matches = dataFetcher.get_all_teams_matches(competition_id, big_chance_threshold)
#         matches.to_csv(filename, index=False)
#         return matches

def get_cached_matches(competition_id, data_dir='data'):
    filename = f"{data_dir}/matches_comp_{competition_id}.pkl"
    
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    else:
        matches = dataFetcher.get_all_teams_matches(competition_id)
        matches.to_pickle(filename)

        matches1 = pd.read_pickle(filename)
        return matches1


def main():
     # Initialize session state variables
    for key in ["selected_competition_name", "selected_team_name", "selected_match_name"]:
        if key not in st.session_state:
            st.session_state[key] = None
    
    competitions = dataFetcher.get_competition_teams_matches()
    st.dataframe(competitions[["competition_name", "season_name","competition_id", "season_id", "matches", "teams count", "teams"]])
    st.text(f"Total Competitions: {len(competitions)}, Total Teams: {competitions['teams count'].sum()}, Total Matches: {competitions['matches'].sum()}")

    for competition_id in competitions['competition_id']:
        matches_all = get_cached_matches(competition_id)
        
    
    # Step 1: User selects competition
    selected_competition_name = st.selectbox("Competition", competitions["competition_name"].unique())
    comp = competitions.query('competition_name == @selected_competition_name')

    # Step 2: Combine team lists and remove duplicates
    # Flatten the list of lists
    
    
    all_teams = [team for sublist in comp["teams"] for team in sublist]
    # unique_teams = list(all_teams)
      
    st.subheader(f"Selected Competition: {selected_competition_name}")

    
     # 📏 Add threshold slider for big chance definition
    big_chance_threshold = st.slider(
        "Define what xG value counts as a Big Chance: (Changing from 0.2 will need new data to be fetched)",
        min_value=0.1,
        max_value=0.99,
        value=0.2,
        step=0.01,
        help="Bins with xG above this threshold will be considered a big chance"
    )
    st.text(f"Big Chance Threshold: {big_chance_threshold}")
    
    # matches = dataFetcher.get_teams_matches(comp.competition_id.iloc[0], selected_team_name)
    # matches = dataFetcher.get_all_teams_matches(comp.competition_id.iloc[0], big_chance_threshold)

    comp_id = comp.competition_id.iloc[0]
    matches = get_cached_matches(comp_id)
    matches['big_chance_for_next_10'], matches['big_chance_against_next_10'] = dataFetcher.add_future_big_chance_labels_per_bin(matches['binned_xg_for'], matches['binned_xg_against'], threshold=big_chance_threshold)

    
    # Step 3: Team selectbox
    selected_team_name = st.selectbox("🎯 Select a team to see their stats:", sorted(all_teams))  # sort for easier UX
    # team_comp_rows = comp[comp["teams"].apply(lambda team_list: selected_team_name in team_list)]
    # team_comp_rows_str = ", ".join(
    #                         [f"{comp_row['competition_name']} {comp_row['season_name']}" for _, comp_row in team_comp_rows.iterrows()]
    #                         ) if not team_comp_rows.empty else ""
    
    # Lists of columns
    binned_columns = [
        ("binned_xg_for", "binned_xg_against"),
        ("binned_final_third_passes_for", "binned_final_third_passes_against"),
        ("binned_box_passes_for", "binned_box_passes_against"),
        ("binned_final_third_carries_for", "binned_final_third_carries_against"),
        ("binned_box_carries_for", "binned_box_carries_against")
    ]

    rolling_columns = [
        ("rolling_xg_for", "rolling_xg_against"),
        ("rolling_final_third_passes_for", "rolling_final_third_passes_against"),
        ("rolling_box_passes_for", "rolling_box_passes_against"),
        ("rolling_final_third_carries_for", "rolling_final_third_carries_against"),
        ("rolling_box_carries_for", "rolling_box_carries_against")
    ]  
    
    st.text(f"Selected Team: {selected_team_name}")
    
    team_matches = matches[matches["team"] == selected_team_name]
    # Flatten binned_columns and rolling_columns
    binned_columns_flat = [col for pair in binned_columns for col in pair]
    rolling_columns_flat = [col for pair in rolling_columns for col in pair]

    # Combine all columns to display
    columns_to_display = [
        "match_id", "season", "opponent", "match_date", "team_score", "opponent_score", "venue",
        "match_time", "goals_scored", "goals_conceded", "shots_attempted", "shots_conceded",
        "big_chance_for_next_10", "big_chance_against_next_10"
    ] + binned_columns_flat + rolling_columns_flat

    # Display the DataFrame
    st.dataframe(
        team_matches[columns_to_display],
        column_config={
            "goals_scored": st.column_config.JsonColumn(width="large"),
            "goals_conceded": st.column_config.JsonColumn(width="large"),
            "shots_attempted": st.column_config.JsonColumn(width="large"),
            "shots_conceded": st.column_config.JsonColumn(width="large")
        },
        hide_index=True
    )
        
    st.text(f"Total Matches: {len(team_matches)}, Total goals scored: {team_matches['team_score'].sum()}, Total Goals Conceded: {team_matches['opponent_score'].sum()}, Minutes Played: {matches['match_time'].sum()}")

    st.subheader("Match Events Graphs")
    selected_match_name = st.selectbox("🎯 Select a match to see their events:", team_matches['match_summary'])
    # Get the corresponding row from matches
    selected_match = matches.query('match_summary == @selected_match_name')
    selected_match = matches[matches['match_summary'] == selected_match_name].iloc[0]
    
    selected_match.index.name = 'bin'

    goals_scored_df = pd.DataFrame(selected_match['goals_scored'])
    goals_conceded_df = pd.DataFrame(selected_match['goals_conceded'])
    
    def clean_title(col_name):
        return col_name.replace("_for", "").replace("_", " ").title()
    
    # Create Binned Tabs
    binned_tabs = st.tabs([clean_title(f) for f, _ in binned_columns])
    for i, (f_col, a_col) in enumerate(binned_columns):
        with binned_tabs[i]:
            visuals.plot_xg_histograms(
                binned_for=selected_match[f_col],
                binned_against=selected_match[a_col],
                goals_scored=goals_scored_df,
                goals_conceded=goals_conceded_df,
                bin_width=10,
                for_name=f_col,
                against_name=a_col
            )
    
    rolling_tabs = st.tabs([clean_title(f) for f, _ in rolling_columns])    
    for i, (f_col, a_col) in enumerate(rolling_columns):
        with rolling_tabs[i]:
            visuals.plot_rolling_xg(
                rolling_for=selected_match[f_col],
                rolling_against=selected_match[a_col],
                goals_scored=goals_scored_df,
                goals_conceded=goals_conceded_df,
                bin_width=10,
                for_name=f_col,
                against_name=a_col
            )
            
    
    # model_for, model_against = ml.train_big_chance_prediction_models(matches)
    ml.train_big_chance_prediction_models(matches)
    
if __name__ == "__main__":
    main()