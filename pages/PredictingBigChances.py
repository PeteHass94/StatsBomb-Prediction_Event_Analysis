# Library imports
import traceback
import copy

# Core libraries for data manipulation, UI rendering, and project-specific modules.

import streamlit as st
import pandas as pd
import os
import time

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

st.header("Big Chance Predictor", divider=True)
st.subheader("Predicting xG Chance with XGBoost")
st.markdown(
    """
Welcome to the **Big Chance Predictor** - an interactive football analytics app powered by **StatsBomb Open Data** and **XGBoost**.

This tool breaks matches into 10-minute segments, recalculating time to run from **0 to 100 minutes** instead of two halves. 
Each segment is analyzed for attacking momentum using binned and rolling stats like:
- Expected Goals (xG)
- Final third passes and carries
- Box passes and carries

You can:
- 🧠 Train models to predict big chances *before they happen*
- ⚽ Explore individual matches through shot maps and carry/passing plots
- 🔍 Customize thresholds and time intervals for your analysis

Use the sidebar to begin by selecting a competition and team!
    """
)

# Loads preprocessed match data from disk for performance. If not found, triggers fetch + save.
# Useful when changing thresholds or rerunning sessions without re-fetching large JSON files.


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
    
    # Ensures selected dropdown values persist across Streamlit interactions for UX consistency.
    
     # Initialize session state variables
    for key in ["selected_competition_name", "selected_team_name", "selected_match_name"]:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Overview table showing all competitions, number of matches, and teams. 
    # Helps the user grasp available data coverage before diving in.

    
    competitions = dataFetcher.get_competition_teams_matches()
    competitions_summary = f"Total Competitions: {len(competitions)}, Total Teams: {competitions['teams count'].sum()}, Total Matches: {competitions['matches'].sum()}"

    st.expandable_text = st.expander(f"See selectable competitions and their teams - {competitions_summary}")
    with st.expandable_text:
        st.dataframe(competitions[["competition_name", "season_name","competition_id", "season_id", "matches", "teams count", "teams"]])
        st.markdown("""
        **Data Sourced:** StatsBomb Open Data for the 2015/16 season of the Big 5 European Leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).
        [The 2015/16 Big 5 Leagues Free Data](https://statsbomb.com/news/the-2015-16-big-5-leagues-free-data-release-la-liga/).
        """)
    
    for competition_id in competitions['competition_id']:
        matches_all = get_cached_matches(competition_id)
        
    

    # Step 1: User selects competition
    selected_competition_name = st.sidebar.selectbox("🎯 Select a Competiton for its analysis:", competitions["competition_name"].unique())
    comp = competitions.query('competition_name == @selected_competition_name')

    # Step 2: Combine team lists and remove duplicates
    # Flatten the list of lists
    
    # Flattens and deduplicates teams from selected competition — used for filtering match data later.

    
    all_teams = [team for sublist in comp["teams"] for team in sublist]
    # unique_teams = list(all_teams)
      
    st.subheader(f"Selected Competition: `{selected_competition_name}`")
    competition_summary = f"Total Matches: {comp['matches'].iloc[0]}, Total Teams: {comp['teams count'].iloc[0]}"
    st.text(competition_summary)
    
    
    # Allows analysts to adjust what qualifies as a “big chance” (based on xG). 
    # Influences target labels in model training.
    st.markdown("""
    ### Define Big Chance Threshold
    >Adjust the xG threshold to define what counts as a Big Chance. This will change how matches are labeled and analyzed.
    
    >The default is 0.2, but you can set it to any value between 0.1 and 0.99.
    >Changing this will require re-fetching the data.
    """)
    
     # 📏 Add threshold slider for big chance definition
    big_chance_threshold = st.slider(
        "Define what xG value counts as a Big Chance:",
        min_value=0.1,
        max_value=0.99,
        value=0.2,
        step=0.01,
        help="Bins with xG above this threshold will be considered a big chance"
    )
    st.write("Big Chance Threshold:", big_chance_threshold)
    
    # matches = dataFetcher.get_teams_matches(comp.competition_id.iloc[0], selected_team_name)
    # matches = dataFetcher.get_all_teams_matches(comp.competition_id.iloc[0], big_chance_threshold)

    # Applies future label logic: For each bin in a match, does a big chance occur in the next 10 mins?
    # This helps us turn match time series into predictive supervised learning examples.

    
    comp_id = comp.competition_id.iloc[0]
    matches = get_cached_matches(comp_id)
    matches[['big_chance_for_next_10', 'big_chance_against_next_10']] = matches.apply(
        lambda row: pd.Series(
            dataFetcher.add_future_big_chance_labels_per_bin(
                row['binned_xg_for'],
                row['binned_xg_against'],
                threshold=big_chance_threshold
            )
        ),
        axis=1
    )

    # Filters full competition dataset to only include matches from the selected team.
    
    # Step 3: Team selectbox
    selected_team_name = st.sidebar.selectbox("🎯 Select a team to see their stats:", sorted(all_teams))  # sort for easier UX
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
    
    st.subheader(f"Selected Team: `{selected_team_name}`")
    
    # Display of relevant match-level stats including binned/rolling features and target variables.
    # Helps analysts spot trends across matches and bins.

    
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

    matches_summary = f"Total Matches: {len(team_matches)}, Total goals scored: {team_matches['team_score'].sum()}, Total Goals Conceded: {team_matches['opponent_score'].sum()}, Minutes Played: {team_matches['match_time'].sum()}"
    
    st.expandable_matches = st.expander(f"See selected team matches - {matches_summary}")
    with st.expandable_matches:    
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
        
    
    # For each pair of features (e.g., xG for/against), plots histograms and rolling trends.
    # These are essential for understanding feature distributions used by the model.

    st.subheader("Match Events Graphs")
    selected_match_name = st.selectbox("🎯 Select a match to see their events:", team_matches['match_summary'])
    st.write("Selected Match:", f"`{selected_team_name} {selected_match_name}`")
    
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
    
    # For each pair of features (e.g., xG for/against), plots histograms and rolling trends.
    # These are essential for understanding feature distributions used by the model.
    
    st.subheader("Match Events Visualizations")
    
    # Load and process events
    selected_match_events = dataFetcher.process_match_events(dataFetcher.get_match_events(selected_match['match_id']))
    
    # Create time bins
    time_bins = [(i, i + 10) for i in range(0, 100, 10)]
    time_labels = [f"{start}-{end}" for start, end in time_bins]

    # Assign a time label column (if not already done)
    selected_match_events['bin_label'] = selected_match_events['timeValue'].apply(
        lambda x: f"{int(x // 10 * 10)}-{int((x // 10 + 1) * 10)}" if pd.notna(x) else None
    )

    # Lets users drill into a single match and filter events by 10-min intervals — crucial for in-depth scouting.
    
    # UI: Let user select time bins
    selected_bins = st.multiselect(
        "Filter by 10-minute time intervals:",
        options=time_labels,
        default=time_labels,
        key="time_bin_selector"
    )
    
    filtered_match_events = selected_match_events[selected_match_events['bin_label'].isin(selected_bins)]    
    
    event_tabs = st.tabs(["Shots", "Passes Final Third", "Passes Box", "Carries Final Third", "Carries Box"])
    shots_df = filtered_match_events[filtered_match_events['type'] == 'Shot']
    passes_df = filtered_match_events[filtered_match_events['type'] == 'Pass']    
    carries_df = filtered_match_events[filtered_match_events['type'] == 'Carry']
    
    team_name = selected_match['team']

    with st.spinner("Creating Visuals...", show_time=True):
        time.sleep(15)
    
    # Uses StatsBomb events (passes, shots, carries) to generate pitch maps, split by type and zone.
    # Good for storytelling around tactical phases and player involvement.
    
    with event_tabs[0]:
        st.subheader("Shot Map")
        visuals.plot_shot_map(shots_df, team_name=team_name)

    with event_tabs[1]:
        st.subheader("Passes in Final Third (Completed Passes)")
        visuals.plot_pass_map(passes_df, team_name=team_name, pass_type="Final Third")

    with event_tabs[2]:
        st.subheader("Passes into the Box (Completed Passes)")
        visuals.plot_pass_map(passes_df, team_name=team_name, pass_type="Box")

    with event_tabs[3]:
        st.subheader("Carries in Final Third (Completed Carries)")
        visuals.plot_carry_map(carries_df, team_name=team_name, carry_type="Final Third")

    with event_tabs[4]:
        st.subheader("Carries into the Box (Completed Carries)")
        visuals.plot_carry_map(carries_df, team_name=team_name, carry_type="Box")     
            
    # Triggers pipeline that explodes match data into bin-level rows, trains classifiers, and visualizes results.
                
    st.subheader("📊 Big Chance Model Evaluation")
    st.markdown(
    """
    This section uses the match data you've explored to **train an XGBoost classifier** that predicts whether a team will **produce or concede a big chance** in the **next 10 minutes** of play.

    ### How it works:
    - **Match Binning**: Each match is split into 10-minute segments, and stats are aggregated per bin.
    - **Feature Engineering**: We use historical context — cumulative xG, passes, and carries — up to that bin.
    - **Target Labels**: The label is whether a big chance occurs in the **next** time bin.
    - **Train/Test Split**: Matches are split 80/20 **by match ID** to prevent data leakage.
    - **Model Level**: Training is done **per-bin**, but results are summarized **per match** to keep them interpretable.

    Use the results to identify patterns in team momentum, early signs of defensive vulnerability, or consistent attacking intent across matches.

    """
    )
    
    # model_for, model_against = ml.train_big_chance_prediction_models(matches)
    ml.train_big_chance_prediction_models(matches[matches['team'] == selected_team_name])
    
if __name__ == "__main__":
    main()