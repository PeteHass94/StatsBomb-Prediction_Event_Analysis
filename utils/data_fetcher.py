import streamlit as st
from requests.exceptions import HTTPError
import time
import pandas as pd
from statsbombpy import sb
import numpy as np

from requests_cache import install_cache

from datetime import timedelta, datetime
from decimal import Decimal, ROUND_HALF_UP

install_cache("statsbomb_cache", backend="sqlite", expire_after=3600)  # Cache expires after 1 hour

def fetch_with_retries(fetch_function, retries=5, lineup=False, team="", **kwargs):
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

def parse_timestamp(ts_str):
    """Convert timestamp string 'HH:MM:SS.MS' to timedelta"""
    return datetime.strptime(ts_str, "%H:%M:%S.%f") - datetime.strptime("00:00:00.000", "%H:%M:%S.%f")


def compute_time_value(row, max_time_period1):
        if pd.isna(row['timestamp']):
            return np.nan
        if row['period'] == 1:
            base_time = parse_timestamp(row['timestamp']) 
        elif row['period'] == 2:
            base_time = parse_timestamp(row['timestamp']) + max_time_period1
        total_minutes = base_time.total_seconds() / 60
        return round(total_minutes, 2)


def process_match_events(match_events):
    if not match_events.empty:
        max_time_P1 = parse_timestamp(match_events[match_events['period'] == 1]['timestamp'].max())        
        match_events['timeValue'] = match_events.apply(lambda row: compute_time_value(row, max_time_period1=max_time_P1), axis=1)
        
        match_events[['location_x', 'location_y']] = match_events['location'].apply(pd.Series)
        match_events[['carry_end_location_x', 'carry_end_location_y']] = match_events['carry_end_location'].apply(pd.Series)
        match_events[['pass_end_location_x', 'pass_end_location_y']] = match_events['pass_end_location'].apply(pd.Series)
        match_events['shot_end_location_x'], match_events['shot_end_location_y'], match_events['shot_end_location_z'] = np.nan, np.nan, np.nan
        end_locations = np.vstack(match_events.loc[match_events.type == 'Shot'].shot_end_location.apply(
            lambda x: x if len(x) == 3 else x + [np.nan]).values)
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_x'] = end_locations[:, 0]
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_y'] = end_locations[:, 1]
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_z'] = end_locations[:, 2]
    return match_events





# Cached API calls
@st.cache_data
def get_competitions():
    competitions = fetch_with_retries(sb.competitions)
    only_computition_ids = [9, 11, 7, 2, 12]
    only_season_ids = [27]
    competitions = competitions[
        (competitions["competition_id"].isin(only_computition_ids)) & 
        (competitions["season_id"].isin(only_season_ids))
        ]
    return competitions

@st.cache_data
def get_matches(competition_id, season_id):
    return fetch_with_retries(sb.matches, competition_id=competition_id, season_id=season_id)

@st.cache_data
def get_match_events(match_id):
    return fetch_with_retries(sb.events, match_id=match_id)

@st.cache_data
def get_match_lineups(match):
    lineup_fetch = fetch_with_retries(sb.lineups, match_id=match.match_id.iloc[0])
    all_players = []
    for team_name, df in lineup_fetch.items():
        df = df.copy()
        df['team'] = team_name  # Add team column
        all_players.append(df)
    lineup_df = pd.concat(all_players, ignore_index=True)
    lineup_df['position'] = lineup_df['positions'].apply(lambda x: x[0]['position'] if x else None)
    lineup_df = lineup_df.drop(columns=['positions'])  # Drop original complex column
    return lineup_df

@st.cache_data
def get_competition_teams_matches():
    competitions = get_competitions()
    # competitions = competitions[0:9]

    competitions["teams"] = pd.NA
    # competitions["teams_str"] = pd.NA
    competitions["teams count"] = 0
    competitions["matches"] = 0

    for i, row in competitions.iterrows():
        matches = get_matches(row["competition_id"], row["season_id"])
        if matches is None or matches.empty:
            continue

        # Step 1: Extract unique team names from home and away team dicts
        teams = pd.concat([matches["home_team"], matches["away_team"]], ignore_index=True)
        unique_team_names = sorted(teams.dropna().drop_duplicates().values.tolist())
        
        # Step 2: Create a string representation of the team names
        competitions.at[i, "teams"] = unique_team_names
        # competitions.at[i, "teams_str"] = team_str
        competitions.at[i, "teams count"] = len(unique_team_names)
        competitions.at[i, "matches"] = len(matches)

    return competitions

@st.cache_data
def get_teams_matches(competition_id, team, threshold=0.2):
    competitions = get_competitions()
    seasons = competitions[competitions['competition_id'] == competition_id]

    match_data_list = []  # 🧠 List of row dictionaries

    for season_id in seasons['season_id']:
        matches = get_matches(competition_id, season_id)
        if matches is None or matches.empty:
            continue

        # Filter for matches involving the selected team
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ].copy()

        for _, match_row in team_matches.iterrows():
            # Determine home/away, scores, opponent
            is_home = match_row['home_team'] == team
            opponent = match_row['away_team'] if is_home else match_row['home_team']
            team_score = match_row['home_score'] if is_home else match_row['away_score']
            opponent_score = match_row['away_score'] if is_home else match_row['home_score']
            venue = 'Home' if is_home else 'Away'

            # Get timeline events
            events = get_match_events_timeline(match_row, team)
            if events is None:
                st.warning(f"No events found for match ID {match_row['match_id']}.")
                continue
            
            binned_rolling_df = events['binned_data']
            
            # Build row dict
            match_dict = {
                "match_id": match_row["match_id"],
                "match_date": match_row["match_date"],
                "kick_off": match_row["kick_off"],
                "competition": match_row["competition"],
                "season": match_row["season"],
                "home_team": match_row["home_team"],
                "away_team": match_row["away_team"],
                "home_score": match_row["home_score"],
                "away_score": match_row["away_score"],
                "match_week": match_row.get("match_week"),
                "team": team,
                "opponent": opponent,
                "team_score": team_score,
                "opponent_score": opponent_score,
                "venue": venue,
                "match_summary": f"Vs {opponent} ({venue[0]}, {pd.to_datetime(match_row['match_date']).strftime('%d/%m/%Y')} - {match_row['season']})",
                "team_match_summary": f"{team} Vs {opponent} ({venue[0]}, {pd.to_datetime(match_row['match_date']).strftime('%d/%m/%Y')} - {match_row['season']})",
                
                "match_time": events["match_time"],
                "goals_scored": events["goals_scored"].to_dict(orient='records'),
                "goals_conceded": events["goals_conceded"].to_dict(orient='records'),
                "shots_attempted": events["shots_attempted"].to_dict(orient='records'),
                "shots_conceded": events["shots_conceded"].to_dict(orient='records'),
                
                "binned_xg_for": binned_rolling_df['binned_xg_for'].tolist(),
                "binned_xg_against": binned_rolling_df['binned_xg_against'].tolist(),
                "binned_final_third_passes_for": binned_rolling_df['binned_final_third_passes_for'].tolist(),
                "binned_final_third_passes_against": binned_rolling_df['binned_final_third_passes_against'].tolist(),
                "binned_box_passes_for": binned_rolling_df['binned_box_passes_for'].tolist(),
                "binned_box_passes_against": binned_rolling_df['binned_box_passes_against'].tolist(),
                "binned_final_third_carries_for": binned_rolling_df['binned_final_third_carries_for'].tolist(),
                "binned_final_third_carries_against": binned_rolling_df['binned_final_third_carries_against'].tolist(),
                "binned_box_carries_for": binned_rolling_df['binned_box_carries_for'].tolist(),
                "binned_box_carries_against": binned_rolling_df['binned_box_carries_against'].tolist(),
                
                "rolling_xg_for": binned_rolling_df['rolling_xg_for'].tolist(),
                "rolling_xg_against": binned_rolling_df['rolling_xg_against'].tolist(),
                "rolling_final_third_passes_for": binned_rolling_df['rolling_final_third_passes_for'].tolist(),
                "rolling_final_third_passes_against": binned_rolling_df['rolling_final_third_passes_against'].tolist(),
                "rolling_box_passes_for": binned_rolling_df['rolling_box_passes_for'].tolist(),
                "rolling_box_passes_against": binned_rolling_df['rolling_box_passes_against'].tolist(),
                "rolling_final_third_carries_for": binned_rolling_df['rolling_final_third_carries_for'].tolist(),
                "rolling_final_third_carries_against": binned_rolling_df['rolling_final_third_carries_against'].tolist(),
                "rolling_box_carries_for": binned_rolling_df['rolling_box_carries_for'].tolist(),
                "rolling_box_carries_against": binned_rolling_df['rolling_box_carries_against'].tolist(),
                
                "goal_scored_next_10": binned_rolling_df['goal_scored_next_10'].tolist(),
                "goal_conceded_next_10": binned_rolling_df['goal_conceded_next_10'].tolist(),
                # "big_chance_for_next_10": binned_rolling_df['big_chance_for_next_10'].tolist(),
                # "big_chance_against_next_10": binned_rolling_df['big_chance_against_next_10'].tolist()
            }

            # Add to list
            match_data_list.append(match_dict)

    # Create DataFrame from list of dictionaries
    return pd.DataFrame(match_data_list)

def bin_events(event_list, value_col=None, bin_width=10, max_time=100):
    """
    Converts list of events into a binned time series.
    If value_col is provided, sums values in that column per bin.
    """
    df = pd.DataFrame(event_list)
    if df.empty:
        return pd.Series([0] * (max_time // bin_width), name=value_col or 'count')

    df['bin'] = (df['timeValue'] // bin_width).astype(int)
    if value_col:
        binned = df.groupby('bin')[value_col].sum()
    else:
        binned = df.groupby('bin').size()
        
    return binned.reindex(range(max_time // bin_width), fill_value=0)

def rolling_series(series, window=3):
    
    return series.rolling(window=window, min_periods=1).mean()

def add_future_goal_labels_per_bin(goals_scored, goals_conceded, horizon_bins=1):
    """
    Generate per-bin goal labels over the next `horizon_bins`.

    Args:
        goals_scored (list[int]): 0/1 list per bin.
        goals_conceded (list[int]): 0/1 list per bin.
        horizon_bins (int): How far into the future to look.

    Returns:
        Tuple of (goal_scored_next_10, goal_conceded_next_10) — both lists of len = len(goals) - horizon
    """
    num_bins = len(goals_scored) # 10
    # num_bins1 = len(goals_conceded)
    scored_labels = []
    conceded_labels = []

    # st.write(f"Num bins: {num_bins},  Num bins: {num_bins1}, Horizon bins: {horizon_bins}")
    
    for i in range(num_bins - horizon_bins):
        future_scored = sum(goals_scored[i+1:i+1+horizon_bins])
        future_conceded = sum(goals_conceded[i+1:i+1+horizon_bins])

        scored_labels.append(int(future_scored > 0))
        conceded_labels.append(int(future_conceded > 0))
    
    # Fill the rest with np.nan
    for _ in range(horizon_bins):
        scored_labels.append(np.nan)
        conceded_labels.append(np.nan)
    
    return scored_labels, conceded_labels

def add_future_big_chance_labels_per_bin(xg_for, xg_against, threshold=0.2, horizon_bins=1):
    """
    Generate binary labels per bin for big chances (based on xG threshold) in the next N bins.

    Args:
        xg_for (list[float]): xG values per bin for the team.
        xg_against (list[float]): xG values per bin against the team.
        threshold (float): xG threshold for a big chance.
        horizon_bins (int): How far into the future to look.

    Returns:
        Tuple of (big_chance_for_next_10, big_chance_against_next_10) — both lists of len = len(xg) - horizon
    """
    # Flatten values in case they're wrapped in a list (e.g., [0.23])
    xg_for = [v[0] if isinstance(v, list) else v for v in xg_for]
    xg_against = [v[0] if isinstance(v, list) else v for v in xg_against]
    
    num_bins = len(xg_for)
    big_chance_for = []
    big_chance_against = []

    for i in range(num_bins - horizon_bins):
        future_for = xg_for[i+1:i+1+horizon_bins]
        future_against = xg_against[i+1:i+1+horizon_bins]

        big_chance_for.append(int(any(v > threshold for v in future_for)))
        big_chance_against.append(int(any(v > threshold for v in future_against)))

    for _ in range(horizon_bins):
        big_chance_for.append(np.nan)
        big_chance_against.append(np.nan)
    
    return big_chance_for, big_chance_against


def get_match_events_timeline(match, team_name, bin_width=10, rolling_window=10, horizon_bins=1):
    # Load and process events
    events = process_match_events(get_match_events(match['match_id']))
    
    max_time = events['timeValue'].max()
    
    # Filter events
    team_events = events[events['team'] == team_name]
    opp_events = events[events['team'] != team_name]

    # Collect relevant data(type == 'Shot' or type == 'Own Goal Against')
    goals_scored = team_events[((team_events['type'] == "Shot") & (team_events['shot_outcome'] == "Goal")) | (team_events['type'] == 'Own Goal For')][['timeValue', 'period']]
    goals_conceded = opp_events[((opp_events['type'] == "Shot") & (opp_events['shot_outcome'] == "Goal")) | (opp_events['type'] == 'Own Goal For')][['timeValue', 'period']]
    shots_attempted = team_events[team_events['type'] == "Shot"][['timeValue', 'shot_statsbomb_xg']]
    shots_conceded = opp_events[opp_events['type'] == "Shot"][['timeValue', 'shot_statsbomb_xg']]

    team_passes = team_events[team_events['type'] == "Pass"]
    team_passes['pass_outcome'] = team_passes['pass_outcome'].fillna('Complete')
    team_passes = team_passes[team_passes['pass_outcome'] == 'Complete']
    opp_passes = opp_events[opp_events['type'] == "Pass"]
    opp_passes['pass_outcome'] = opp_passes['pass_outcome'].fillna('Complete')
    opp_passes = opp_passes[opp_passes['pass_outcome'] == 'Complete']
    
    team_final_third_passes = team_passes[team_passes['pass_end_location_x'] > 80]
    opp_final_third_passes = opp_passes[opp_passes['pass_end_location_x'] > 80]
    
    team_box_passes = team_passes[(team_passes['pass_end_location_x'] > 102) & (team_passes['pass_end_location_y'] > 18) & (team_passes['pass_end_location_y'] < 62)]
    opp_box_passes = opp_passes[(opp_passes['pass_end_location_x'] > 102) & (opp_passes['pass_end_location_y'] > 18) & (opp_passes['pass_end_location_y'] < 62)]
    
    team_carries = team_events[team_events['type'] == "Carry"]
    opp_carries = opp_events[opp_events['type'] == "Carry"]
    
    team_final_third_carries = team_carries[team_carries['carry_end_location_x'] > 80]
    opp_final_third_carries = opp_carries[opp_carries['carry_end_location_x'] > 80]
    
    team_box_carries = team_carries[(team_carries['carry_end_location_x'] > 102) & (team_carries['carry_end_location_y'] > 18) & (team_carries['carry_end_location_y'] < 62)]
    opp_box_carries = opp_carries[(opp_carries['carry_end_location_x'] > 102) & (opp_carries['carry_end_location_y'] > 18) & (opp_carries['carry_end_location_y'] < 62)]
    
    
    # Bin all relevant metrics
    binned_goals = bin_events(goals_scored.to_dict(orient='records'), bin_width=bin_width)
    binned_conceded = bin_events(goals_conceded.to_dict(orient='records'), bin_width=bin_width)
    binned_xg_for = bin_events(shots_attempted.to_dict(orient='records'), value_col='shot_statsbomb_xg', bin_width=bin_width)
    binned_xg_against = bin_events(shots_conceded.to_dict(orient='records'), value_col='shot_statsbomb_xg', bin_width=bin_width)

    binned_final_third_passes_for = bin_events(team_final_third_passes.to_dict(orient='records'), bin_width=bin_width)
    binned_final_third_passes_against = bin_events(opp_final_third_passes.to_dict(orient='records'), bin_width=bin_width)
    binned_box_passes_for = bin_events(team_box_passes.to_dict(orient='records'), bin_width=bin_width)
    binned_box_passes_against = bin_events(opp_box_passes.to_dict(orient='records'), bin_width=bin_width)
    binned_final_third_carries_for = bin_events(team_final_third_carries.to_dict(orient='records'), bin_width=bin_width)
    binned_final_third_carries_against = bin_events(opp_final_third_carries.to_dict(orient='records'), bin_width=bin_width)
    binned_box_carries_for = bin_events(team_box_carries.to_dict(orient='records'), bin_width=bin_width)
    binned_box_carries_against = bin_events(opp_box_carries.to_dict(orient='records'), bin_width=bin_width)
    
    # Combine into single DataFrame
    # df = pd.DataFrame({
    #     'goals_scored': binned_goals,
    #     'goals_conceded': binned_conceded,
    #     'xg_for': binned_xg_for,
    #     'xg_against': binned_xg_against
        
    # })

    # Rolling averages
    # df['rolling_xg_for'] = rolling_series(df['xg_for'], window=rolling_window)
    # df['rolling_xg_against'] = rolling_series(df['xg_against'], window=rolling_window)
    rolling_xg_for = rolling_series(binned_xg_for, window=rolling_window)
    rolling_xg_against = rolling_series(binned_xg_against, window=rolling_window)
    
    rolling_final_third_passes_for = rolling_series(binned_final_third_passes_for, window=rolling_window)
    rolling_final_third_passes_against = rolling_series(binned_final_third_passes_against, window=rolling_window)
    rolling_box_passes_for = rolling_series(binned_box_passes_for, window=rolling_window)
    rolling_box_passes_against = rolling_series(binned_box_passes_against, window=rolling_window)
    rolling_final_third_carries_for = rolling_series(binned_final_third_carries_for, window=rolling_window)
    rolling_final_third_carries_against = rolling_series(binned_final_third_carries_against, window=rolling_window)
    rolling_box_carries_for = rolling_series(binned_box_carries_for, window=rolling_window)
    rolling_box_carries_against = rolling_series(binned_box_carries_against, window=rolling_window)
    
        
    # Combine into single DataFrame
    df = pd.DataFrame({
        'goals_scored': binned_goals,
        'goals_conceded': binned_conceded,
        
        'binned_xg_for': binned_xg_for,
        'binned_xg_against': binned_xg_against,
        'binned_final_third_passes_for': binned_final_third_passes_for,
        'binned_final_third_passes_against': binned_final_third_passes_against,
        'binned_box_passes_for': binned_box_passes_for,
        'binned_box_passes_against': binned_box_passes_against,
        'binned_final_third_carries_for': binned_final_third_carries_for,
        'binned_final_third_carries_against': binned_final_third_carries_against,
        'binned_box_carries_for': binned_box_carries_for,
        'binned_box_carries_against': binned_box_carries_against,
        
        'rolling_xg_for': rolling_xg_for,
        'rolling_xg_against': rolling_xg_against,
        'rolling_final_third_passes_for': rolling_final_third_passes_for,
        'rolling_final_third_passes_against': rolling_final_third_passes_against,
        'rolling_box_passes_for': rolling_box_passes_for,
        'rolling_box_passes_against': rolling_box_passes_against,
        'rolling_final_third_carries_for': rolling_final_third_carries_for,
        'rolling_final_third_carries_against': rolling_final_third_carries_against,
        'rolling_box_carries_for': rolling_box_carries_for,
        'rolling_box_carries_against': rolling_box_carries_against        
    })
    
    # Future labels
    
    df['goal_scored_next_10'], df['goal_conceded_next_10']  = add_future_goal_labels_per_bin(binned_goals, binned_conceded, horizon_bins=horizon_bins)
    # df['big_chance_for_next_10'], df['big_chance_against_next_10'] = add_future_big_chance_labels_per_bin(binned_xg_for, binned_xg_against, horizon_bins=horizon_bins, threshold=threshold)

    return {
        "match_time": max_time,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "shots_attempted": shots_attempted,
        "shots_conceded": shots_conceded,
        "binned_data": df  # Contains binned + rolling + label
    }
    
@st.cache_data
def get_all_teams_matches(competition_id):
    competitions = get_competition_teams_matches()
    teams_df = competitions[competitions["competition_id"] == competition_id]

    if teams_df.empty or teams_df.iloc[0]["teams"] is pd.NA:
        st.warning("No team data available for this competition.")
        return pd.DataFrame()

    all_teams = teams_df.iloc[0]["teams"]
    all_matches = []

    with st.container(height=150):
        for i, team in enumerate(all_teams):            
            st.info(f"Fetching matches for {team}... {i + 1} / {len(all_teams)}")
            team_matches = get_teams_matches(competition_id, team)
            all_matches.append(team_matches)

    return pd.concat(all_matches, ignore_index=True)
