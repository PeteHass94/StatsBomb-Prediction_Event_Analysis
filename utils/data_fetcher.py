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

def process_match_events(match_events):
    if not match_events.empty:
        match_events[['location_x', 'location_y']] = match_events['location'].apply(pd.Series)
        match_events[['pass_end_location_x', 'pass_end_location_y']] = match_events['pass_end_location'].apply(pd.Series)
        match_events['shot_end_location_x'], match_events['shot_end_location_y'], match_events['shot_end_location_z'] = np.nan, np.nan, np.nan
        end_locations = np.vstack(match_events.loc[match_events.type == 'Shot'].shot_end_location.apply(
            lambda x: x if len(x) == 3 else x + [np.nan]).values)
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_x'] = end_locations[:, 0]
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_y'] = end_locations[:, 1]
        match_events.loc[match_events.type == 'Shot', 'shot_end_location_z'] = end_locations[:, 2]
    return match_events



def parse_timestamp(ts_str):
    """Convert timestamp string 'HH:MM:SS.MS' to timedelta"""
    return datetime.strptime(ts_str, "%H:%M:%S.%f") - datetime.strptime("00:00:00.000", "%H:%M:%S.%f")


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
        unique_team_names = teams.dropna().drop_duplicates().values.tolist()
        
        # target_team_names = set(unique_team_names)
        # st.text(unique_team_names)

        # found_teams = dict()  # name -> (id, name)

        # # Step 2: Loop through matches until all team names accounted for
        # for j in range(len(matches)):
        #     # st.text(f"{len(found_teams)} , {len(unique_team_names)}")
        #     if len(found_teams) == len(unique_team_names):
        #         # st.text(f"{j}, {row.competition_name}")
        #         break  # Stop early if all teams found

        #     match_id = matches.loc[j, "match_id"]
        #     # st.text(f"{j}, {match_id}")

        #     try:
        #         events = get_match_events(match_id)
        #         # st.text(f"{j}, {match_id} , {len(events)}")
        #     except:
        #         continue

        #     if "team_id" not in events.columns or "team" not in events.columns:
        #         continue

        #     # Get unique teams in this match
        #     event_teams = events[["team_id", "team"]].dropna().drop_duplicates()

        #     for _, team_row in event_teams.iterrows():
        #         name = team_row["team"]
        #         if name not in found_teams:
        #             found_teams[name] = (team_row["team_id"], name)

        # Step 3: Assign to DataFrame
        # team_list = list(found_teams.values())
        # team_str = ", ".join([f"{name}(id:{id})" for (id, name) in team_list]) if isinstance(team_list, list) else ""
        # st.text(team_list)
        competitions.at[i, "teams"] = unique_team_names
        # competitions.at[i, "teams_str"] = team_str
        competitions.at[i, "teams count"] = len(unique_team_names)
        competitions.at[i, "matches"] = len(matches)

    return competitions

# @st.cache_data
def get_teams_matches(competition_id, team):
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
                "match_time": events["match_time"],
                "goals_scored": events["goals_scored"],
                "goals_conceded": events["goals_conceded"],
                "shots_attempted": events["shots_attempted"],
                "shots_conceded": events["shots_conceded"]                
            }

            # Add to list
            match_data_list.append(match_dict)

    # Create DataFrame from list of dictionaries
    return pd.DataFrame(match_data_list)

def get_match_events_timeline(match, team_name):
    # Load and process events
    events = process_match_events(get_match_events(match['match_id']))
    
    max_time_p1 = parse_timestamp(events[events['period'] == 1]['timestamp'].max())
    
    # Add timeValue column (float minutes since KO)
    def compute_time_value(row, max_time_period1=max_time_p1):
        if pd.isna(row['timestamp']):
            return np.nan
        if row['period'] == 1:
            base_time = parse_timestamp(row['timestamp']) 
        elif row['period'] == 2:
            base_time = parse_timestamp(row['timestamp']) + max_time_period1
        total_minutes = base_time.total_seconds() / 60
        
        return round(total_minutes, 2)
    
    events['timeValue'] = events.apply(compute_time_value, axis=1)
    # events['timeValue'] = events['timeValue'].round(2)
    
    # Filter by team
    team_events = events[events['team'] == team_name]
    opp_events = events[events['team'] != team_name]
    
    # ---------- GOALS ----------
    # Goals scored by team
    goals_scored = team_events[
        (team_events['type'] == "Shot") & (team_events['shot_outcome'] == "Goal")
    ][['timeValue', 'period']].to_dict("records")

    # Own goals scored for the team
    own_goals_for = team_events[
        team_events['type'] == "Own Goal For"
    ][['timeValue', 'period']].to_dict("records")

    # Goals conceded by team (opponent scored)
    goals_conceded = opp_events[
        (opp_events['type'] == "Shot") & (opp_events['shot_outcome'] == "Goal")
    ][['timeValue', 'period']].to_dict("records")

    # Own goals against the team
    own_goals_against = opp_events[
        opp_events['type'] == "Own Goal For"
    ][['timeValue', 'period']].to_dict("records")

    # Combine goal timelines
    team_goal_timeline = goals_scored + own_goals_for
    team_conceded_timeline = goals_conceded + own_goals_against

    # ---------- SHOTS ----------
    # Shots attempted with xG
    team_shots = team_events[
        team_events['type'] == "Shot"
    ][['timeValue', 'period', 'shot_statsbomb_xg', 'shot_outcome']].to_dict("records")

    # Shots conceded with xG
    shots_conceded = opp_events[
        opp_events['type'] == "Shot"
    ][['timeValue', 'period', 'shot_statsbomb_xg', 'shot_outcome']].to_dict("records")
    
    # ---------- MATCH TIME ----------
    # max_p1 = events[events['period'] == 1]['timeValue'].max()
    # max_p2 = events[events['period'] == 2]['timeValue'].max()
    # # max_p2 -= 45  # remove added 45
    # total_time = max_p1 + max_p2 - 45.0

    match_time = events['timeValue'].max()

    return {
        "goals_scored": team_goal_timeline,
        "goals_conceded": team_conceded_timeline,
        "shots_attempted": team_shots,
        "shots_conceded": shots_conceded,
        "match_time": match_time
    }