import streamlit as st
from requests.exceptions import HTTPError
import time
import pandas as pd
from statsbombpy import sb
import numpy as np

from requests_cache import install_cache

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
    # competitions = competitions[0:5]
    
    competitions["teams"] = pd.NA
    competitions["teams count"] = 0
    competitions["matches"] = 0
    
    for i, row in competitions.iterrows():
        
        matches = get_matches(row["competition_id"], row["season_id"])

        if matches is None or matches.empty:
            continue
        
        # Get unique teams from home and away columns
        teams = pd.concat([matches["home_team"], matches["away_team"]], ignore_index=True)
        team_names = teams.apply(pd.Series)
        teams_unique = team_names.dropna().drop_duplicates().values.tolist()

        # if i == 1:
        #     # st.text(matches.columns)
        #     st.text(teams_unique)
        #     st.text(len(teams_unique))
        #     st.text(type(teams_unique))
        
        # Assign counts back
        competitions.at[i, "teams"] = teams_unique
        competitions.at[i, "teams count"] = len(teams_unique)
        competitions.at[i, "matches"] = len(matches)

    return competitions