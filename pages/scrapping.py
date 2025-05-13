"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st
import pandas as pd

from utils.data_fetcher import fetch_json, fetch_season_json, fetch_standing_json


from utils.page_components import (
    add_common_page_elements,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.header("Web Scrapping", divider=True)
st.text("Where and how I get my data")



# Updated tournaments list
tournaments = [
    {"name": "Premier League", "id": 1, "unique_tournament": 17},
    {"name": "La Liga", "id": 36, "unique_tournament": 8},
    {"name": "Bundesliga", "id": 42, "unique_tournament": 35},
    {"name": "Serie A", "id": 33, "unique_tournament": 23},
    {"name": "Ligue 1", "id": 34, "unique_tournament": 34},
]

st.title("Football Tournament & Season Selector")

# Tournament selection
tournament_names = [t["name"] for t in tournaments]
selected_tournament_name = st.selectbox("Select a Tournament", tournament_names, index=0)
selected_tournament = next(t for t in tournaments if t["name"] == selected_tournament_name)

# Fetch seasons
# seasons_url = f"https://api.sofascore.com/api/v1/tournament/{selected_tournament['id']}/seasons"
try:
    seasons_response = fetch_season_json(selected_tournament)
    seasons_data = seasons_response.get("seasons", [])
    st.subheader("Available Seasons")
    # st.dataframe(pd.json_normalize(seasons_data))

    if seasons_data:
        season_names = [f"{s.get('name')} ({s.get('year')})" for s in seasons_data]
        selected_index = st.selectbox("Select a Season", range(len(season_names)), format_func=lambda i: season_names[i], index=1)
        selected_season = seasons_data[selected_index]

        # # Fetch standings
        # standings_url = (
        #     f"https://www.sofascore.com/api/v1/unique-tournament/"
        #     f"{selected_tournament['unique_tournament']}/season/{selected_season['id']}/standings/total"
        # )
        standings_response = fetch_standing_json(selected_tournament, selected_season)
        tables = standings_response.get("standings", [])

        if tables:
            
            rows = tables[0].get("rows", [])
            # df1 = pd.json_normalize(rows)
            # st.text(df1.columns)
            
            # Extract and reshape data
            def flatten_table_row(row):
                base = {
                    "id": row.get("team", {}).get("id"),
                    "team_name": row.get("team", {}).get("name"),
                    "position": row.get("position"),
                    "wins": row.get("wins"),
                    "draws": row.get("draws"),
                    "losses": row.get("losses"),
                    "scoresFor": row.get("scoresFor"),
                    "scoresAgainst": row.get("scoresAgainst"),
                    "scoreDiffFormatted": row.get("scoreDiffFormatted"),
                    "description": " | ".join(desc["text"] for desc in row.get("descriptions", [])) if row.get("descriptions") else "",
                    "team_nameCode": row.get("team", {}).get("nameCode"),
                    "league_Result": row.get("promotion", {}).get("text"),
                    "team_Colours": row.get("team", {}).get("teamColors"),
                    "team_National": row.get("team", {}).get("national")
                }
                # Include all other fields too
                return base

            table_data = [flatten_table_row(r) for r in rows]
            table_df = pd.DataFrame(table_data)

            # Reorder columns
            table_first_cols = ["id", "team_name", "position", "wins", "draws", "losses", "scoresFor", "scoresAgainst", "scoreDiffFormatted", "description", "team_nameCode", "league_Result", "team_Colours", "team_National"]
            # remaining_cols = [col for col in df.columns if col not in first_cols]
            table_df_formatted = table_df[table_first_cols] #+ remaining_cols]

            st.subheader("League Standings (Flattened Data)")
            st.dataframe(table_df_formatted)
            
            # Fetch rounds
            rounds_url = f"https://www.sofascore.com/api/v1/unique-tournament/{selected_tournament['unique_tournament']}/season/{selected_season['id']}/rounds"
            rounds_data = fetch_json(rounds_url)
            
            if "currentRound" in rounds_data and "rounds" in rounds_data:
                current_round = rounds_data["currentRound"].get("round", 0)
                available_rounds = [r["round"] for r in rounds_data["rounds"] if r["round"] <= current_round]

                selected_round = st.selectbox("Select a Round", available_rounds)

                st.subheader("Selected Round")
                st.write(f"Selected Round: {selected_round}")
                
                round_url = f"https://www.sofascore.com/api/v1/unique-tournament/{selected_tournament['unique_tournament']}/season/{selected_season['id']}/events/round/{selected_round}"
                round_response = fetch_json(round_url)
                round_events = round_response.get("events", [])
                
                if round_events:
                    # st.text("Raw Events Data:")
                    # st.text(round_events)
                    
                    # round_events_columns = pd.json_normalize(round_events).columns
                    # for column in round_events_columns:
                    #     st.text(column)
                    
                    # round_events_data = pd.json_normalize(round_events)
                    # st.dataframe(round_events_data)
                    
                    def extract_goal_incidents(base_row):
                        home_team_id = base_row["homeTeam.id"]
                        away_team_id = base_row["awayTeam.id"]
                        
                        max_injuryTime1 = base_row["time.injuryTime1"] 
                        max_injuryTime2 = base_row["time.injuryTime2"]
                        
                        eventsId = base_row["id"]
                        
                        incidents_url = f"https://www.sofascore.com/api/v1/event/{eventsId}/incidents"
                        incidents = fetch_json(incidents_url)
                        
                        home_goals = []
                        away_goals = []                       

                        for incident in incidents.get("incidents", []):
                            if incident.get("incidentType") == "goal":
                                goal_event = {                                    
                                    "minute": incident.get("time"),
                                    "timestamp": incident.get("timeSeconds"),
                                    "half": "1st" if incident.get("time") <= 45 else "2nd",
                                    "addedTime": incident.get("addedTime", None), 
                                    "playerId": incident.get("player", {}).get("id"),
                                    "player": incident.get("player", {}).get("name"),
                                    "playerShortName": incident.get("player", {}).get("shortName"),                                    
                                    "isOwnGoal": incident.get("icidentClass") == "ownGoal",
                                    "type": incident.get("incidentClass", "regular"),
                                }
                                
                                if (goal_event["half"] == "1st" and goal_event["addedTime"] is not None):
                                    if goal_event["addedTime"] > max_injuryTime1:
                                        max_injuryTime1 = goal_event["addedTime"]
                                
                                if (goal_event["half"] == "2nd" and goal_event["addedTime"] is not None):
                                    if goal_event["addedTime"] > max_injuryTime2:
                                        max_injuryTime2 = goal_event["addedTime"]                                
                                
                                if incident.get("isHome", False):
                                    goal_event["teamId"] = home_team_id
                                    home_goals.append(goal_event)
                                else:
                                    goal_event["teamId"] = away_team_id
                                    away_goals.append(goal_event)

                        return max_injuryTime1, max_injuryTime2, home_goals, away_goals
                    
                    
                    def flatten_round_row(row):
                        base = {
                            "id": row.get("id"),
                            "customId": row.get("customId"),
                            
                            # Season Details
                            "season.name": row.get("season", {}).get("name"),
                            "season.year": row.get("season", {}).get("year"),
                            "season.id": row.get("season", {}).get("id"),
                            "roundInfo.round": row.get("roundInfo", {}).get("round"),
                            
                            "winnerCode": row.get("winnerCode"),
                            "hasGlobalHighlights": row.get("hasGlobalHighlights"),
                            "hasXg": row.get("hasXg"),
                            "hasEventPlayerStatistics": row.get("hasEventPlayerStatistics"),
                            "hasEventPlayerHeatMap": row.get("hasEventPlayerHeatMap"),
                            "detailId": row.get("detailId"),                            
                            "homeRedCards": row.get("homeRedCards", None),  # Defaults to None if missing
                            "awayRedCards": row.get("awayRedCards", None),  # Defaults to None if missing
                            "slug": row.get("slug"),
                            "startTimestamp": row.get("startTimestamp"),
                            
                            # Tournament Details
                            "tournament.name": row.get("tournament", {}).get("name"),
                            "tournament.slug": row.get("tournament", {}).get("slug"),
                            "tournament.category.country.name": row.get("tournament", {}).get("category", {}).get("country", {}).get("name"),
                            
                            # Teams Details
                            "homeTeam.id": row.get("homeTeam", {}).get("id"),
                            "homeTeam.name": row.get("homeTeam", {}).get("name"),
                            "homeTeam.slug": row.get("homeTeam", {}).get("slug"),
                            "awayTeam.id": row.get("homeTeam", {}).get("id"),
                            "awayTeam.name": row.get("awayTeam", {}).get("name"),
                            "awayTeam.slug": row.get("awayTeam", {}).get("slug"),
                            
                            # Scores
                            "homeScore.display": row.get("homeScore", {}).get("display"),
                            "awayScore.display": row.get("awayScore", {}).get("display"),
                            
                            # Times
                            "time.injuryTime1": row.get("time", {}).get("injuryTime1"),
                            "time.injuryTime2": row.get("time", {}).get("injuryTime2"),
                        }
                                               
                        base["time.injuryTime1"], base["time.injuryTime2"], base["incidents.home_goals"], base["incidents.away_goals"] = extract_goal_incidents(base)
                        
                        base["time.totalTime"] = 90 + base["time.injuryTime1"] + base["time.injuryTime2"]
                        
                        # Include any other columns you want in the same format:
                        # "column_name": row.get("column_name", default_value)
                        
                        return base

                    round_events_data = [flatten_round_row(r) for r in round_events]
                    filtered_round_events = pd.DataFrame(round_events_data)
                    
                    st.subheader("Flattened Round Events Data")
                    st.dataframe(filtered_round_events,
                                    column_config={
                                        "incidents.home_goals": st.column_config.JsonColumn(
                                            "Home Goal Incidents",
                                                help="JSON strings or objects",
                                                width="large",
                                        ),
                                        "incidents.away_goals": st.column_config.JsonColumn(
                                            "Away Goals Incidents",
                                                help="JSON strings or objects",
                                                width="large",
                                        )                                        
                                    }
                                )
                    
                else:
                    st.warning("No events available for this round.")
            else:
                st.warning("No rounds available for this season.")
            
        else:
            st.warning("No standings available for this season.")
    else:
        st.warning("No seasons found.")
except Exception as e:
    st.error(f"Failed to fetch data: {e}")