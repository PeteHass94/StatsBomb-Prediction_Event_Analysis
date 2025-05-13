import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter
from mplsoccer import VerticalPitch, add_image, Pitch, Sbopen, FontManager
from mplsoccer import FontManager, Radar, grid
import matplotlib.patheffects as path_effects
import streamlit as st
from requests.exceptions import HTTPError


#Only get the player filter if selected
def heatmap(data, player_name):
    data_events = data
    data_filter = data_events.query("player == @player_name")
    data = data_filter[['location_x', 'location_y']]

    # Setup pitch
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef', tick=True)
    
    # Draw pitch
    fig, ax = pitch.draw(figsize=(10, 6))
    fig.set_facecolor('#22312b')
    
    # Create bin statistics
    bin_statistic = pitch.bin_statistic(data.location_x, data.location_y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    
    # Plot heatmap
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
    
    # Add the colorbar and format it
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

    # Add title
    fig.suptitle(f"Heatmap of {player_name}", x=0.45, y=0.93, fontsize=15, color='white')

    # Return the figure
    return fig

#Only make a team filter if selected
def shot_analysis(data, team):
    data_events = data
    df_filtered = data_events.loc[(data_events['team'] == team) & (data_events['type'] == 'Shot')& (data_events['location_x'].notnull()) & (data_events['location_y'].notnull())]

    
    pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color = '#FBFBFB', line_color = 'black', corner_arcs= True, linewidth = 1, half = True)
    
    fig, ax = pitch.draw(figsize=(30, 10),  nrows=1,  ncols=3, tight_layout= False)
    fig.set_facecolor('#FBFBFB')


    bins = [(6,5), (1,5), (6,1)]
    
    for i, bin in enumerate(bins):
        bin_statistic = pitch.bin_statistic(df_filtered['location_x'], df_filtered['location_y'], statistic = 'count', bins = bin)
        pitch.heatmap(bin_statistic, ax=ax[i], cmap ='Blues', edgecolors='white')
        bin_statistic['statistic'] = (pd.DataFrame((bin_statistic['statistic'] / bin_statistic['statistic'].sum()))
                                        .applymap(lambda x: '{:.0%}'.format(x)).values)
        
        pitch.label_heatmap(bin_statistic, color = 'black', fontsize=14, ax=ax[i], ha='center', va='bottom')

#Just plot the chart
def shots(data):
    dataframe_filtered = data.query("(type == 'Shot' or type == 'Own Goal Against') and period != 5 ")
    team1, team2 = dataframe_filtered.team.unique()
    all_shots = dataframe_filtered[['team','player','shot_outcome','location_x', 'location_y', 'minute']]
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#fbfbfb', line_color='grey')

    
    fig, ax = pitch.draw(figsize=(15, 10))
    #Size of the pitch in yards (!!!)
    pitchLengthX = 120
    pitchWidthY = 80
    fig.set_facecolor("#fbfbfb")
    #Plot the shots by looping through them.
    for i, shot in all_shots.iterrows():
        #get the information
        x = shot['location_x']
        y = shot['location_y']
        goal = shot['shot_outcome']=='Goal'
        team_name = shot['team']
        #set circlesize
        circleSize = 2.0
        #plot team_1
        if (team_name == team1):
            if goal:
                shotCircle = plt.Circle((x,y), circleSize, color="#ff727c", edgecolor='black')
                plt.text(x-1.5, y-2.5, shot['minute'], fontsize=12, color = "black", fontweight = 'semibold')
            else:
                shotCircle = plt.Circle((x,y), circleSize, color="#ff727c")
                shotCircle.set_alpha(.2)
        #plot team_2
        else:
            if goal:
                shotCircle = plt.Circle((pitchLengthX-x, pitchWidthY-y), circleSize, color="#97b4f5", edgecolor='white')
                plt.text(pitchLengthX-x-1.5, pitchWidthY-y-2.5, shot['minute'], fontsize=12, color = "black", fontweight = 'semibold')
            else:
                shotCircle = plt.Circle((pitchLengthX-x, pitchWidthY-y), circleSize, color="#97b4f5")
                shotCircle.set_alpha(.2)
        ax.add_patch(shotCircle)
    #set title
    fig.suptitle(f"{team2} (blue) and {team1} (red) shots", y = 0.76,fontsize=20, color='black')
    fig.set_size_inches(12.2, 15)
    return plt.show()

#Make only a team filter
def ball_passer_receiver(team, data):
    data_events = data
    mask = (data_events['team'] == team) & (data_events['type'] == 'Pass')
    data_filtered = data_events.loc[mask, ['player', 'pass_recipient']].copy()
    data_filtered['Ball Passer - Receiver'] = data_filtered['player'] + '*' + data_filtered['pass_recipient']
    counts = data_filtered['Ball Passer - Receiver'].value_counts()
    df_counts = pd.DataFrame({'Ball Passer - Receiver': counts.index, 'Total Passes': counts.values})
    df_counts[['Ball Passer', 'Pass Receiver']] = df_counts['Ball Passer - Receiver'].str.split('*', expand=True)
    return df_counts[['Ball Passer', 'Pass Receiver', 'Total Passes']].sort_values(by='Total Passes', ascending=False).head(10)

#Make only a team filter
def pass_network(team_name, data):
    data_events = data
    ball_passes = data_events.loc[data_events['type'] == 'Pass', ['type', 'team', 'player', 'pass_recipient', 
                                                                  'timestamp', 'location_x', 'location_y', 
                                                                  'pass_end_location_x', 'pass_end_location_y', 'minute']]
    ball_passes = ball_passes.loc[(ball_passes['team'] == team_name) & (ball_passes['pass_recipient'].notnull())]
    ball_passes['pair'] = ball_passes['player'] + ' --> ' + ball_passes['pass_recipient']

    # Creating a dataframe with the pair of players and the number of passes
    pass_count = ball_passes.groupby(['pair']).count().reset_index()
    pass_count = pass_count[['pair', 'timestamp']]
    pass_count.columns = ['pair', 'number_pass']

    # Merging the data filtered and the pass count
    pass_merge = ball_passes.merge(pass_count, on='pair')
    pass_merge = pass_merge[['player', 'pass_recipient', 'number_pass']].drop_duplicates()

    # Mean location of each player
    avg_loc_df = ball_passes.groupby(['team', 'player']).agg({'location_x': np.mean, 'location_y': np.mean}).reset_index()
    avg_loc_df = avg_loc_df[['player', 'location_x', 'location_y']]

    # Merging the data
    pass_cleaned = pass_merge.merge(avg_loc_df, on='player')
    pass_cleaned.rename({'location_x': 'pos_x_start', 'location_y': 'pos_y_start'}, axis='columns', inplace=True)
    pass_cleaned = pass_cleaned.merge(avg_loc_df, left_on='pass_recipient', right_on='player', suffixes=['', '_end'])
    pass_cleaned.rename({'location_x': 'pos_x_end', 'location_y': 'pos_y_end'}, axis='columns', inplace=True)

    # Creating column width
    pass_cleaned['width'] = pass_cleaned['number_pass'] / pass_cleaned['number_pass'].max()

    # Defining pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#efefef')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor("#22312b")

    pitch.draw(ax=ax)
    pitch.lines(pass_cleaned.pos_x_start, pass_cleaned.pos_y_start,
                pass_cleaned.pos_x_end, pass_cleaned.pos_y_end, lw=pass_cleaned.width * 5,
                color='white', zorder=1, ax=ax)
    pitch.scatter(pass_cleaned.pos_x_start, pass_cleaned.pos_y_start, s=300,
                  color='lightblue', edgecolors='black', linewidth=1, alpha=1, ax=ax)

    for index, row in pass_cleaned.iterrows():
        ax.text(row.pos_x_start, row.pos_y_start, row.player, fontsize=10, color='white', ha='center', va='center')

    ax.set_title(f'{team_name} Passing Network', fontsize=20, color='white')

    # Return the figure instead of calling st.pyplot
    return fig

def teamShotChart(data, team):
    events_df = data
    shot = events_df[(events_df['type'] == 'Shot')]
    shot = shot[(shot['team'] == team)]
    goal = events_df[(events_df['shot_outcome'] == 'Goal')]
    goal = goal[(goal['team'] == team)]

    # First chart: Shot and Goal Locations
    fig1, ax1 = plt.subplots(figsize=(13, 8.5))
    fig1.set_facecolor('#22312b')
    ax1.patch.set_facecolor('#22312b')

    pitch1 = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch1.draw(ax=ax1)
    plt.gca().invert_yaxis()

    ax1.scatter(shot['location_x'], shot['location_y'], s=100, c='blue', alpha=.7)
    ax1.scatter(goal['location_x'], goal['location_y'], s=250, c='lightblue', alpha=.7, zorder=2)

    ax1.set_title(f'{team} Shot Charts', fontsize=20, color='yellow')
    ax1.text(0.05, 0.95, f'Total Shots: {len(shot)}', transform=ax1.transAxes, fontsize=14, color='white')
    ax1.text(0.05, 0.90, f'Total Goals: {len(goal)}', transform=ax1.transAxes, fontsize=14, color='white')

    # Second chart: Heatmap of Shot Locations
    team_shot = (events_df.team == team) & (events_df.type == 'Shot')
    df_team_shot = events_df.loc[team_shot, ['location_x', 'location_y']]

    pitch2 = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b')
    fig2, ax2 = pitch2.draw(figsize=(10.6, 8.5))
    fig2.set_facecolor('#22312b')

    bin_statistic = pitch2.bin_statistic(df_team_shot.location_x, df_team_shot.location_y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch2.heatmap(bin_statistic, ax=ax2, cmap='hot', edgecolors='#22312b')

    cbar = fig2.colorbar(pcm, ax=ax2, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

    # Third chart: Heatmap of Shot Locations with Percentage
    fig3, ax3 = pitch2.draw(figsize=(9.7, 8.5))
    fig3.set_facecolor('#22312b')

    bin_statistic = pitch2.bin_statistic(df_team_shot.location_x, df_team_shot.location_y, statistic='count', bins=(6, 5), normalize=True)
    pitch2.heatmap(bin_statistic, ax=ax3, cmap='Reds', edgecolor='#22312b')

    # Define path effect
    path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]
    pitch2.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax3, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)

    # Return all three figures
    return fig1, fig2, fig3

def xGEvolution(data):
    # Calculate cumulative xG for each team separately
    data['cumulative_xg'] = data.groupby('team')['shot_statsbomb_xg'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    ax.grid(ls='dotted', lw=.5, color='white', axis='y', zorder=1)
    spines = ['top', 'bottom', 'left', 'right']
    for x in spines:
        ax.spines[x].set_visible(False)

    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels([0, 15, 30, 45, 60, 75, 90], color='white')
    ax.set_xlabel('Minute', color='white', fontsize=16)
    ax.set_ylabel('Cumulative xG', color='white', fontsize=16)

    for team_name, team_df in data.groupby('team'):
        ax.plot(team_df['minute'], team_df['cumulative_xg'], label=team_name, linewidth=2)

    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='y', colors='white')
    ax.set_title('xG Evolution', fontsize=20, color='white')

    # Return the figure instead of calling st.pyplot
    return fig

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