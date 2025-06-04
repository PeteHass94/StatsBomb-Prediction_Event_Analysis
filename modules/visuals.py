import plotly.graph_objects as go
import streamlit as st
import numpy as np

from mplsoccer import VerticalPitch, Pitch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

def goal_lines(goals_df, bin_width=10, color='blue', name='Goal'):
    
    
    return [
        {
            "type": "line",
            "x0": time,
            "x1": time,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {
                "color": color,
                "width": 1.5,
                "dash": "dash"
            },
            "name": name
        }
        for time in goals_df['timeValue']
    ]
    
def plot_xg_histograms(binned_for, binned_against, goals_scored, goals_conceded, bin_width=10, for_name="For", against_name="Against"):
    x = [i * bin_width for i in range(len(binned_for))]
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=binned_for, name=for_name, marker_color='green'))
    fig.add_trace(go.Bar(x=x, y=binned_against, name=against_name, marker_color='red'))

    
    # Add dashed lines for goals is not None 
    if len(goals_scored) > 0:
        for line in goal_lines(goals_scored, bin_width=bin_width, color='goldenrod', name='Goal Scored'):
            fig.add_shape(line)
    if len(goals_conceded) > 0:
        for line in goal_lines(goals_conceded, bin_width=bin_width, color='white', name='Goal Conceded'):
            fig.add_shape(line)

    # Add dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='goldenrod', width=1.5, dash='dash'),
                             name='Goal Scored'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='white', width=1.5, dash='dash'),
                             name='Goal Conceded'))
    
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,           # Start at 0
        dtick=10,          # Show tick every 10 minutes
        range=[0, 105],
        title_text="Minute"
    )
    
    fig.update_layout(
        title=f"{for_name} and {against_name} Histogram",
        xaxis_title="Match Time (mins)",
        yaxis_title="Value",
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_rolling_xg(rolling_for, rolling_against, goals_scored, goals_conceded, bin_width=10, for_name="For", against_name="Against"):
    x = [i * bin_width for i in range(len(rolling_for))]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=rolling_for, name=for_name, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=rolling_against, name=against_name, line=dict(color='red')))

    # Add vertical lines for goals if not None
    if len(goals_scored) > 0:
        for line in goal_lines(goals_scored, bin_width=bin_width, color='goldenrod', name='Goal Scored'):
            fig.add_shape(line)
    if len(goals_conceded) > 0:
        for line in goal_lines(goals_conceded, bin_width=bin_width, color='white', name='Goal Conceded'):
            fig.add_shape(line)

    # Add dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='goldenrod', width=1.5, dash='dash'),
                             name='Goal Scored'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='white', width=1.5, dash='dash'),
                             name='Goal Conceded'))
    
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,           # Start at 0
        dtick=10,          # Show tick every 10 minutes
        range=[0, 105],
        title_text="Minute"
    )
    
    fig.update_layout(
        title=f"{for_name} and {against_name} Rolling Line Chart",
        xaxis_title="Match Time (mins)",
        yaxis_title="Rolling value",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_gantt_chart(sample, lab):
    # After match_sample and sample have been defined
    st.subheader(f"📊 Timeline Comparison Gantt Chart for `{lab}`")

    bin_ranges = sample["bin_range"].tolist()
    predicted_labels = sample["Predicted_Label"].tolist()
    true_labels = sample["True_Label"].tolist()

    colors_pred = ["blue" if val == 1 else "lightgrey" for val in predicted_labels]
    colors_true = ["blue" if val == 1 else "lightgrey" for val in true_labels]
    colors_comp = [
        "green" if pred == true else "red"
        for pred, true in zip(predicted_labels, true_labels)
    ]

    # Define bars as horizontal Gantt-style strips
    fig = go.Figure()

    # Comparison row
    fig.add_trace(go.Bar(
        x=[10] * len(bin_ranges),
        y=["Comparison"] * len(bin_ranges),
        orientation='h',
        width=0.4,
        base=[i * 10 + 10 for i in range(len(bin_ranges))],
        marker_color=colors_comp,
        showlegend=False,
        hovertext=bin_ranges
    ))
    
    # Actual row
    fig.add_trace(go.Bar(
        x=[10] * len(bin_ranges),
        y=["Actual"] * len(bin_ranges),
        orientation='h',
        width=0.4,
        base=[i * 10 + 10 for i in range(len(bin_ranges))],
        marker_color=colors_true,
        showlegend=False,
        hovertext=bin_ranges
    ))
    
    # Prediction row
    fig.add_trace(go.Bar(
        x=[10] * len(bin_ranges),
        y=["Prediction"] * len(bin_ranges),
        orientation='h',
        width=0.4,
        base=[i * 10 + 10 for i in range(len(bin_ranges))],
        marker_color=colors_pred,
        showlegend=False,
        hovertext=bin_ranges
    ))

    fig.update_layout(
        barmode='stack',
        title=f"{lab} - Gantt Comparison (Bin-Level)",
        xaxis=dict(title="Match Minute", range=[10, 105]),
        yaxis=dict(title="", tickvals=["Prediction", "Actual", "Comparison"]),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


### Pitch and Shot Map Functions

def PitchPlotHalf():
    return VerticalPitch(
        pitch_type='statsbomb',
        pitch_color='#e0e0e0',
        line_color='black',
        stripe=True,
        stripe_color='#cccccc',
        linewidth=0.5,
        line_alpha=0.75,
        goal_type='box',
        half=True
    )

def plot_shot_map(df, team_name="", threshold=0.2):
    df['team'] = df['team'].apply(lambda x: x['name'] if isinstance(x, dict) else x)
    team_name_clean = team_name.strip().lower()
    teams = df['team'].dropna().unique()
    opp_team = next((t for t in teams if t.strip().lower() != team_name_clean), "")

    team_color = "#5BAEE3"
    opp_color = "#E35B5B"
    shot_color = "#8E8668"
    shot_size = 50

    team_df = df[df['team'].str.strip().str.lower() == team_name_clean]
    opp_df = df[df['team'].str.strip().str.lower() != team_name_clean]

    pitch = PitchPlotHalf()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout = True, dpi=300, facecolor='#2C2C3A')  # Two side-by-side half pitches

    # Define a map from outcome to edge color
    outcome_colours = {
        'Goal': '#5BE381',
        'Saved To Post': '#E3C65B',        
        'Saved': '#E3C65B',
        'Saved Off T': '#5B67E3',        
        'Blocked': '#5B67E3',
        'Off T': '#5B67E3',
        'Post': '#5B67E3',
        'Wayward': '#5B67E3'
    }
    # Ensure all outcomes have a color, default to black if not found
    

    # # Map edge colors with fallback
    # team_edgecolors = team_df['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black'))
    # opp_edgecolors = opp_df['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black'))

    # Define high and low xG masks  
    team_high = team_df[team_df['shot_statsbomb_xg'] >= threshold]
    team_low = team_df[team_df['shot_statsbomb_xg'] < threshold]

    opp_high = opp_df[opp_df['shot_statsbomb_xg'] >= threshold]
    opp_low = opp_df[opp_df['shot_statsbomb_xg'] < threshold]

    def get_stats(df, label):
        return f"{label}\nShots: {len(df)}, Goals: {sum(df['shot_outcome'] == 'Goal')}, Big Chances: {sum(df['shot_statsbomb_xg'] >= threshold)}, Total xG: {df['shot_statsbomb_xg'].sum():.2f}"

    team_summary = get_stats(team_df, team_name)
    opp_summary = get_stats(opp_df, opp_team)
    
    
    # Team plots
    pitch.draw(ax=axs[0])

    # Low xG with circle
    pitch.scatter(
        team_low['location_x'], team_low['location_y'],
        ax=axs[0], color=team_low['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black')),
        edgecolors='black',
        s= shot_size, #(team_low['shot_statsbomb_xg'] * 2000) ** 0.5,
        alpha=1, linewidth=0.75, marker='o'
    )

    # High xG with 'P'
    pitch.scatter(
        team_high['location_x'], team_high['location_y'],
        ax=axs[0], color=team_high['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black')),
        edgecolors='black',
        s= shot_size, #(team_high['shot_statsbomb_xg'] * 2000) ** 0.5,
        alpha=1, linewidth=0.75, marker='X'
    )

    axs[0].set_title(f"{team_summary}", fontsize=10, color=team_color, fontweight='bold')

    # Same for opponent
    pitch.draw(ax=axs[1])

    pitch.scatter(
        opp_low['location_x'], opp_low['location_y'],
        ax=axs[1], color=opp_low['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black')),
        edgecolors='black',
        s=shot_size,
        alpha=1, linewidth=0.75, marker='o'
    )

    pitch.scatter(
        opp_high['location_x'], opp_high['location_y'],
        ax=axs[1], color=opp_high['shot_outcome'].map(lambda x: outcome_colours.get(x, 'black')),
        edgecolors='black',
        s=shot_size,
        alpha=1, linewidth=0.75, marker='X'
    )

    axs[1].set_title(f"{opp_summary}", fontsize=10, color=opp_color, fontweight='bold')

    legend_ax = fig.add_axes([0.44, 0.05, 0.1, 0.25])  # [left, bottom, width, height]
    legend_ax.axis("off")

    # Draw white background
    legend_ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=legend_ax.transAxes,
        edgecolor='grey', facecolor='whitesmoke', alpha=0.7, zorder=0
    ))

    # Add text and example scatters
    legend_ax.text(0.5, 0.9, 'Shot Key', fontsize=7, va='center', ha='center', transform=legend_ax.transAxes, fontweight='bold')
    
    legend_ax.text(0.55, 0.75, 'Big Chance:', fontsize=7, va='center', ha='right', transform=legend_ax.transAxes)
    legend_ax.scatter(0.725, 0.755, s=shot_size, color=shot_color, edgecolor='black', marker='X', linewidth=0.75, transform=legend_ax.transAxes)
    
    legend_ax.text(0.55, 0.55, 'Goal:', fontsize=7, va='center', ha='right', transform=legend_ax.transAxes)
    legend_ax.scatter(0.725, 0.555, s=shot_size, color=outcome_colours['Goal'], edgecolor='black', linewidth=0.75, transform=legend_ax.transAxes)

    legend_ax.text(0.55, 0.35, 'On Target:', fontsize=7, va='center', ha='right', transform=legend_ax.transAxes)
    legend_ax.scatter(0.725, 0.355, s=shot_size, color=outcome_colours['Saved'], edgecolor='black', linewidth=0.75, transform=legend_ax.transAxes)

    legend_ax.text(0.55, 0.15, 'Off Target:', fontsize=7, va='center', ha='right', transform=legend_ax.transAxes)
    legend_ax.scatter(0.725, 0.155, s=shot_size, color=outcome_colours['Off T'], edgecolor='black', linewidth=0.75, transform=legend_ax.transAxes)
     
    # fig.suptitle(f"Shot Map: {team_name} vs {opp_team}\n{team_summary}\n{opp_summary}", fontsize=10)
    st.pyplot(fig, use_container_width=True)
    
def plot_pass_map(df, team_name="", pass_type="Final Third"):

    df['pass_outcome'] = df['pass_outcome'].fillna('Complete')
    # df = df[df['pass_outcome'] == 'Complete']
    
    df['pass_type_category'] = df.apply(
        lambda row: 'goal-assist' if row.get('pass_goal_assist', 0) == 1 else (
            'shot-assist' if row.get('pass_shot_assist', 0) == 1 else 'other'
        ),
        axis=1
    )

    df['team'] = df['team'].apply(lambda x: x['name'] if isinstance(x, dict) else x)
    team_name_clean = team_name.strip().lower()
    teams = df['team'].dropna().unique()
    opp_team = next((t for t in teams if t.strip().lower() != team_name_clean), "")

    team_color = "#5BAEE3"
    opp_color = "#E35B5B"
    line_width = 1.5
    line_alpha = 0.8
    arrow_outline = [path_effects.withStroke(linewidth=0.25, foreground='black')]
    
    team_df = df[df['team'].str.strip().str.lower() == team_name_clean]
    opp_df = df[df['team'].str.strip().str.lower() != team_name_clean]
    
    if pass_type == "Final Third":
        team_df = team_df[team_df['pass_end_location_x'] >= 80]
        opp_df = opp_df[opp_df['pass_end_location_x'] >= 80]
    elif pass_type == "Box":
        team_df = team_df[(team_df['pass_end_location_x'] >= 102) & (team_df['pass_end_location_y'].between(18, 62))]
        opp_df = opp_df[(opp_df['pass_end_location_x'] >= 102) & (opp_df['pass_end_location_y'].between(18, 62))]
    
    pass_colors = {
        'goal-assist': '#5BE381',
        'shot-assist': '#E3C65B',
        'other': '#5B67E3'
    }
    
    pitch = PitchPlotHalf()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, dpi=300, facecolor='#2C2C3A')

    def get_pass_stats(df, label, type=""):
        total_passes = len(df)
        completed = sum(df['pass_outcome'] == 'Complete')
        pct_completed = (completed / total_passes * 100) if total_passes > 0 else 0
        shot_assists = sum(df['pass_type_category'] == 'shot-assist')
        goal_assists = sum(df['pass_type_category'] == 'goal-assist')
        return f"{label}\n{type} Passes: {total_passes}  Completed: {pct_completed:.1f}%\n Shot Assists: {shot_assists}  Goal Assists: {goal_assists}"

    team_summary = get_pass_stats(team_df, team_name, type=pass_type)
    opp_summary = get_pass_stats(opp_df, opp_team, type=pass_type)

    team_df = team_df[team_df['pass_outcome'] == 'Complete']
    opp_df = opp_df[opp_df['pass_outcome'] == 'Complete']
    
    pitch.draw(ax=axs[0])
    
    for pass_type, color in pass_colors.items():
        sub_df = team_df[team_df['pass_type_category'] == pass_type]
        arrows = pitch.arrows(sub_df['location_x'], sub_df['location_y'],
                    sub_df['pass_end_location_x'], sub_df['pass_end_location_y'],
                    ax=axs[0], color=color, width=line_width, alpha=line_alpha)
        # Apply black stroke outline
        arrows.set_path_effects(arrow_outline)
    
    axs[0].set_title(team_summary, fontsize=10, color=team_color, fontweight='bold')

    pitch.draw(ax=axs[1])
    
    for pass_type, color in pass_colors.items():
        sub_df = opp_df[opp_df['pass_type_category'] == pass_type]
        arrows = pitch.arrows(sub_df['location_x'], sub_df['location_y'],
                    sub_df['pass_end_location_x'], sub_df['pass_end_location_y'],
                    ax=axs[1], color=color, width=line_width, alpha=line_alpha)
        # Apply black stroke outline
        arrows.set_path_effects(arrow_outline)
    
    axs[1].set_title(opp_summary, fontsize=10, color=opp_color, fontweight='bold')

    legend_ax = fig.add_axes([0.45, 0.05, 0.1, 0.2]) # [left, bottom, width, height]
    legend_ax.axis("off")
    legend_ax.add_patch(Rectangle((0, 0), 1, 1, transform=legend_ax.transAxes,
                              edgecolor='grey', facecolor='whitesmoke', alpha=0.8, zorder=0))

    # Title
    legend_ax.text(0.5, 0.9, "Pass Key", fontsize=8, fontweight='bold',
                ha='center', va='center', transform=legend_ax.transAxes)
    
    # Goal Assist
    arrow1 = legend_ax.annotate('', xy=(0.8, 0.7), xytext=(0.6, 0.7),
                                arrowprops=dict(arrowstyle='->', color='#5BE381', lw=2),
                                xycoords='axes fraction')
    legend_ax.text(0.55, 0.70, 'Goal Assist:', fontsize=7,
                va='center', ha='right', transform=legend_ax.transAxes)

    # Shot Assist
    arrow2 = legend_ax.annotate('', xy=(0.8, 0.5), xytext=(0.6, 0.5),
                                arrowprops=dict(arrowstyle='->', color='#E3C65B', lw=2),
                                xycoords='axes fraction')
    legend_ax.text(0.55, 0.50, 'Shot Assist:', fontsize=7,
                va='center', ha='right', transform=legend_ax.transAxes)

    # Other Pass
    arrow3 = legend_ax.annotate('', xy=(0.8, 0.3), xytext=(0.6, 0.3),
                                arrowprops=dict(arrowstyle='->', color='#5B67E3', lw=2),
                                xycoords='axes fraction')
    legend_ax.text(0.55, 0.30, 'Other Pass:', fontsize=7,
                va='center', ha='right', transform=legend_ax.transAxes)   
   
    st.pyplot(fig, use_container_width=True)
    

def plot_carry_map(df, team_name="", carry_type="Final Third"):
    df['team'] = df['team'].apply(lambda x: x['name'] if isinstance(x, dict) else x)
    team_name_clean = team_name.strip().lower()
    teams = df['team'].dropna().unique()
    opp_team = next((t for t in teams if t.strip().lower() != team_name_clean), "")

    team_color = "#5BAEE3"
    opp_color = "#E35B5B"
    line_width = 1.5
    line_alpha = 0.8
    arrow_outline = [path_effects.withStroke(linewidth=0.25, foreground='black')]

    team_df = df[df['team'].str.strip().str.lower() == team_name_clean]
    opp_df = df[df['team'].str.strip().str.lower() != team_name_clean]

    if carry_type == "Final Third":
        team_df = team_df[team_df['location_x'] >= 80]
        opp_df = opp_df[opp_df['location_x'] >= 80]
    elif carry_type == "Box":
        team_df = team_df[(team_df['carry_end_location_x'] >= 102) & (team_df['carry_end_location_y'].between(18, 62))]
        opp_df = opp_df[(opp_df['carry_end_location_x'] >= 102) & (opp_df['carry_end_location_y'].between(18, 62))]

    pitch = PitchPlotHalf()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, dpi=300, facecolor='#2C2C3A')

    def get_carry_stats(df, label, type=""):
        return f"{label}\n{type} Carries: {len(df)}"

    team_summary = get_carry_stats(team_df, team_name, type=carry_type)
    opp_summary = get_carry_stats(opp_df, opp_team, type=carry_type)

    pitch.draw(ax=axs[0])
    arrows = pitch.arrows(team_df['location_x'], team_df['location_y'],
                 team_df['carry_end_location_x'], team_df['carry_end_location_y'],
                 ax=axs[0], color=team_color, width=line_width, alpha=line_alpha)
    # Apply black stroke outline
    arrows.set_path_effects(arrow_outline)
    
    axs[0].set_title(team_summary, fontsize=10, color=team_color, fontweight='bold')

    pitch.draw(ax=axs[1])
    arrows = pitch.arrows(opp_df['location_x'], opp_df['location_y'],
                 opp_df['carry_end_location_x'], opp_df['carry_end_location_y'],
                 ax=axs[1], color=opp_color, width=line_width, alpha=line_alpha)
    # Apply black stroke outline
    arrows.set_path_effects(arrow_outline)
    axs[1].set_title(opp_summary, fontsize=10, color=opp_color, fontweight='bold')

    st.pyplot(fig, use_container_width=True)
