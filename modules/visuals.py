import plotly.graph_objects as go
import streamlit as st
import numpy as np


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
    
def plot_xg_histograms(binned_for, binned_against, goals_scored, goals_conceded, bin_width=10):
    x = range(len(binned_for)) * bin_width
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=binned_for, name=binned_for.name, marker_color='green'))
    fig.add_trace(go.Bar(x=x, y=binned_against, name=binned_against.name, marker_color='red'))

    # Add dashed lines for goals
    for line in goal_lines(goals_scored, bin_width=bin_width, color='goldenrod', name='Goal Scored'):
        fig.add_shape(line)
    for line in goal_lines(goals_conceded, bin_width=bin_width, color='white', name='Goal Conceded'):
        fig.add_shape(line)

    # Add dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='goldenrod', width=1.5, dash='dash'),
                             name='Goal Scored'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='white', width=1.5, dash='dash'),
                             name='Goal Conceded'))
    
    fig.update_layout(
        title=f"{binned_for.name} and {binned_against.name} Histogram",
        xaxis_title="Match Time (mins)",
        yaxis_title="value",
        barmode='group',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_rolling_xg(rolling_for, rolling_against, goals_scored, goals_conceded, bin_width=10):
    x = range(len(rolling_for))  * bin_width
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=rolling_for, name=rolling_for.name, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=rolling_against, name=rolling_against.name, line=dict(color='red')))

    # Add vertical lines for goals
    for line in goal_lines(goals_scored, bin_width=bin_width, color='goldenrod', name='Goal Scored'):
        fig.add_shape(line)
    for line in goal_lines(goals_conceded, bin_width=bin_width, color='white', name='Goal Conceded'):
        fig.add_shape(line)

    # Add dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='goldenrod', width=1.5, dash='dash'),
                             name='Goal Scored'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='white', width=1.5, dash='dash'),
                             name='Goal Conceded'))
    
    fig.update_layout(
        title=f"{rolling_for.name} and {rolling_against.name}",
        xaxis_title="Match Time (mins)",
        yaxis_title="Rolling value",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)