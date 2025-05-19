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
    
def plot_xg_histograms(binned_for, binned_against, goals_scored, goals_conceded, bin_width=10, for_name="For", against_name="Against"):
    x = [i * bin_width for i in range(len(binned_for))]
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=binned_for, name=for_name, marker_color='green'))
    fig.add_trace(go.Bar(x=x, y=binned_against, name=against_name, marker_color='red'))

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
        title=f"{for_name} and {against_name} Histogram",
        xaxis_title="Match Time (mins)",
        yaxis_title="Value",
        barmode='group',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_rolling_xg(rolling_for, rolling_against, goals_scored, goals_conceded, bin_width=10, for_name="For", against_name="Against"):
    x = [i * bin_width for i in range(len(rolling_for))]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=rolling_for, name=for_name, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=rolling_against, name=against_name, line=dict(color='red')))

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
        title=f"{for_name} and {against_name} Rolling Line Chart",
        xaxis_title="Match Time (mins)",
        yaxis_title="Rolling value",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)