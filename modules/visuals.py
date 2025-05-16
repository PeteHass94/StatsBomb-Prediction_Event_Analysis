import plotly.graph_objects as go
import streamlit as st
import numpy as np

def plot_goals_histogram(goals_column):
    goals_data = [goal for sublist in goals_column for goal in sublist]

    if not isinstance(goals_data, list) or not all(isinstance(goal, dict) and 'period' in goal and 'timeValue' in goal for goal in goals_data):
        raise TypeError("goals_data must be a list of dictionaries with 'period' and 'timeValue' keys.")

    # Split goals
    period_1_goals = [goal['timeValue'] for goal in goals_data if goal['period'] == 1]
    period_2_goals = [goal['timeValue'] for goal in goals_data if goal['period'] == 2]

    fig = go.Figure()

    # Period 1
    fig.add_trace(go.Histogram(
        x=period_1_goals,
        xbins=dict(start=0, end=45, size=10),
        name='Period 1',
        marker_color='blue',
        opacity=0.75
    ))

    # Period 2
    fig.add_trace(go.Histogram(
        x=period_2_goals,
        xbins=dict(start=45, end=105, size=10),
        name='Period 2',
        marker_color='green',
        opacity=0.75
    ))

    # Add labels above each bin
    for trace in fig.data:
        counts, edges = np.histogram(trace.x, bins=np.arange(trace.xbins.start, trace.xbins.end + trace.xbins.size, trace.xbins.size))
        for i, count in enumerate(counts):
            if count > 0:
                x_pos = edges[i] + trace.xbins.size / 2
                fig.add_annotation(
                    x=x_pos,
                    y=count,
                    text=str(count),
                    showarrow=False,
                    font=dict(color='white'),
                    yanchor='bottom'
                )

    # Layout: shared single x-axis and y-axis only
    fig.update_layout(
        barmode='overlay',
        # bargap=0.2,
        # bargroupgap=0.1,
        xaxis=dict(title='Time (minutes)'),
        yaxis=dict(title=''),
        legend=dict(x=0.8, y=1.2),
    )

    st.plotly_chart(fig, use_container_width=True)
