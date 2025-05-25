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
        range=[0, 100],
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
        range=[0, 100],
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
