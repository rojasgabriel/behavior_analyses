"""
Interactive Performance Dashboard for Behavioral Data Analysis

This dashboard provides an interactive interface to visualize performance summaries
for behavioral experiments. It replaces the static CLI-based script with a web-based
dashboard that allows real-time interaction with the data.

Launch with: python dashboard.py
"""

from labdata.schema import DecisionTask  # type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


def fetch_mouse_data(mouse: str, sessions_back: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch data for a specific mouse from the database."""
    data = pd.DataFrame(DecisionTask.TrialSet() & f"subject_name = '{mouse}'")
    if data.empty:
        return None
    
    ses_back = sessions_back if len(data) >= sessions_back else len(data)
    data = data.tail(ses_back)
    sesdata = data[data.session_name == data.session_name.iloc[-1]]
    
    return data, sesdata


def calculate_metrics(data: pd.DataFrame, sesdata: pd.DataFrame, ses_back: int) -> dict:
    """Calculate all performance metrics from the data."""
    # Calculate early withdrawal rate
    ew_rate = []
    for ses in data.itertuples(index=False):
        ew_trials = ~(np.isin(np.array(ses.response_values), [-1, 1]))
        ew_rate.append((ew_trials).sum() / ew_trials.shape[0])
    
    # Get valid trials (removing early withdrawal trials)
    valid_trials = sesdata.response_values.apply(lambda x: np.isin(x, [-1, 1])).values[0]
    stims = sesdata.intensity_values.values[0][valid_trials]
    responses = np.array(sesdata.response_values.values[0])[valid_trials]
    correct = sesdata.correct_values.values[0][valid_trials]
    react_times = sesdata.reaction_times.values[0][valid_trials]
    react_times = react_times[react_times < 2]
    
    # Calculate fraction correct and p(right) per stimulus
    unique_stims = np.unique(sesdata.intensity_values.values[0])
    frac_correct = []
    p_right = []
    for ustim in unique_stims:
        mask = stims == ustim
        right_mask = responses[mask] == 1
        frac_correct.append(correct[mask].sum() / correct[mask].shape[0])
        p_right.append(right_mask.sum() / right_mask.shape[0])
    
    xvalues = np.arange(0, ses_back, 1)
    
    return {
        'unique_stims': unique_stims,
        'frac_correct': frac_correct,
        'p_right': p_right,
        'xvalues': xvalues,
        'ew_rate': ew_rate,
        'react_times': react_times,
        'data': data,
        'sesdata': sesdata,
    }


def create_performance_figure(metrics: dict) -> go.Figure:
    """Create the interactive performance dashboard figure."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Fraction Correct per Stimulus',
            'Performance Over Sessions',
            'P(Right) per Stimulus',
            'Early Withdrawal Rate',
            'Reaction Times',
            'Trial Counts'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )
    
    # Plot 1: Fraction correct per stim
    fig.add_trace(
        go.Scatter(
            x=metrics['unique_stims'],
            y=metrics['frac_correct'],
            mode='lines+markers',
            name='Fraction Correct',
            marker=dict(color='dodgerblue', size=8),
            line=dict(color='dodgerblue'),
            showlegend=True,
        ),
        row=1, col=1
    )
    
    # Plot 2: Performance on easy trials
    fig.add_trace(
        go.Scatter(
            x=metrics['xvalues'],
            y=metrics['data']['performance_easy'][::-1],
            mode='lines+markers',
            name='Easy',
            marker=dict(color='violet', size=8),
            line=dict(color='violet'),
            opacity=0.8,
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=metrics['xvalues'],
            y=metrics['data']['performance'][::-1],
            mode='lines+markers',
            name='All',
            marker=dict(color='grey', size=8),
            line=dict(color='grey'),
            opacity=0.8,
        ),
        row=1, col=2
    )
    
    # Plot 3: P(right) per stim
    fig.add_trace(
        go.Scatter(
            x=metrics['unique_stims'],
            y=metrics['p_right'],
            mode='lines+markers',
            name='P(Right)',
            marker=dict(color='dodgerblue', size=8),
            line=dict(color='dodgerblue'),
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # Plot 4: Early withdrawal rate
    fig.add_trace(
        go.Scatter(
            x=[metrics['xvalues'][0], metrics['xvalues'][-1]],
            y=[0.5, 0.5],
            mode='lines',
            name='Threshold',
            line=dict(color='black', dash='dash'),
            showlegend=False,
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=metrics['xvalues'],
            y=metrics['ew_rate'],
            mode='lines+markers',
            name='E.W. Rate',
            marker=dict(color='crimson', size=8),
            line=dict(color='crimson'),
            opacity=0.8,
            showlegend=False,
        ),
        row=2, col=2
    )
    
    # Plot 5: Reaction times histogram
    fig.add_trace(
        go.Histogram(
            x=metrics['react_times'],
            nbinsx=20,
            name='Reaction Times',
            marker=dict(color='dodgerblue'),
            opacity=0.8,
            showlegend=False,
        ),
        row=3, col=1
    )
    
    # Plot 6: Trial counts (stacked bar chart)
    fig.add_trace(
        go.Bar(
            x=metrics['xvalues'],
            y=metrics['data']['n_trials'][::-1],
            name='Total',
            marker=dict(color='grey'),
            opacity=0.8,
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Bar(
            x=metrics['xvalues'],
            y=metrics['data']['n_with_choice'][::-1],
            name='With Choice',
            marker=dict(color='dodgerblue'),
            opacity=0.8,
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Bar(
            x=metrics['xvalues'],
            y=metrics['data']['n_correct'][::-1],
            name='Correct',
            marker=dict(color='deeppink'),
            opacity=0.8,
        ),
        row=3, col=2
    )
    
    # Update axes labels and ranges
    fig.update_xaxes(title_text="Stim Intensities", row=1, col=1)
    fig.update_yaxes(title_text="Fraction Correct", range=[0.3, 1], row=1, col=1)
    
    fig.update_xaxes(title_text="Sessions Back", row=1, col=2)
    fig.update_yaxes(title_text="Performance", range=[0.3, 1], row=1, col=2)
    
    fig.update_xaxes(title_text="Stim Intensities", row=2, col=1)
    fig.update_yaxes(title_text="P(Right)", range=[0, 1], row=2, col=1)
    
    fig.update_xaxes(title_text="Sessions Back", row=2, col=2)
    fig.update_yaxes(title_text="E.W. Rate", range=[0, 1], row=2, col=2)
    
    fig.update_xaxes(title_text="Reaction Times", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    fig.update_xaxes(title_text="Sessions Back", row=3, col=2)
    fig.update_yaxes(title_text="Trials", row=3, col=2)
    
    # Update layout
    session_name = metrics['sesdata'].session_name.values[0]
    subject_name = metrics['sesdata'].subject_name.values[0]
    
    fig.update_layout(
        height=900,
        title_text=f"<b>{subject_name}</b><br>{session_name}",
        title_x=0.1,
        title_xanchor='left',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
    )
    
    return fig


def get_available_mice() -> list[str]:
    """Fetch all available mice from the database."""
    try:
        data = pd.DataFrame(DecisionTask.TrialSet())
        if not data.empty and 'subject_name' in data.columns:
            return sorted(data['subject_name'].unique().tolist())
        return []
    except Exception as e:
        print(f"Error fetching mice: {e}")
        return []


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Performance Dashboard"

# Get available mice
available_mice = get_available_mice()

# Create the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Behavioral Performance Dashboard", className="text-center mb-4 mt-4"),
            html.Hr(),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Mouse:", className="fw-bold"),
            dcc.Dropdown(
                id='mouse-selector',
                options=[{'label': mouse, 'value': mouse} for mouse in available_mice],
                value=available_mice[0] if available_mice else None,
                placeholder="Select a mouse...",
                clearable=False,
            ),
        ], width=6),
        
        dbc.Col([
            html.Label("Sessions Back:", className="fw-bold"),
            dcc.Slider(
                id='sessions-slider',
                min=1,
                max=30,
                step=1,
                value=10,
                marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='error-message', className="alert alert-warning", style={'display': 'none'}),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=dcc.Graph(id='performance-graph', style={'height': '900px'}),
            ),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Interactive dashboard for visualizing behavioral performance metrics. "
                "Hover over plots for detailed information. Use the controls above to select "
                "different mice and adjust the number of sessions to display.",
                className="text-muted text-center mb-4"
            ),
        ])
    ]),
], fluid=True)


@app.callback(
    [Output('performance-graph', 'figure'),
     Output('error-message', 'children'),
     Output('error-message', 'style')],
    [Input('mouse-selector', 'value'),
     Input('sessions-slider', 'value')]
)
def update_graph(mouse: str, sessions_back: int):
    """Update the performance graph based on user selections."""
    if not mouse:
        return go.Figure(), "Please select a mouse.", {'display': 'block'}
    
    try:
        result = fetch_mouse_data(mouse, sessions_back)
        if result is None:
            return go.Figure(), f"No data found for mouse {mouse}.", {'display': 'block'}
        
        data, sesdata = result
        metrics = calculate_metrics(data, sesdata, sessions_back)
        fig = create_performance_figure(metrics)
        
        return fig, "", {'display': 'none'}
    
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        print(error_msg)
        return go.Figure(), error_msg, {'display': 'block'}


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Performance Dashboard...")
    print("=" * 60)
    print("\nDashboard will be available at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
