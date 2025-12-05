import dash
from dash import dcc, html, Input, Output, ALL, ctx, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

MAP_STYLE_LIGHT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
MAP_STYLE_DARK = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
MAP_STYLE_DARK_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"

METRICS = ["crime_rate", "employment_rate", "gdp_per_capita", "cost_index"]
DEFAULT_CENTER = {"lat": 48.5, "lon": 9.0, "zoom": 3}
CITY_ZOOM = 10


def load_data():
    db_path = "heterogeneous_cities.db"
    if not os.path.exists(db_path):
        print("⚠️ Warning: Database not found.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT name, country, lat, lon, crime_rate, employment_rate, gdp_per_capita, cost_index FROM cities",
        conn,
    )
    conn.close()
    df = df.dropna(subset=["lat", "lon"]).copy()

    for col in METRICS:
        df[col] = df[col].fillna(df[col].median())
        df[col + "_norm"] = MinMaxScaler().fit_transform(df[[col]])

    return df


def create_slider(slider_id, label):
    return [
        html.Label(label, className="mt-2" if slider_id != "w_cost" else ""),
        dcc.Slider(
            id=slider_id,
            min=0,
            max=2,
            step=0.1,
            value=1.0,
            marks={0: "0", 1: "1", 2: "2"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ]


def get_common_layout(title, template):
    grid_color = "#e0e0e0" if template == "plotly" else "#2a3f5f"
    line_color = "#d0d0d0" if template == "plotly" else "#4a5f7f"
    return dict(
        title=title,
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor=grid_color, linecolor=line_color, zerolinecolor=line_color
        ),
        yaxis=dict(
            gridcolor=grid_color, linecolor=line_color, zerolinecolor=line_color
        ),
    )


def fig_empty(template, title):
    fig = go.Figure()
    fig.update_layout(**get_common_layout(title, template))
    return fig


def create_horizontal_safety_bar(df, template):
    if df.empty:
        return fig_empty(template, "Safety (No Data)")
    top10 = df.head(10).copy()
    safety = 1 - top10["crime_rate_norm"]

    fig = go.Figure(
        go.Bar(x=safety, y=top10["name"], orientation="h", marker_color="indianred")
    )

    layout = get_common_layout("Top 10 Safety (Inverse Crime)", template)
    layout["yaxis"]["autorange"] = "reversed"
    layout["xaxis"]["title"] = "Safety Score"
    fig.update_layout(**layout)
    return fig


def create_simple_scatter(df, template):
    if df.empty:
        return fig_empty(template, "GDP vs Cost (No Data)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["gdp_per_capita"],
            y=df["cost_index"],
            mode="markers",
            marker=dict(color="steelblue", size=7, opacity=0.7),
            text=df["name"],
            hovertemplate="<b>%{text}</b><br>GDP: %{x:.0f}<br>Cost Index: %{y:.2f}<extra></extra>",
            name="Cities",
        )
    )

    # Trend line
    if len(df) > 2:
        try:
            x = df["gdp_per_capita"].values
            y = df["cost_index"].values
            coef = np.polyfit(x, y, 1)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=coef[0] * x + coef[1],
                    mode="lines",
                    line=dict(color="orange", dash="dash"),
                    name="Trend",
                )
            )
        except:
            pass

    layout = get_common_layout("GDP vs Cost Index", template)
    layout["xaxis"]["title"] = "GDP per Capita"
    layout["yaxis"]["title"] = "Cost Index"
    fig.update_layout(**layout)
    return fig


def create_score_histogram(df, template):
    if df.empty:
        return fig_empty(template, "Score Histogram (No Data)")
    fig = go.Figure(
        go.Histogram(
            x=df["score"],
            xbins=dict(start=0, end=10, size=0.5),
            marker_color="royalblue",
        )
    )

    layout = get_common_layout("Score Distribution", template)
    layout["xaxis"].update(title="Score", range=[0, 10])
    layout["yaxis"]["title"] = "Count"
    layout["bargap"] = 0.05
    fig.update_layout(**layout)
    return fig


def create_quartile_line(df, template):
    if df.empty:
        return fig_empty(template, "Avg Employment per Quartile (No Data)")
    df_q = df.copy()
    df_q["quartile"] = pd.qcut(df_q["score"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    agg = df_q.groupby("quartile", observed=False).employment_rate.mean().reset_index()

    fig = go.Figure(
        go.Scatter(
            x=agg["quartile"],
            y=agg["employment_rate"],
            mode="lines+markers",
            line=dict(color="darkorange"),
        )
    )

    layout = get_common_layout("Average Employment Rate by Score Quartile", template)
    layout["xaxis"]["title"] = "Score Quartile"
    layout["yaxis"]["title"] = "Employment Rate"
    fig.update_layout(**layout)
    return fig


def create_podium_chart(df, template):
    if df.empty or len(df) < 5:
        return fig_empty(template, "Top 5 Podium (No Data)")

    top5 = df.head(5).reset_index(drop=True)
    colors = ["#FFD700", "#C0C0C0", "#B87333", "#ABD2F3", "#ABD2F3"]
    fig = go.Figure(layout=dict(template=template))
    sorted_indices = top5["score"].argsort()[::-1]
    positions = [(0, 0), (0.6, 0), (1.2, 0), (0.3, 0.6), (0.9, 0.6)]

    max_score = top5["score"].max()
    min_score = top5["score"].min()
    z_range = max_score - min_score
    annotations = []

    for rank_idx in sorted_indices:
        x_pos, y_pos = positions[rank_idx]
        city = top5.iloc[rank_idx]
        z_height = city["score"]
        bar_size = 0.25

        # Bar
        fig.add_trace(
            go.Mesh3d(
                x=[
                    x_pos - bar_size,
                    x_pos - bar_size,
                    x_pos + bar_size,
                    x_pos + bar_size,
                ]
                * 2,
                y=[
                    y_pos - bar_size,
                    y_pos + bar_size,
                    y_pos + bar_size,
                    y_pos - bar_size,
                ]
                * 2,
                z=[0] * 4 + [z_height] * 4,
                alphahull=0,
                color=colors[int(rank_idx)],
                opacity=1.0,
                flatshading=True,
                hovertemplate=f"<b>{city['name']}</b><br>Rank: #{rank_idx+1}<br>Score: {z_height:.2f}<extra></extra>",
                showlegend=False,
            )
        )

        # Annotation
        medal = ["1st", "2nd", "3rd", "4th", "5th"][rank_idx]
        annotations.append(
            dict(
                showarrow=False,
                x=x_pos,
                y=y_pos,
                z=z_height + 0.05,
                text=f"<b>{medal} {city['name']}</b><br>{z_height:.2f}",
                font=dict(
                    size=11, color="white" if template == "plotly_dark" else "black"
                ),
                bgcolor=(
                    "rgba(0,0,0,0.7)"
                    if template == "plotly_dark"
                    else "rgba(255,255,255,0.85)"
                ),
                bordercolor=colors[rank_idx],
                borderwidth=2,
                borderpad=4,
                xanchor="center",
                yanchor="bottom",
            )
        )

    fig.update_layout(
        title="🏆 Top 5 Cities Podium",
        scene=dict(
            xaxis=dict(visible=False, range=[-0.3, 1.5]),
            yaxis=dict(visible=False, range=[-0.8, 1.2]),
            zaxis=dict(
                visible=False,
                range=[min_score - z_range * 0.5, max_score + max(z_range * 0.4, 0.6)],
            ),
            annotations=annotations,
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=1, y=1, z=1),
                up=dict(z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=0.8, y=0.8, z=2.5),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=75, b=0),
        uirevision="constant",
        showlegend=False,
    )
    return fig


df_master = load_data()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "City Recommender"

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1("🌆 City Recommender", className="text-center mb-1"),
                    html.P(
                        "An interactive city recommendation system based on weighted socioeconomic factors.",
                        className="text-center mb-0 small-text",
                    ),
                ]
            ),
            style={"flexShrink": "0", "padding": "10px 10px 0 10px"},
        ),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H2("Model Settings", className="mb-0"),
                                dbc.Switch(
                                    id="theme-switch",
                                    label="Dark Mode",
                                    value=False,
                                    className="ms-2",
                                ),
                            ],
                            className="d-flex justify-content-between align-items-center mb-3",
                        ),
                        html.P("Adjust the weights of each factor:"),
                        *create_slider("w_cost", "Cost weight (penalizes)"),
                        *create_slider("w_crime", "Crime weight (penalizes)"),
                        *create_slider("w_gdp", "GDP weight (benefits)"),
                        *create_slider("w_emp", "Employment weight (benefits)"),
                        html.Hr(),
                        html.H4("🏆 Ranking", className="mb-3"),
                        html.Div(id="top-10-list", className="scrollable-list"),
                    ],
                    width=3,
                    className="sidebar-container p-4",
                ),
                # Map
                dbc.Col(
                    [
                        html.Div(
                            dcc.Graph(
                                id="map-graph",
                                style={"height": "100%", "width": "100%"},
                                config={"scrollZoom": True},
                            ),
                            className="graph-card h-100",
                        ),
                    ],
                    className="p-0 h-100",
                ),
            ],
            className="g-0 main-row",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Open City Comparator",
                        id="open-city-comparator-btn",
                        color="primary",
                        n_clicks=0,
                        className="shadow-btn",
                    ),
                    width="auto",
                    className="p-3 text-center",
                ),
            ],
            className="justify-content-center mb-3",
        ),
        dbc.Row(
            [
                html.Div(
                    dcc.Graph(
                        id="podium-graph",
                        style={"height": "60vh", "pointerEvents": "none"},
                        config={"scrollZoom": False, "displayModeBar": False},
                    ),
                    className="graph-card podium-container",
                ),
            ],
            className="p-0 m-2",
        ),
        # Graphs Grid
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="crime-graph", className="graph-h40"),
                        className="graph-card",
                    ),
                    className="p-0",
                ),
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="emp-graph", className="graph-h40"),
                        className="graph-card",
                    ),
                    className="p-0",
                ),
            ],
            className="graph-row",
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="gdp-graph", className="graph-h40"),
                        className="graph-card",
                    ),
                    className="p-0",
                ),
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="cost-graph", className="graph-h40"),
                        className="graph-card",
                    ),
                    className="p-0",
                ),
            ],
            className="graph-row pb-5",
        ),
        # Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("City Comparator")),
                dbc.ModalBody(
                    [
                        html.Div(
                            id="city-dropdowns-container",
                            children=[
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id={"type": "city-dropdown", "index": 0},
                                            placeholder="Select City 1",
                                            style={"width": "250px"},
                                        ),
                                        dcc.Dropdown(
                                            id={"type": "city-dropdown", "index": 1},
                                            placeholder="Select City 2",
                                            style={"width": "250px"},
                                        ),
                                        dbc.Button(
                                            "+",
                                            id="add-city-dropdown-btn",
                                            size="sm",
                                            color="primary",
                                            n_clicks=0,
                                        ),
                                    ],
                                    className="d-flex gap-2 align-items-center mb-4",
                                ),
                            ],
                        ),
                        html.Div(
                            dcc.Graph(
                                id="comparison-radar",
                                style={"height": "500px"},
                                config={"displayModeBar": False},
                            ),
                            className="graph-card",
                        ),
                    ]
                ),
            ],
            id="city-comparator-modal",
            is_open=False,
            centered=True,
            size="xl",
            backdrop=True,
        ),
        html.Div(id="theme-trigger", style={"display": "none"}),
    ],
    fluid=True,
    className="app-root",
    id="app-root",
)


@app.callback(
    [
        Output("map-graph", "figure"),
        Output("top-10-list", "children"),
        Output("podium-graph", "figure"),
        Output("crime-graph", "figure"),
        Output("emp-graph", "figure"),
        Output("gdp-graph", "figure"),
        Output("cost-graph", "figure"),
    ],
    [
        Input("w_cost", "value"),
        Input("w_crime", "value"),
        Input("w_gdp", "value"),
        Input("w_emp", "value"),
        Input("theme-switch", "value"),
        Input({"type": "city-btn", "index": ALL}, "n_clicks"),
    ],
)
def update_app(w_cost, w_crime, w_gdp, w_emp, dark_mode, btn_clicks):
    template = "plotly_dark" if dark_mode else "plotly"

    df = df_master.copy()
    df["score"] = (
        -w_crime * df["crime_rate_norm"]
        + w_emp * df["employment_rate_norm"]
        + w_gdp * df["gdp_per_capita_norm"]
        - w_cost * df["cost_index_norm"]
    )
    df["score"] = MinMaxScaler(feature_range=(0, 10)).fit_transform(df[["score"]])
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Map Logic
    lat_center, lon_center, zoom_level = (
        DEFAULT_CENTER["lat"],
        DEFAULT_CENTER["lon"],
        DEFAULT_CENTER["zoom"],
    )
    triggered_id = ctx.triggered_id

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "city-btn":
        target = df_master.loc[df_master.index == triggered_id["index"]]
        if not target.empty:
            lat_center, lon_center, zoom_level = (
                target.iloc[0]["lat"],
                target.iloc[0]["lon"],
                CITY_ZOOM,
            )


    if dark_mode:
        my_layers = [
            {"sourcetype": "raster", "source": [MAP_STYLE_DARK], "below": "traces"},
            {"sourcetype": "raster", "source": [MAP_STYLE_DARK_LABELS], "below": "traces"},
        ]
    else:
        my_layers = [{"sourcetype": "raster", "source": [MAP_STYLE_LIGHT], "below": "traces"}]

    fig_map = go.Figure(layout=dict(template=template))

    others, top10 = df_sorted.iloc[10:], df_sorted.head(10)

    if not others.empty:
        fig_map.add_trace(
            go.Scattermap(
                lat=others.lat,
                lon=others.lon,
                mode="markers",
                marker=dict(size=10, color="rgb(55, 126, 184)", opacity=0.8),
                text=others.name + ", " + others.country,
                customdata=others.score,
                hovertemplate="<b>%{text}</b><br>Score: %{customdata:.2f}<extra></extra>",
                name="Cities",
            )
        )

    fig_map.add_trace(
        go.Scattermap(
            lat=top10.lat,
            lon=top10.lon,
            mode="markers",
            marker=dict(size=15, color="gold", opacity=0.8),
            text=top10.name + ", " + top10.country,
            customdata=top10.score,
            hovertemplate="<b>%{text}</b><br>Rank: Top 10<br>Score: %{customdata:.2f}<extra></extra>",
            name="Top 10",
        )
    )

    fig_map.update_layout(
        map=dict(
            center={"lat": lat_center, "lon": lon_center},
            zoom=zoom_level,
            style="open-street-map",
            layers=my_layers,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
        uirevision="constant" if not isinstance(triggered_id, dict) else None,
    )

    # Ranking List
    list_items = []
    for i, row in df_sorted.iterrows():
        original_idx = df_master.index[df_master["name"] == row["name"]][0]
        list_items.append(
            dbc.ListGroupItem(
                html.Div(
                    [
                        html.H5(f"#{i+1} {row['name']}", className="mb-1"),
                        html.Small(f"{row['country']} | Score: {row['score']:.2f}"),
                    ],
                    className="d-flex w-100 justify-content-between align-items-center",
                ),
                action=True,
                id={"type": "city-btn", "index": int(original_idx)},
                className=f"mb-1 ranking-item {'bg-top' if i < 3 else ''}",
            )
        )

    return (
        fig_map,
        list_items,
        create_podium_chart(df_sorted, template),
        create_horizontal_safety_bar(df_sorted, template),
        create_simple_scatter(df_sorted, template),
        create_score_histogram(df_sorted, template),
        create_quartile_line(df_sorted, template),
    )


@app.callback(
    Output("city-comparator-modal", "is_open"),
    [Input("open-city-comparator-btn", "n_clicks")],
    [State("city-comparator-modal", "is_open")],
)
def toggle_city_comparator(n, is_open):
    return not is_open if n else is_open


@app.callback(
    Output({"type": "city-dropdown", "index": ALL}, "options"),
    Input("city-dropdowns-container", "children"),
)
def populate_dropdowns(children):
    opts = [
        {"label": f"{r['name']}, {r['country']}", "value": r["name"]}
        for _, r in df_master.iterrows()
    ]
    return [opts] * len(
        [c for c in children[0]["props"]["children"] if "Dropdown" in str(c)]
    )


@app.callback(
    [
        Output("city-dropdowns-container", "children"),
        Output("add-city-dropdown-btn", "disabled"),
    ],
    Input("add-city-dropdown-btn", "n_clicks"),
    State("city-dropdowns-container", "children"),
    prevent_initial_call=True,
)
def add_city_dropdown(n, children):
    if not n:
        return children, False
    container = children[0]["props"]["children"]
    count = len([c for c in container if "Dropdown" in str(c)])

    if count < 4:
        container.insert(
            -1,
            dcc.Dropdown(
                id={"type": "city-dropdown", "index": count},
                options=[],
                placeholder=f"Select City {count + 1}",
                style={"width": "250px"},
            ),
        )
    return [children[0]], count + 1 >= 4


@app.callback(
    Output("comparison-radar", "figure"),
    [
        Input({"type": "city-dropdown", "index": ALL}, "value"),
        Input("theme-switch", "value"),
    ],
)
def update_comparison_radar(cities, dark_mode):
    template = "plotly_dark" if dark_mode else "plotly"
    fig = go.Figure(layout=dict(template=template))
    cities = [c for c in cities if c]
    cats = ["Crime Rate", "Employment Rate", "GDP per Capita", "Cost Index"]

    grid_color = "#e0e0e0" if template == "plotly" else "#2a3f5f"
    polar_layout = dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            showticklabels=False,
            gridcolor=grid_color,
            linecolor=grid_color,
        ),
        angularaxis=dict(gridcolor=grid_color, linecolor=grid_color),
        bgcolor="rgba(0,0,0,0)",
    )

    if not cities:
        fig.add_trace(
            go.Scatterpolar(r=[0] * 4, theta=cats, line=dict(color="rgba(0,0,0,0)"))
        )
        fig.update_layout(
            polar=polar_layout,
            showlegend=False,
            title="Select cities to compare",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    for city in cities:
        row = df_master[df_master["name"] == city].iloc[0]
        values = [
            1 - row["crime_rate_norm"],
            row["employment_rate_norm"],
            row["gdp_per_capita_norm"],
            1 - row["cost_index_norm"],
        ]
        fig.add_trace(go.Scatterpolar(r=values, theta=cats, fill="toself", name=city))

    fig.update_layout(
        polar=polar_layout,
        showlegend=True,
        title="City Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


app.clientside_callback(
    """function(dark){ document.body.classList.toggle('dark-mode', dark); return ''; }""",
    Output("theme-trigger", "children"),
    Input("theme-switch", "value"),
)

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
