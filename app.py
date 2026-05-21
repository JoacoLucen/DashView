import os
import base64
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import threading

# Importación de módulos internos
from src.database_manager import DatabaseManager
from src.metrics import MetricsCalculator
from src.etl_pipeline import ETLPipeline

# =============================================================================
# 1. CONFIGURACIÓN CORPORATIVA Y ESTILOS
# =============================================================================
COLOR_PRIMARY = "#1F4E78"    
COLOR_ACCENT = "#2E75B6"     
COLOR_BACKGROUND = "#FFFFFF" 
COLOR_NEUTRAL_1 = "#F8F9FA"  
COLOR_NEUTRAL_2 = "#A6A6A6"  

FONT_FAMILY = "Segoe UI, Arial, sans-serif"

def apply_corporate_layout(fig, barmode=None, margin=None, hide_x_title=False, hide_y_title=False):
    """Aplica las reglas de UI/UX. Oculta títulos de ejes si son obvios."""
    default_margin = dict(l=120, r=40, t=60, b=80)
    if margin:
        default_margin.update(margin)
        
    fig.update_layout(
        font=dict(family=FONT_FAMILY, color="#333333"),
        plot_bgcolor=COLOR_BACKGROUND,
        paper_bgcolor=COLOR_BACKGROUND,
        margin=default_margin,
        barmode=barmode,
        colorway=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_NEUTRAL_2, "#4A90E2", "#6C757D"],
        title_font=dict(size=18, color=COLOR_PRIMARY, family=FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#F0F0F0', 
        title_text="" if hide_x_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11)
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#F0F0F0', 
        title_text="" if hide_y_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11)
    )
    return fig

STYLE_KPI = {
    "backgroundColor": COLOR_NEUTRAL_1,
    "borderRadius": "12px",
    "padding": "25px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
    "border": "1px solid #EAEAEA",
    "textAlign": "center",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center"
}

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"],
    title="DashView Analytics",
    suppress_callback_exceptions=True
)

db_manager = DatabaseManager()
metrics_calc = MetricsCalculator(db_manager)
pipeline_etl = ETLPipeline()

# Garantizar estructura de carpetas al arranque
for path in [pipeline_etl.staging_dir, pipeline_etl.EXTRACTED_DIR]:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# 2. LAYOUT
# =============================================================================
app.layout = html.Div(style={"backgroundColor": COLOR_BACKGROUND, "fontFamily": FONT_FAMILY}, children=[
    dcc.Store(id="processing-status", data={"status": "ready"}),
    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),

    html.Div(
        id="main-dashboard-container",
        style={"display": "block" if os.path.exists(pipeline_etl.target_db_path) else "none"},
        children=[
            dbc.NavbarSimple(
                brand="DashView Analytics",
                brand_style={"fontWeight": "bold", "fontSize": "24px"},
                children=[
                    dbc.Button([html.I(className="bi bi-cloud-upload me-2"), "Cambiar Datos"], 
                               id="btn-load-new", color="light", size="sm", className="ms-2 text-dark border")
                ],
                color=COLOR_PRIMARY, dark=True, fluid=True, className="mb-4 shadow-sm"
            ),

            dbc.Container(fluid=True, children=[
                html.Div(id="global-alert-container"),
                
                # PANEL DE FILTROS (STICKY)
                html.Div(style={"position": "sticky", "top": "10px", "zIndex": 1000}, children=[
                    dbc.Card(className="mb-4 border-0 shadow-lg", children=dbc.CardBody([
                        dbc.Row([
                            dbc.Col(md=2, children=[
                                html.Label("Período", className="small fw-bold text-muted"),
                                dcc.RangeSlider(id="filter-period", min=2010, max=2025, step=1, value=[2010, 2025],
                                                marks={str(y): str(y) for y in range(2010, 2026, 3)})
                            ]),
                            dbc.Col(md=2, children=[
                                html.Label("Plataforma", className="small fw-bold text-muted"),
                                dcc.Dropdown(id="filter-source", multi=True, placeholder="Todas")
                            ]),
                            dbc.Col(md=2, children=[
                                html.Label("Empresa", className="small fw-bold text-muted"),
                                dcc.Dropdown(id="filter-company", multi=True, placeholder="Todas")
                            ]),
                            dbc.Col(md=2, children=[
                                html.Label("Producto", className="small fw-bold text-muted"),
                                dcc.Dropdown(id="filter-product", multi=True, placeholder="Todos")
                            ]),
                            dbc.Col(md=2, children=[
                                html.Label("Acción", className="small fw-bold text-muted"),
                                dcc.Dropdown(id="filter-action", multi=True, placeholder="Todas")
                            ]),
                            dbc.Col(md=2, children=[
                                html.Label("Sentimiento", className="small fw-bold text-muted"),
                                dcc.RadioItems(id="filter-sentiment", inline=True, className="mt-1",
                                               options=[{"label": " Todos", "value": "ALL"}, {"label": " Pos", "value": "Positive"}, {"label": " Neg", "value": "Negative"}], 
                                               value="ALL", inputStyle={"marginLeft": "10px"})
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col(md=12, className="text-end", children=[
                                dbc.Button("✕ Limpiar Filtros", id="btn-clear-filters", color="link", size="sm", className="text-muted p-0 mt-2")
                            ])
                        ])
                    ])),
                ]),

                dbc.Tabs(
                    id="tabs-stakeholders", 
                    active_tab="tab-marketing", 
                    children=[
                        dbc.Tab(label="Marketing", tab_id="tab-marketing"),
                        dbc.Tab(label="Dirección General", tab_id="tab-dir-general"),
                        dbc.Tab(label="Retención y Facturación", tab_id="tab-retencion"),
                        dbc.Tab(label="Equipo de Producto / App", tab_id="tab-producto"),
                    ]
                ),

                dcc.Loading(
                    id="loading-dashboard", type="dot", color=COLOR_PRIMARY,
                    children=html.Div(id="tab-content-container", className="mt-4 pb-5")
                )
            ])
        ]
    ),

    # OVERLAY INICIAL
    html.Div(
        id="welcome-overlay",
        style={
            "position": "fixed", "top": 0, "left": 0, "width": "100vw", "height": "100vh",
            "backgroundColor": "rgba(255, 255, 255, 0.98)", "zIndex": 9999, "display": "flex", 
            "alignItems": "center", "justifyContent": "center"
        } if not os.path.exists(pipeline_etl.target_db_path) else {"display": "none"},
        children=[
            dbc.Card(style={"width": "500px", "borderRadius": "16px"}, className="shadow-lg border-0", children=dbc.CardBody([
                html.Div(style={"textAlign": "center"}, children=[
                    html.H1("DashView", style={"color": COLOR_PRIMARY, "fontWeight": "900", "marginBottom": "10px"}),
                    html.P("Inteligencia de Datos de Clientes", className="text-muted mb-4"),
                    dcc.Loading(type="circle", color=COLOR_PRIMARY, children=[
                        dcc.Upload(
                            id="upload-data-file", accept=".zip",
                            style={'width': '100%', 'height': '120px', 'lineHeight': '120px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': COLOR_PRIMARY, 'borderRadius': '12px', 'backgroundColor': COLOR_NEUTRAL_1, 'cursor': 'pointer'},
                            children=html.Div(["Cargar archivo ", html.B(".ZIP")])
                        ),
                        html.Div(id="upload-status-message", className="mt-3 fw-bold", style={"color": COLOR_PRIMARY})
                    ])
                ])
            ]))
        ]
    )
])

# =============================================================================
# 3. CALLBACKS
# =============================================================================

@app.callback(
    Output("filter-period", "value"),
    Output("filter-source", "value"),
    Output("filter-company", "value"),
    Output("filter-product", "value"),
    Output("filter-action", "value"),
    Output("filter-sentiment", "value"),
    Input("btn-clear-filters", "n_clicks"),
    prevent_initial_call=True
)
def clear_filters(n):
    return [2010, 2025], None, None, None, None, "ALL"

@app.callback(
    Output("welcome-overlay", "style", allow_duplicate=True),
    Output("main-dashboard-container", "style", allow_duplicate=True),
    Input("btn-load-new", "n_clicks"),
    prevent_initial_call=True
)
def reset_system(n):
    if n:
        return {"display": "flex", "position": "fixed", "top": 0, "left": 0, "width": "100vw", "height": "100vh", "backgroundColor": "rgba(255, 255, 255, 0.98)", "zIndex": 9999, "alignItems": "center", "justifyContent": "center"}, {"display": "none"}
    return dash.no_update, dash.no_update

@app.callback(
    Output("tab-content-container", "children"),
    Output("welcome-overlay", "style", allow_duplicate=True),
    Output("main-dashboard-container", "style", allow_duplicate=True),
    Input("tabs-stakeholders", "active_tab"),
    Input("filter-period", "value"),
    Input("filter-source", "value"),
    Input("filter-company", "value"),
    Input("filter-product", "value"),
    Input("filter-action", "value"),
    Input("filter-sentiment", "value"),
    Input("processing-status", "data"),
    prevent_initial_call='initial_duplicate'
)
def update_view(tab, period, sources, companies, products, actions, sentiment, proc_status):
    if not os.path.exists(pipeline_etl.target_db_path):
        return html.Div(), dash.no_update, {"display": "none"}
    if proc_status.get("status") == "processing":
        return dash.no_update, dash.no_update, dash.no_update

    filters = {"period": period, "sources": sources, "companies": companies, "products": products, "actions": actions, "sentiment": sentiment}

    try:
        if tab == "tab-marketing": view = render_marketing(filters)
        elif tab == "tab-dir-general": view = render_general_direction(filters)
        elif tab == "tab-retencion": view = render_retention(filters)
        elif tab == "tab-producto": view = render_product_team(filters)
        else: view = html.Div()
        return view, {"display": "none"}, {"display": "block"}
    except Exception as e:
        return html.Div([dbc.Alert(f"Aviso del Sistema: {str(e)}", color="warning")]), {"display": "none"}, {"display": "block"}

# ---------------- RENDERS ----------------

def render_marketing(filters):
    kpis = metrics_calc.get_marketing_kpis(filters)
    df_sent = metrics_calc.get_sentiment_by_channel(filters)
    df_peaks = metrics_calc.get_monthly_activity_peaks(filters)
    df_impact = metrics_calc.get_source_impact(filters)

    fig_pie = px.pie(kpis["pie_data"], values='Cantidad', names='Tipo', hole=0.5, title="Engagement: Usuarios Activos vs Pasivos")
    fig_pie = apply_corporate_layout(fig_pie)

    fig_sent = px.bar(df_sent, x='avg_sentiment', y='source', orientation='h', title="Puntaje de Sentimiento por Canal")
    fig_sent = apply_corporate_layout(fig_sent, margin=dict(l=150), hide_x_title=True, hide_y_title=True)

    fig_peaks = px.bar(df_peaks, x='mes_label', y='volumen', title="Estacionalidad Mensual de Actividad")
    fig_peaks.update_traces(marker_color=COLOR_ACCENT)
    fig_peaks = apply_corporate_layout(fig_peaks, hide_x_title=True, hide_y_title=True)

    fig_impact = px.bar(df_impact, x='source', y=['pct_positive', 'pct_negative'], title="Impacto Comportamental por Plataforma (%)", barmode='group')
    fig_impact = apply_corporate_layout(fig_impact, barmode='group', hide_x_title=True, hide_y_title=True)

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.H2(f"{kpis['total_signals']:,}", style={"color": COLOR_PRIMARY}), html.P("Señales Totales", className="text-muted small fw-bold text-uppercase")], style=STYLE_KPI), md=6),
            dbc.Col(html.Div([html.H2(f"{kpis['pct_activos']}%", style={"color": COLOR_PRIMARY}), html.P("Usuarios Activos", className="text-muted small fw-bold text-uppercase")], style=STYLE_KPI), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pie), md=6),
            dbc.Col(dcc.Graph(figure=fig_sent), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_peaks), md=6),
            dbc.Col(dcc.Graph(figure=fig_impact), md=6),
        ])
    ])

def render_general_direction(filters):
    kpis = metrics_calc.get_general_direction_kpis(filters)
    df_bench = metrics_calc.get_competitive_benchmark(filters)
    df_heat = metrics_calc.get_company_product_heatmap(filters)

    fig_churn = px.pie(kpis["distribucion"], values='cantidad', names='causa_label', hole=0.4, title="Motivos de Salida (Churn)")
    fig_churn = apply_corporate_layout(fig_churn)

    fig_bench = px.bar(df_bench, x='avg_sentiment', y='company', orientation='h', title="Benchmark de Sentimiento por Empresa")
    fig_bench = apply_corporate_layout(fig_bench, margin=dict(l=150), hide_x_title=True, hide_y_title=True)

    if not df_heat.empty:
        pivot = df_heat.pivot(index='company', columns='product_service', values='avg_sentiment')
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#EF4444"], [0.5, "#FFFFFF"], [1, "#22C55E"]],
            zmid=0, colorbar=dict(title="Score")
        ))
        fig_heat.update_layout(title="Matriz de Riesgo: Empresa vs Producto", font=dict(family=FONT_FAMILY))
    else:
        fig_heat = go.Figure().update_layout(title="Sin datos para la matriz")
    
    fig_heat = apply_corporate_layout(fig_heat, margin=dict(l=150, b=100), hide_x_title=True, hide_y_title=True)

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.H2(f"{kpis['total_churn']:,}", style={"color": COLOR_PRIMARY}), html.P("Total Churn", className="text-muted small fw-bold text-uppercase")], style=STYLE_KPI), md=4),
            dbc.Col(dcc.Graph(figure=fig_churn), md=8),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_bench), md=6),
            dbc.Col(dcc.Graph(figure=fig_heat), md=6),
        ])
    ])

def render_retention(filters):
    esc_rate = metrics_calc.get_escalation_rate(filters)
    avg_cycle = metrics_calc.get_average_behavior_cycle(filters)
    df_radar = metrics_calc.get_product_risk_radar(filters)
    df_topics = metrics_calc.get_complaint_topics(filters)
    df_map = metrics_calc.get_state_intensity_map(filters)

    fig_radar = go.Figure()
    if not df_radar.empty:
        fig_radar.add_trace(go.Scatterpolar(r=df_radar['score'], theta=df_radar['product'], fill='toself', line_color=COLOR_PRIMARY))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Radar de Riesgo por Producto", font=dict(family=FONT_FAMILY))

    fig_topics = px.bar(df_topics, x='Frecuencia', y='Topic', orientation='h', title="Temas Frecuentes en Quejas")
    fig_topics.update_traces(marker_color=COLOR_ACCENT)
    fig_topics = apply_corporate_layout(fig_topics, margin=dict(l=180), hide_x_title=True, hide_y_title=True)

    fig_map = px.bar(df_map, x='estado', y='quejas', title="Volumen de Quejas por Región")
    fig_map = apply_corporate_layout(fig_map, hide_x_title=True, hide_y_title=True)

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.H2(f"{esc_rate}%", style={"color": COLOR_PRIMARY}), html.P("Tasa de Escalada", className="text-muted small fw-bold text-uppercase")], style=STYLE_KPI), md=6),
            dbc.Col(html.Div([html.H2(f"{avg_cycle} días", style={"color": COLOR_PRIMARY}), html.P("Ciclo de Retención Promedio", className="text-muted small fw-bold text-uppercase"), html.P("⚠ Aproximación por producto", className="small text-muted mb-0")], style=STYLE_KPI), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_radar), md=6),
            dbc.Col(dcc.Graph(figure=fig_topics), md=6),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_map), md=12)])
    ])

def render_product_team(filters):
    # FIX: Evitar colisión de filtros globales con filtros de plataforma para App
    device_filters = {k: v for k, v in filters.items() if k != 'sources'}
    df_dev = metrics_calc.get_device_usage_comparison(device_filters)
    df_nlp = metrics_calc.get_app_reviews_nlp(device_filters)
    df_yoy = metrics_calc.get_yoy_volume_and_sentiment(device_filters)

    banner = dbc.Alert("Vista exclusiva: AppStore y GooglePlay.", color="info", className="mb-4 shadow-sm border-0")

    fig_dev = go.Figure()
    if not df_dev.empty:
        fig_dev.add_trace(go.Bar(y=df_dev['source'], x=df_dev['avg_sentiment'], name='Sentimiento', orientation='h', marker_color=COLOR_PRIMARY))
        if 'avg_rating' in df_dev.columns:
            fig_dev.add_trace(go.Bar(y=df_dev['source'], x=df_dev['avg_rating'], name='Rating', orientation='h', marker_color=COLOR_ACCENT))
    fig_dev.update_layout(title="iOS vs Android: Experiencia Comparada", barmode='group')
    fig_dev = apply_corporate_layout(fig_dev, margin=dict(l=120), hide_x_title=True, hide_y_title=True)

    fig_nlp = px.bar(df_nlp, x='Frecuencia', y='Problema', orientation='h', title="Problemas Técnicos Críticos")
    fig_nlp.update_traces(marker_color="#E74C3C")
    fig_nlp = apply_corporate_layout(fig_nlp, margin=dict(l=150), hide_x_title=True, hide_y_title=True)

    fig_yoy = go.Figure()
    if not df_yoy.empty:
        fig_yoy.add_trace(go.Bar(x=df_yoy['year'], y=df_yoy['volumen'], name='Volumen', marker_color="#EAEAEA", yaxis='y1'))
        fig_yoy.add_trace(go.Scatter(x=df_yoy['year'], y=df_yoy['avg_sentiment'], name='Sentimiento', mode='lines+markers', marker_color=COLOR_PRIMARY, yaxis='y2'))
    fig_yoy.update_layout(title="Tendencia Interanual: Volumen vs Reputación", yaxis=dict(title="Volumen", side="left"), yaxis2=dict(title="Score", side="right", overlaying="y", showgrid=False))
    fig_yoy = apply_corporate_layout(fig_yoy, hide_x_title=True)

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_dev), md=6),
            dbc.Col(dcc.Graph(figure=fig_nlp), md=6),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_yoy), md=12)])
    ])


# =============================================================================
# 4. ETL Y FILTROS
# =============================================================================

@app.callback(
    Output("filter-source", "options"), Output("filter-company", "options"),
    Output("filter-product", "options"), Output("filter-action", "options"),
    Input("processing-status", "data"),
)
def populate(proc):
    if not os.path.exists(pipeline_etl.target_db_path) or proc.get("status") == "processing": return [], [], [], []
    try:
        s = db_manager._execute_query("SELECT DISTINCT source FROM client_signals ORDER BY source")["source"].tolist()
        c = db_manager._execute_query("SELECT DISTINCT company FROM client_signals ORDER BY company")["company"].tolist()
        p = db_manager._execute_query("SELECT DISTINCT product_service FROM client_signals ORDER BY product_service")["product_service"].tolist()
        a = db_manager._execute_query("SELECT DISTINCT customer_action FROM client_signals ORDER BY customer_action")["customer_action"].tolist()
        return s, c, p, a
    except: return dash.no_update

def _bg_process():
    try:
        pipeline_etl.process_zip_file()
        db_manager.clear_cache()
        with open("etl_status.txt", "w") as f: f.write("success")
    except Exception as e:
        with open("etl_status.txt", "w") as f: f.write(f"error: {str(e)}")

@app.callback(
    Output("processing-status", "data"), Output("upload-status-message", "children"),
    Input("upload-data-file", "contents"), State("upload-data-file", "filename"),
    prevent_initial_call=True
)
def upload(contents, name):
    if not contents or not name.lower().endswith(".zip"): return dash.no_update, "Error: Solo ZIP"
    try:
        _, s = contents.split(',')
        pipeline_etl.cleanup_staging()
        with open(pipeline_etl.UPLOADED_ZIP_PATH, "wb") as f: f.write(base64.b64decode(s))
        threading.Thread(target=_bg_process).start()
        return {"status": "processing"}, "Ingestando datos corporativos..."
    except Exception as e: return {"status": "error"}, f"Fallo: {str(e)}"

@app.callback(
    Output("processing-status", "data", allow_duplicate=True),
    Output("upload-status-message", "children", allow_duplicate=True),
    Input("status-interval", "n_intervals"), State("processing-status", "data"),
    prevent_initial_call=True
)
def check(n, curr):
    if curr.get("status") != "processing": return dash.no_update, dash.no_update
    if os.path.exists("etl_status.txt"):
        with open("etl_status.txt", "r") as f: res = f.read()
        os.remove("etl_status.txt")
        if res == "success": return {"status": "ready"}, "Listo"
        else: return {"status": "error"}, f"Fallo: {res}"
    return dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run(debug=False, port=8050)
