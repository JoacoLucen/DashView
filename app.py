import os
import sqlite3
import base64
import threading
import zipfile
import shutil
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.database_manager import _execute_query, clear_cache
from src.metrics import (
    get_nps_proxy, get_complaint_velocity,
    get_sentiment_by_channel, get_monthly_activity_peaks, get_source_impact,
    get_general_direction_kpis, get_regulatory_exposure, get_prechurn_signals_trend,
    get_competitive_benchmark, get_company_product_heatmap,
    get_escalation_rate, get_average_behavior_cycle,
    get_product_risk_radar, get_complaint_topics, get_state_intensity_map,
    get_device_usage_comparison, get_rating_distribution,
    get_app_reviews_nlp, get_yoy_volume_and_sentiment,
)
from src.etl_pipeline import (
    TARGET_DB_PATH, STAGING_DIR, EXTRACTED_DIR, UPLOADED_ZIP_PATH,
    cleanup_staging, process_zip_file,
    get_datasets, delete_dataset, delete_all_datasets,
)

# =============================================================================
# DESIGN SYSTEM
# =============================================================================
COLOR_PRIMARY      = "#1F4E78"
COLOR_ACCENT       = "#2E75B6"
COLOR_BACKGROUND   = "#FFFFFF"
COLOR_PAGE_BG      = "#F4F7FB"
COLOR_NEUTRAL_1    = "#F8FAFD"
COLOR_NEUTRAL_2    = "#A6A6A6"
COLOR_NEUTRAL_DARK = "#595959"
COLOR_SUCCESS      = "#70AD47"
COLOR_WARNING      = "#FFC000"
COLOR_DANGER       = "#C55A11"
COLOR_BORDER       = "#E8EEF4"

FONT_FAMILY = "'DM Sans', 'Segoe UI', system-ui, sans-serif"
FONT_MONO   = "'DM Mono', 'Cascadia Code', 'Courier New', monospace"

_OVERLAY_VISIBLE = {
    "position": "fixed", "top": 0, "left": 0,
    "width": "100vw", "height": "100vh",
    "backgroundColor": "rgba(244, 247, 251, 0.98)",
    "zIndex": 9999, "display": "flex",
    "alignItems": "center", "justifyContent": "center",
}


def get_data_with_fallback(metric_func, filters, title_base):
    """Fetch data with filters. If empty, return an empty/zero result without falling back to global totals."""
    data = metric_func(filters)
    empty = False

    if isinstance(data, pd.DataFrame):
        empty = data.empty
    elif isinstance(data, dict):
        if "total_signals" in data:
            empty = data["total_signals"] == 0
        elif "distribucion" in data:
            empty = data["distribucion"].empty
        else:
            empty = not bool(data)
    elif isinstance(data, (int, float)):
        empty = data == 0

    title = title_base
    if empty:
        title = f"{title_base} (Sin datos para la selección)"

    return data, title


_KPI_BASE = {
    "backgroundColor": COLOR_BACKGROUND,
    "borderRadius": "14px",
    "padding": "22px 24px",
    "boxShadow": "0 1px 4px rgba(31,78,120,0.06), 0 4px 16px rgba(31,78,120,0.07)",
    "border": f"1px solid {COLOR_BORDER}",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
}
STYLE_KPI         = {**_KPI_BASE, "borderTop": f"3px solid {COLOR_PRIMARY}"}
STYLE_KPI_DANGER  = {**_KPI_BASE, "borderTop": f"3px solid {COLOR_DANGER}"}
STYLE_KPI_SUCCESS = {**_KPI_BASE, "borderTop": f"3px solid {COLOR_SUCCESS}"}
STYLE_KPI_WARNING = {**_KPI_BASE, "borderTop": f"3px solid {COLOR_WARNING}"}

STYLE_KPI_VALUE = {
    "color": COLOR_PRIMARY,
    "fontWeight": "500",
    "fontSize": "2.6rem",
    "lineHeight": "1",
    "letterSpacing": "-0.03em",
    "marginBottom": "6px",
    "fontFamily": FONT_MONO,
}
STYLE_KPI_LABEL = {
    "fontSize": "0.67rem",
    "fontWeight": "700",
    "color": COLOR_NEUTRAL_2,
    "letterSpacing": "0.1em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
}
STYLE_FILTER_LABEL = {
    "fontSize": "0.67rem",
    "fontWeight": "700",
    "letterSpacing": "0.08em",
    "color": COLOR_NEUTRAL_2,
    "textTransform": "uppercase",
    "marginBottom": "4px",
    "display": "block",
}


def kpi_card(value: str, label: str, style: dict = None, subtitle: str = None) -> html.Div:
    children = [
        html.P(label, style=STYLE_KPI_LABEL),
        html.Div(value, style=STYLE_KPI_VALUE),
    ]
    if subtitle:
        children.append(html.P(subtitle, style={"fontSize": "0.72rem", "color": COLOR_NEUTRAL_2, "marginBottom": "0"}))
    return html.Div(children, style=style or STYLE_KPI, className="kpi-hover")


def apply_corporate_layout(fig, barmode=None, margin=None, hide_x_title=False, hide_y_title=False):
    default_margin = dict(l=120, r=40, t=52, b=72)
    if margin:
        default_margin.update(margin)
    fig.update_layout(
        font=dict(family=FONT_FAMILY, color=COLOR_NEUTRAL_DARK, size=12),
        plot_bgcolor=COLOR_BACKGROUND,
        paper_bgcolor=COLOR_BACKGROUND,
        margin=default_margin,
        barmode=barmode,
        colorway=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, "#4A90E2", "#7B61C0"],
        title_font=dict(size=13, color=COLOR_PRIMARY, family=FONT_FAMILY, weight=600),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, family=FONT_FAMILY), bgcolor="rgba(255,255,255,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#EEF2F7", gridwidth=1,
        showline=False,
        title_text="" if hide_x_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11, color=COLOR_NEUTRAL_DARK),
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EEF2F7", gridwidth=1,
        showline=False,
        title_text="" if hide_y_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11, color=COLOR_NEUTRAL_DARK),
        zeroline=False,
    )
    return fig


def _G(fig, cfg=None):
    """Wrap a Plotly figure in a styled chart-card div."""
    return html.Div(
        dcc.Graph(figure=fig, config=cfg or {"displayModeBar": False}),
        className="chart-card",
    )


def _build_datasets_checklist(datasets: list):
    """Render a checklist of loaded datasets for the management modal."""
    if not datasets:
        return html.P(
            "No hay datasets cargados.",
            style={
                "color": COLOR_NEUTRAL_2, "textAlign": "center",
                "padding": "28px 0", "fontSize": "0.88rem",
            },
        )
    return dcc.Checklist(
        id="dataset-checklist",
        options=[
            {
                "label": f"  {d['name']}   ·   {d['row_count']:,} filas   ·   {d['loaded_at'][:10]}",
                "value": str(d["id"]),
            }
            for d in datasets
        ],
        value=[],
        labelStyle={
            "display": "flex",
            "alignItems": "center",
            "padding": "11px 4px",
            "borderBottom": f"1px solid {COLOR_BORDER}",
            "fontSize": "0.875rem",
            "cursor": "pointer",
        },
        inputStyle={"marginRight": "10px", "accentColor": COLOR_PRIMARY, "cursor": "pointer"},
    )


# =============================================================================
# APP INITIALIZATION
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css",
        "https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&family=DM+Mono:wght@400;500&display=swap",
    ],
    title="DashView Analytics",
    suppress_callback_exceptions=True,
)

for path in [STAGING_DIR, EXTRACTED_DIR]:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = html.Div(
    style={"backgroundColor": COLOR_PAGE_BG, "fontFamily": FONT_FAMILY, "minHeight": "100vh"},
    children=[
        dcc.Store(id="processing-status", data={"status": "ready"}),
        dcc.Store(id="dataset-refresh", data=0),
        dcc.Interval(id="status-interval", interval=1000, n_intervals=0),

        # ── DATASET MANAGEMENT MODAL ───────────────────────────────────────
        dbc.Modal(
            id="datasets-modal",
            is_open=False,
            size="lg",
            backdrop="static",
            children=[
                dbc.ModalHeader(
                    dbc.ModalTitle([
                        html.I(className="bi bi-layers-fill me-2", style={"color": COLOR_ACCENT}),
                        "Gestión de Datasets",
                    ]),
                    close_button=True,
                ),
                dbc.ModalBody(
                    html.Div(id="datasets-modal-body"),
                    style={"padding": "20px 24px", "maxHeight": "420px", "overflowY": "auto"},
                ),
                dbc.ModalFooter(
                    style={"gap": "8px"},
                    children=[
                        dbc.Button(
                            [html.I(className="bi bi-trash me-1"), "Eliminar seleccionados"],
                            id="btn-delete-selected",
                            color="danger", outline=True, size="sm",
                            style={"marginRight": "auto"},
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-trash-fill me-1"), "Eliminar todos"],
                            id="btn-delete-all-datasets",
                            color="danger", size="sm",
                        ),
                        dbc.Button(
                            "Cerrar",
                            id="btn-close-datasets-modal",
                            color="secondary", outline=True, size="sm",
                        ),
                    ],
                ),
            ],
        ),

        # ── MAIN DASHBOARD ────────────────────────────────────────────────
        html.Div(
            id="main-dashboard-container",
            style={"display": "block" if os.path.exists(TARGET_DB_PATH) else "none"},
            children=[
                dbc.NavbarSimple(
                    brand=html.Span([
                        html.Span("DV", style={
                            "display": "inline-flex", "alignItems": "center",
                            "justifyContent": "center", "width": "30px", "height": "30px",
                            "backgroundColor": "rgba(255,255,255,0.16)", "borderRadius": "7px",
                            "fontWeight": "800", "fontSize": "13px", "letterSpacing": "-0.01em",
                            "marginRight": "10px", "verticalAlign": "middle",
                            "fontFamily": FONT_MONO,
                        }),
                        html.Span("Dash", style={"fontWeight": "300", "letterSpacing": "-0.01em"}),
                        html.Span("View", style={"fontWeight": "800", "letterSpacing": "-0.03em"}),
                        html.Span(" Analytics", style={
                            "fontWeight": "300", "opacity": "0.65",
                            "fontSize": "0.72em", "marginLeft": "7px", "letterSpacing": "0.04em",
                        }),
                    ]),
                    brand_style={"fontSize": "26px", "fontFamily": FONT_FAMILY},
                    children=[
                        dbc.Button(
                            [
                                html.I(className="bi bi-layers me-2"),
                                html.Span(id="datasets-badge-text", children="Gestión de Dataset's"),
                            ],
                            id="btn-manage-datasets",
                            color="light", size="sm", className="ms-2",
                            style={
                                "fontWeight": "600", "fontSize": "0.78rem",
                                "letterSpacing": "0.05em", "borderRadius": "8px",
                            },
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-cloud-upload me-2"), "Importar Datos"],
                            id="btn-load-new", color="light", size="sm", className="ms-2",
                            style={
                                "fontWeight": "600", "fontSize": "0.78rem",
                                "letterSpacing": "0.05em", "borderRadius": "8px",
                            },
                        ),
                    ],
                    color=COLOR_PRIMARY, dark=True, fluid=True, className="mb-0 shadow-sm",
                    style={"background": f"linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%)"},
                ),

                dbc.Container(fluid=True, style={"paddingTop": "0"}, children=[
                    html.Div(id="global-alert-container"),

                    # ── STICKY FILTER PANEL ────────────────────────────────
                    html.Div(
                        style={"position": "sticky", "top": "0px", "zIndex": 1000},
                        children=[
                            dbc.Card(
                                className="mb-0 border-0",
                                style={
                                    "borderRadius": "0 0 16px 16px",
                                    "boxShadow": "0 8px 32px rgba(31,78,120,0.10)",
                                    "borderTop": f"3px solid {COLOR_PRIMARY}",
                                    "backgroundColor": "rgba(255,255,255,0.97)",
                                    "backdropFilter": "blur(12px)",
                                    "WebkitBackdropFilter": "blur(12px)",
                                },
                                children=dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-calendar3 me-1"), "Período"], style=STYLE_FILTER_LABEL),
                                            dcc.RangeSlider(
                                                id="filter-period", min=2010, max=2027, step=1,
                                                value=[2010, 2027],
                                                marks={str(y): {"label": str(y), "style": {"fontSize": "10px"}} for y in range(2010, 2028, 2)},
                                            ),
                                        ]),
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-broadcast me-1"), "Plataforma"], style=STYLE_FILTER_LABEL),
                                            dcc.Dropdown(id="filter-source", multi=True, placeholder="Todas", style={"fontSize": "13px"}),
                                        ]),
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-building me-1"), "Empresa"], style=STYLE_FILTER_LABEL),
                                            dcc.Dropdown(id="filter-company", multi=True, placeholder="Todas", style={"fontSize": "13px"}),
                                        ]),
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-box-seam me-1"), "Producto"], style=STYLE_FILTER_LABEL),
                                            dcc.Dropdown(id="filter-product", multi=True, placeholder="Todos", style={"fontSize": "13px"}),
                                        ]),
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-cursor me-1"), "Acción"], style=STYLE_FILTER_LABEL),
                                            dcc.Dropdown(id="filter-action", multi=True, placeholder="Todas", style={"fontSize": "13px"}),
                                        ]),
                                        dbc.Col(md=2, children=[
                                            html.Label([html.I(className="bi bi-emoji-smile me-1"), "Sentimiento"], style=STYLE_FILTER_LABEL),
                                            dcc.RadioItems(
                                                id="filter-sentiment", inline=True, className="mt-1",
                                                options=[
                                                    {"label": " Todos",    "value": "ALL"},
                                                    {"label": " Positivo", "value": "Positive"},
                                                    {"label": " Negativo", "value": "Negative"},
                                                ],
                                                value="ALL",
                                                inputStyle={"marginLeft": "10px"},
                                                style={"fontSize": "13px"},
                                            ),
                                        ]),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(md=12, className="text-end", children=[
                                            dbc.Button(
                                                [html.I(className="bi bi-x-circle me-1"), "Limpiar Filtros"],
                                                id="btn-clear-filters", color="link", size="sm",
                                                style={
                                                    "color": COLOR_NEUTRAL_2, "fontSize": "0.72rem",
                                                    "fontWeight": "600", "letterSpacing": "0.05em",
                                                    "textDecoration": "none",
                                                },
                                            )
                                        ])
                                    ]),
                                ], style={"padding": "16px 28px 12px"}),
                            ),
                        ],
                    ),

                    dbc.Tabs(
                        id="tabs-stakeholders",
                        active_tab="tab-marketing",
                        children=[
                            dbc.Tab(label="Marketing",            tab_id="tab-marketing"),
                            dbc.Tab(label="Dirección General",    tab_id="tab-dir-general"),
                            dbc.Tab(label="Retención y Facturación", tab_id="tab-retencion"),
                            dbc.Tab(label="Equipo de Producto",   tab_id="tab-producto"),
                        ],
                    ),

                    dcc.Loading(
                        id="loading-dashboard", type="dot", color=COLOR_PRIMARY,
                        children=html.Div(id="tab-content-container", className="mt-4 pb-5"),
                    ),
                ]),
            ],
        ),

        # ── WELCOME OVERLAY ────────────────────────────────────────────────
        html.Div(
            id="welcome-overlay",
            style=_OVERLAY_VISIBLE if not os.path.exists(TARGET_DB_PATH) else {"display": "none"},
            children=[
                dbc.Card(
                    style={"width": "480px", "borderRadius": "20px", "border": "none", "overflow": "hidden"},
                    className="shadow-lg",
                    children=[
                        html.Div(
                            className="overlay-header",
                            style={
                                "background": f"linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%)",
                                "padding": "36px 32px",
                                "textAlign": "center",
                            },
                            children=[
                                html.Div(
                                    "DV",
                                    style={
                                        "display": "inline-flex", "alignItems": "center",
                                        "justifyContent": "center", "width": "44px", "height": "44px",
                                        "backgroundColor": "rgba(255,255,255,0.15)",
                                        "borderRadius": "10px", "fontWeight": "800",
                                        "fontSize": "16px", "color": "#FFFFFF",
                                        "marginBottom": "14px", "fontFamily": FONT_MONO,
                                        "letterSpacing": "-0.01em",
                                    },
                                ),
                                html.Div([
                                    html.Span("Dash", style={
                                        "fontWeight": "300", "color": "rgba(255,255,255,0.85)",
                                        "fontSize": "2.8rem", "letterSpacing": "-0.02em",
                                    }),
                                    html.Span("View", style={
                                        "fontWeight": "800", "color": "#FFFFFF",
                                        "fontSize": "2.8rem", "letterSpacing": "-0.04em",
                                    }),
                                ]),
                                html.P(
                                    "Inteligencia de Datos de Clientes",
                                    style={"color": "rgba(255,255,255,0.65)", "marginBottom": "0",
                                           "fontSize": "0.85rem", "letterSpacing": "0.08em",
                                           "textTransform": "uppercase"},
                                ),
                            ],
                        ),
                        dbc.CardBody([
                            dcc.Loading(type="circle", color=COLOR_PRIMARY, children=[
                                dcc.Upload(
                                    id="upload-data-file", accept=".zip",
                                    className="upload-zone",
                                    style={
                                        "width": "100%", "height": "110px", "lineHeight": "110px",
                                        "borderWidth": "2px", "borderStyle": "dashed",
                                        "borderColor": COLOR_BORDER, "borderRadius": "12px",
                                        "backgroundColor": COLOR_NEUTRAL_1, "cursor": "pointer",
                                        "textAlign": "center", "transition": "all 0.2s ease",
                                        "color": COLOR_NEUTRAL_DARK, "fontSize": "14px",
                                    },
                                    children=html.Div([
                                        html.I(className="bi bi-cloud-upload me-2", style={"fontSize": "1.4rem", "color": COLOR_ACCENT, "verticalAlign": "middle"}),
                                        "Arrastrá o clickeá para cargar un ", html.B(".ZIP"),
                                    ]),
                                ),
                                html.Hr(style={"margin": "18px 0", "borderColor": COLOR_BORDER}),
                                html.P("O cargá desde una ruta local (para archivos grandes):",
                                       style={"fontSize": "12px", "color": COLOR_NEUTRAL_DARK, "marginBottom": "8px"}),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="local-path-input",
                                        placeholder=r"Ej: C:\Users\lauta\Downloads\zip.zip",
                                        type="text", size="sm",
                                        style={"fontSize": "12px"},
                                    ),
                                    dbc.Button("Cargar", id="btn-load-local", color="primary", size="sm"),
                                ], style={"marginBottom": "8px"}),
                                html.Div(
                                    id="upload-status-message",
                                    className="mt-3 fw-semibold text-center",
                                    style={"color": COLOR_PRIMARY, "fontSize": "0.9rem"},
                                ),
                            ]),
                            # Existing datasets summary (shown when re-importing)
                            html.Div(id="overlay-datasets-section"),
                        ], style={"padding": "28px"}),
                    ],
                )
            ],
        ),
    ],
)

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output("filter-period", "value"),
    Output("filter-source", "value"),
    Output("filter-company", "value"),
    Output("filter-product", "value"),
    Output("filter-action", "value"),
    Output("filter-sentiment", "value"),
    Input("btn-clear-filters", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(n):
    return [2010, 2025], None, None, None, None, "ALL"


@app.callback(
    Output("welcome-overlay", "style", allow_duplicate=True),
    Output("main-dashboard-container", "style", allow_duplicate=True),
    Input("btn-load-new", "n_clicks"),
    prevent_initial_call=True,
)
def show_import_overlay(n):
    if n:
        return _OVERLAY_VISIBLE, {"display": "none"}
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
    prevent_initial_call='initial_duplicate',
)
def update_view(tab, period, sources, companies, products, actions, sentiment, proc_status):
    if not os.path.exists(TARGET_DB_PATH):
        return html.Div(), dash.no_update, {"display": "none"}
    if proc_status.get("status") == "processing":
        return dash.no_update, dash.no_update, dash.no_update

    filters = {
        "period": period, "sources": sources, "companies": companies,
        "products": products, "actions": actions, "sentiment": sentiment,
    }

    try:
        if   tab == "tab-marketing":    view = render_marketing(filters)
        elif tab == "tab-dir-general":  view = render_general_direction(filters)
        elif tab == "tab-retencion":    view = render_retention(filters)
        elif tab == "tab-producto":     view = render_product_team(filters)
        else:                           view = html.Div()
        return view, {"display": "none"}, {"display": "block"}
    except Exception as e:
        return (
            html.Div(dbc.Alert(f"Aviso del Sistema: {e}", color="warning")),
            {"display": "none"},
            {"display": "block"},
        )


@app.callback(
    Output("filter-source", "options"),
    Output("filter-company", "options"),
    Output("filter-product", "options"),
    Output("filter-action", "options"),
    Input("processing-status", "data"),
)
def populate_filters(proc):
    if not os.path.exists(TARGET_DB_PATH) or proc.get("status") == "processing":
        return [], [], [], []
    try:
        with sqlite3.connect(TARGET_DB_PATH) as conn:
            s = pd.read_sql_query("SELECT DISTINCT source FROM client_signals WHERE source IS NOT NULL ORDER BY source", conn)["source"].tolist()
            c = pd.read_sql_query("SELECT DISTINCT company FROM client_signals WHERE company IS NOT NULL ORDER BY company", conn)["company"].tolist()
            p = pd.read_sql_query("SELECT DISTINCT product_service FROM client_signals WHERE product_service IS NOT NULL ORDER BY product_service", conn)["product_service"].tolist()
            a = pd.read_sql_query("SELECT DISTINCT customer_action FROM client_signals WHERE customer_action IS NOT NULL ORDER BY customer_action", conn)["customer_action"].tolist()
            return s, c, p, a
    except Exception as e:
        print(f"[Populate Filters Error] {e}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# ── Dataset modal ──────────────────────────────────────────────────────────────

@app.callback(
    Output("datasets-modal", "is_open"),
    Input("btn-manage-datasets", "n_clicks"),
    Input("btn-close-datasets-modal", "n_clicks"),
    State("datasets-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_datasets_modal(n_open, n_close, is_open):
    return not is_open


@app.callback(
    Output("datasets-modal-body", "children"),
    Input("datasets-modal", "is_open"),
    Input("dataset-refresh", "data"),
)
def render_datasets_modal_body(is_open, refresh):
    return _build_datasets_checklist(get_datasets())


@app.callback(
    Output("datasets-badge-text", "children"),
    Input("dataset-refresh", "data"),
    Input("processing-status", "data"),
)
def update_datasets_badge(refresh, proc_status):
    count = len(get_datasets())
    return f"Gestión de Dataset's ({count})" if count else "Gestión de Dataset's"


@app.callback(
    Output("overlay-datasets-section", "children"),
    Input("dataset-refresh", "data"),
    Input("processing-status", "data"),
)
def render_overlay_datasets(refresh, proc_status):
    datasets = get_datasets()
    if not datasets:
        return []
    count = len(datasets)
    total_rows = sum(d["row_count"] for d in datasets)
    return html.Div([
        html.Hr(style={"borderColor": COLOR_BORDER, "margin": "16px 0 12px"}),
        html.Div([
            html.I(className="bi bi-layers-fill me-2", style={"color": COLOR_ACCENT}),
            html.Span(
                f"{count} dataset{'s' if count > 1 else ''} ya cargado{'s' if count > 1 else ''}",
                style={"fontWeight": "600", "fontSize": "0.83rem", "color": COLOR_PRIMARY},
            ),
            html.Span(
                f" · {total_rows:,} filas totales",
                style={"fontSize": "0.78rem", "color": COLOR_NEUTRAL_2},
            ),
        ], style={"textAlign": "center", "marginBottom": "5px"}),
        html.P(
            "El nuevo dataset se acumulará a los existentes sin reemplazarlos.",
            style={"fontSize": "0.75rem", "color": COLOR_NEUTRAL_2, "textAlign": "center", "margin": "0"},
        ),
    ])


@app.callback(
    Output("dataset-refresh", "data"),
    Output("processing-status", "data", allow_duplicate=True),
    Output("welcome-overlay", "style", allow_duplicate=True),
    Output("main-dashboard-container", "style", allow_duplicate=True),
    Input("btn-delete-selected", "n_clicks"),
    Input("btn-delete-all-datasets", "n_clicks"),
    State("dataset-checklist", "value"),
    State("dataset-refresh", "data"),
    prevent_initial_call=True,
)
def handle_dataset_deletion(n_sel, n_all, selected_ids, refresh_count):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "btn-delete-all-datasets":
        delete_all_datasets()
    elif triggered_id == "btn-delete-selected" and selected_ids:
        for did in selected_ids:
            delete_dataset(int(did))
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    clear_cache()
    remaining = get_datasets()
    new_count = (refresh_count or 0) + 1

    if not remaining:
        if os.path.exists(TARGET_DB_PATH):
            os.remove(TARGET_DB_PATH)
        return new_count, {"status": "ready"}, _OVERLAY_VISIBLE, {"display": "none"}

    return new_count, {"status": "ready"}, dash.no_update, dash.no_update


# =============================================================================
# ETL BACKGROUND
# =============================================================================

def _bg_process(dataset_name: str):
    try:
        process_zip_file(dataset_name)
        clear_cache()
        with open("etl_status.txt", "w", encoding="utf-8") as f:
            f.write("success")
    except Exception as e:
        with open("etl_status.txt", "w", encoding="utf-8") as f:
            f.write(f"error: {e}")


@app.callback(
    Output("processing-status", "data"),
    Output("upload-status-message", "children"),
    Input("upload-data-file", "contents"),
    State("upload-data-file", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, name):
    if not contents or not name.lower().endswith(".zip"):
        return dash.no_update, dbc.Alert("Solo se aceptan archivos .ZIP", color="danger")
    try:
        _, encoded = contents.split(',')
        cleanup_staging()
        with open(UPLOADED_ZIP_PATH, "wb") as f:
            f.write(base64.b64decode(encoded))
        dataset_name = os.path.splitext(name)[0] if name else "Dataset"
        threading.Thread(target=_bg_process, args=(dataset_name,)).start()
        return {"status": "processing"}, html.Span([
            dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
            "Ingestando datos corporativos…",
        ])
    except Exception as e:
        return {"status": "error"}, dbc.Alert(f"Fallo: {e}", color="danger")


@app.callback(
    Output("processing-status", "data", allow_duplicate=True),
    Output("upload-status-message", "children", allow_duplicate=True),
    Input("btn-load-local", "n_clicks"),
    State("local-path-input", "value"),
    prevent_initial_call=True,
)
def handle_local_path(n_clicks, path):
    if not path or not path.strip():
        return dash.no_update, dbc.Alert("Ingresá una ruta válida.", color="warning")
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        return dash.no_update, dbc.Alert(f"Archivo no encontrado: {path}", color="danger")
    if not zipfile.is_zipfile(path):
        return dash.no_update, dbc.Alert("El archivo no es un ZIP válido.", color="danger")
    try:
        cleanup_staging()
        shutil.copy2(path, UPLOADED_ZIP_PATH)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        threading.Thread(target=_bg_process, args=(dataset_name,)).start()
        return {"status": "processing"}, html.Span([
            dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
            "Ingestando datos corporativos…",
        ])
    except Exception as e:
        return {"status": "error"}, dbc.Alert(f"Fallo: {e}", color="danger")


@app.callback(
    Output("processing-status", "data", allow_duplicate=True),
    Output("upload-status-message", "children", allow_duplicate=True),
    Input("status-interval", "n_intervals"),
    State("processing-status", "data"),
    prevent_initial_call=True,
)
def check_processing(n, curr):
    if curr.get("status") != "processing":
        return dash.no_update, dash.no_update
    if os.path.exists("etl_status.txt"):
        with open("etl_status.txt", "r", encoding="utf-8") as f:
            res = f.read()
        os.remove("etl_status.txt")
        if res == "success":
            return {"status": "ready"}, html.Span([
                html.I(className="bi bi-check-circle-fill me-2", style={"color": COLOR_SUCCESS}),
                "¡Datos cargados exitosamente!",
            ])
        return {"status": "error"}, dbc.Alert(f"Fallo: {res}", color="danger")
    return dash.no_update, dash.no_update


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_marketing(filters: dict) -> html.Div:
    nps, nps_title = get_data_with_fallback(get_nps_proxy, filters, "Promotores · Pasivos · Detractores")
    df_vel, vel_title = get_data_with_fallback(get_complaint_velocity, filters, "Tendencia Trimestral de Quejas y Deserciones")
    df_sent, sent_title = get_data_with_fallback(get_sentiment_by_channel, filters, "Índice de Satisfacción por Canal")
    df_peaks, peaks_title = get_data_with_fallback(get_monthly_activity_peaks, filters, "Volumen de Señales por Mes")
    df_impact, impact_title = get_data_with_fallback(get_source_impact, filters, "Satisfacción vs Insatisfacción por Plataforma (%)")

    # NPS breakdown stacked bar
    fig_nps = go.Figure()
    total_sig = nps["total_signals"] or 1
    if not nps["breakdown_df"].empty:
        seg_colors = {
            "Promotores": COLOR_SUCCESS,
            "Pasivos": COLOR_NEUTRAL_2,
            "Detractores": COLOR_DANGER,
        }
        for _, row in nps["breakdown_df"].iterrows():
            pct = round(row["Cantidad"] / total_sig * 100, 1)
            fig_nps.add_trace(go.Bar(
                x=[pct], y=["Clientes"],
                name=row["Segmento"],
                orientation="h",
                marker_color=seg_colors.get(row["Segmento"], COLOR_ACCENT),
                text=[f"{pct:.0f}%"] if pct >= 5 else [None],
                textposition="inside",
                textfont=dict(color="white", size=12, family=FONT_FAMILY),
                hovertemplate=f"<b>{row['Segmento']}</b><br>Clientes: {row['Cantidad']:,}<br>Proporción: {pct}%<extra></extra>",
            ))
    fig_nps.update_layout(
        title=nps_title,
        barmode="stack",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
    )
    fig_nps = apply_corporate_layout(fig_nps, barmode="stack", hide_x_title=True, hide_y_title=True)

    # Complaint velocity trend
    fig_vel = go.Figure()
    if not df_vel.empty:
        fig_vel.add_trace(go.Scatter(
            x=df_vel["periodo"], y=df_vel["quejas"],
            mode="lines+markers",
            line=dict(color=COLOR_DANGER, width=2.5),
            marker=dict(size=6, color=COLOR_DANGER),
            fill="tozeroy", fillcolor="rgba(197,90,17,0.08)",
            hovertemplate="<b>%{x}</b><br>Quejas + Deserciones: %{y:,}<extra></extra>",
        ))
    fig_vel.update_layout(title=vel_title)
    fig_vel = apply_corporate_layout(fig_vel, hide_x_title=True, hide_y_title=True)
    fig_vel.update_xaxes(tickangle=-45, nticks=12)

    # Sentiment by channel with conditional coloring
    if not df_sent.empty:
        bar_colors = [
            COLOR_DANGER if v < 0 else (COLOR_WARNING if v < 0.1 else COLOR_SUCCESS)
            for v in df_sent["avg_sentiment"]
        ]
        fig_sent = go.Figure(go.Bar(
            x=df_sent["avg_sentiment"], y=df_sent["source"],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>Índice: %{x:.3f}<extra></extra>",
        ))
        fig_sent.add_vline(x=0, line_dash="dash", line_color=COLOR_NEUTRAL_2, line_width=1)
        fig_sent.update_layout(title=sent_title, xaxis_title="Índice de Satisfacción")
    else:
        fig_sent = go.Figure().update_layout(title=sent_title)
    fig_sent = apply_corporate_layout(fig_sent, margin=dict(l=150), hide_y_title=True)

    fig_peaks = px.bar(
        df_peaks, x="mes_label", y="volumen",
        title=peaks_title,
        labels={"mes_label": "Mes", "volumen": "Señales"},
    )
    fig_peaks.update_traces(
        marker_color=COLOR_ACCENT,
        hovertemplate="<b>%{x}</b><br>Señales: %{y:,}<extra></extra>",
    )
    fig_peaks = apply_corporate_layout(fig_peaks, hide_x_title=True, hide_y_title=True)

    fig_impact = px.bar(
        df_impact, x="source", y=["pct_positive", "pct_negative"],
        title=impact_title,
        labels={
            "source": "Plataforma", "value": "Porcentaje",
            "pct_positive": "% Satisfechos", "pct_negative": "% Insatisfechos",
        },
        barmode="group",
    )
    fig_impact.update_traces(
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>",
    )
    fig_impact = apply_corporate_layout(fig_impact, barmode="group", hide_x_title=True, hide_y_title=True)

    nps_score = nps["nps_score"]
    nps_style = STYLE_KPI_SUCCESS if nps_score >= 0 else STYLE_KPI_DANGER

    warnings = []
    if filters.get("sentiment") and filters["sentiment"] != "ALL":
        warnings.append(dbc.Alert(
            "⚠ El filtro de Sentimiento afecta el cálculo del NPS. Para ver el NPS real, usar 'Todos'.",
            color="warning", className="mb-3 py-2 small"
        ))
    
    action_note = ""
    if filters.get("actions"):
        action_note = "Esta métrica no cambia con el filtro de Acción del Cliente. Se basa en la etiqueta de sentimiento."

    return html.Div([
        html.Div(warnings),
        dbc.Row([
            dbc.Col(kpi_card(
                f"{nps['total_signals']:,}", "Total Señales de Clientes",
                subtitle="Interacciones registradas en todos los canales digitales",
            ), md=3),
            dbc.Col(html.Div(id="kpi-nps-proxy", children=kpi_card(
                f"{nps_score:+.0f}", "NPS Proxy",
                nps_style,
                subtitle=action_note if action_note else "Promotores menos Detractores — rango de −100 a +100",
            )), md=3),
            dbc.Col(kpi_card(
                f"{nps['pct_promoters']}%", "Promotores",
                STYLE_KPI_SUCCESS,
                subtitle="Clientes que defienden o recomiendan activamente la marca",
            ), md=3),
            dbc.Col(kpi_card(
                f"{nps['pct_detractors']}%", "Detractores",
                STYLE_KPI_DANGER,
                subtitle="Clientes con comportamiento de queja, deserción o malestar",
            ), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(_G(fig_vel), md=8),
            dbc.Col(_G(fig_nps), md=4),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(html.Div([
                _G(fig_sent),
                html.P(
                    "El Índice de Satisfacción se basa en sentiment_score. El filtro de Acción reduce el universo de registros pero no cambia la definición de la métrica.",
                    style={"fontSize": "0.7rem", "color": COLOR_NEUTRAL_2, "marginTop": "4px"}
                )
            ]), md=6),
            dbc.Col(_G(fig_impact), md=6),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(_G(fig_peaks), md=12),
        ], className="g-3"),
    ])


def render_general_direction(filters: dict) -> html.Div:
    kpis, churn_title = get_data_with_fallback(get_general_direction_kpis, filters, "¿Por qué se van los clientes?")
    reg, _            = get_data_with_fallback(get_regulatory_exposure, filters, "Exposición Regulatoria")
    df_pre, pre_title = get_data_with_fallback(get_prechurn_signals_trend, filters, "Señales de Alerta Temprana por Año")
    df_bench, bench_title = get_data_with_fallback(get_competitive_benchmark, filters, "Empresas con Menor Satisfacción de Clientes")
    df_heat, heat_title   = get_data_with_fallback(get_company_product_heatmap, filters, "¿Qué empresa-producto tiene clientes más insatisfechos?")

    dist_df = kpis["distribucion"]
    if not dist_df.empty and "causa_label" in dist_df.columns:
        fig_churn = px.pie(
            dist_df, values="cantidad", names="causa_label", hole=0.5,
            title=churn_title,
            labels={"causa_label": "Motivo de Salida", "cantidad": "Clientes Afectados"},
        )
        fig_churn.update_traces(
            hovertemplate="<b>%{label}</b><br>Clientes: %{value:,}<br>Del total: %{percent}<extra></extra>",
        )
    else:
        fig_churn = go.Figure().update_layout(title=f"{churn_title} — Sin datos suficientes")
    fig_churn = apply_corporate_layout(fig_churn)

    # Pre-churn early warning trend
    fig_pre = go.Figure()
    if not df_pre.empty:
        fig_pre.add_trace(go.Scatter(
            x=df_pre["year"], y=df_pre["prechurn"],
            mode="lines+markers",
            line=dict(color=COLOR_WARNING, width=2.5),
            marker=dict(size=7, color=COLOR_WARNING),
            fill="tozeroy", fillcolor="rgba(255,192,0,0.10)",
            hovertemplate="<b>%{x}</b><br>Señales Pre-Deserción: %{y:,}<extra></extra>",
        ))
    fig_pre.update_layout(title=pre_title)
    fig_pre = apply_corporate_layout(fig_pre, hide_x_title=True, hide_y_title=True)

    if not df_bench.empty:
        df_b = df_bench.copy()
        df_b["sat_score"] = ((df_b["avg_sentiment"] + 1) / 2 * 100).round(1)
        # Sort descending so worst company is at TOP of horizontal bar chart
        df_b = df_b.sort_values("sat_score", ascending=False)
        bar_colors = [
            COLOR_DANGER if v < 40 else (COLOR_WARNING if v < 55 else COLOR_SUCCESS)
            for v in df_b["sat_score"]
        ]
        fig_bench = go.Figure(go.Bar(
            x=df_b["sat_score"], y=df_b["company"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.0f}%" for v in df_b["sat_score"]],
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=11),
            hovertemplate="<b>%{y}</b><br>Satisfacción: %{x:.1f}%<extra></extra>",
        ))
        fig_bench.update_layout(
            title=bench_title,
            xaxis=dict(range=[0, 110], ticksuffix="%"),
        )
    else:
        fig_bench = go.Figure().update_layout(title=f"{bench_title} — Sin datos suficientes")
    fig_bench = apply_corporate_layout(fig_bench, margin=dict(l=160), hide_x_title=True, hide_y_title=True)

    if not df_heat.empty:
        df_hd = df_heat.copy()
        df_hd["sat_score"] = ((df_hd["avg_sentiment"] + 1) / 2 * 100).round(0)
        # Limit to top 8 companies by data coverage for readability
        top_cos = df_hd.groupby("company")["sat_score"].count().nlargest(8).index
        df_hd = df_hd[df_hd["company"].isin(top_cos)]
        pivot = df_hd.pivot_table(
            index="company", columns="product_service", values="sat_score", aggfunc="mean"
        )
        text_matrix = [
            [f"{v:.0f}%" if not pd.isna(v) else "" for v in row]
            for row in pivot.values
        ]
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, family=FONT_FAMILY),
            colorscale=[[0, COLOR_DANGER], [0.4, "#FFB68A"], [0.5, "#F5F5F5"], [0.6, "#AED490"], [1, COLOR_SUCCESS]],
            zmin=0, zmax=100, zmid=50,
            colorbar=dict(title="Satisf. %", tickfont=dict(family=FONT_FAMILY), ticksuffix="%"),
            hovertemplate="<b>%{y}</b><br>Producto: %{x}<br>Satisfacción: %{z:.0f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            title=heat_title,
            font=dict(family=FONT_FAMILY),
            annotations=[dict(
                text="0% = muy insatisfechos · 50% = neutro · 100% = muy satisfechos",
                x=0.5, y=-0.20, xref="paper", yref="paper", showarrow=False,
                font=dict(size=10, color=COLOR_NEUTRAL_2, family=FONT_FAMILY),
            )],
        )
    else:
        fig_heat = go.Figure().update_layout(title=heat_title)
    fig_heat = apply_corporate_layout(fig_heat, margin=dict(l=160, b=130), hide_x_title=True, hide_y_title=True)

    reg_style = STYLE_KPI_DANGER if reg["pct"] > 50 else STYLE_KPI_WARNING
    prechurn_total = int(df_pre["prechurn"].sum()) if not df_pre.empty else 0

    warnings = []
    if filters.get("sentiment") and filters["sentiment"] != "ALL":
        warnings.append(dbc.Alert(
            f"Mostrando datos para registros con sentimiento: {filters['sentiment']}. Esto puede sesgar los indicadores de churn y satisfacción.",
            color="warning", className="mb-3 py-2 small"
        ))

    reg_note = "Esta métrica no cambia con el filtro de Acción o Sentimiento. El porcentaje varía según la empresa seleccionada."

    return html.Div([
        html.Div(warnings),
        dbc.Row([
            dbc.Col(kpi_card(
                f"{kpis['total_churn']:,}", "Señales de Deserción", STYLE_KPI_DANGER,
                subtitle="Clientes que expresaron intención de cancelar o buscar alternativas",
            ), md=4),
            dbc.Col(kpi_card(
                f"{reg['pct']}%", "Exposición Regulatoria (CFPB)", reg_style,
                subtitle=reg_note if filters.get("company") or filters.get("actions") or filters.get("sentiment") != "ALL" else f"{reg['regulatorias']:,} quejas regulatorias de {reg['total']:,} señales totales",
            ), md=4),
            dbc.Col(kpi_card(
                f"{prechurn_total:,}", "Señales Pre-Deserción", STYLE_KPI_WARNING,
                subtitle="Clientes buscando alternativas o reaccionando a cambios de precio o política",
            ), md=4),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(_G(fig_churn), md=5),
            dbc.Col(_G(fig_pre),   md=7),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(html.Div([
                _G(fig_bench),
                html.P("El Índice de Satisfacción se basa en sentiment_score. El filtro de Acción reduce el volumen pero no cambia la definición de la métrica.",
                       style={"fontSize": "0.7rem", "color": COLOR_NEUTRAL_2, "marginTop": "4px"})
            ]), md=6),
            dbc.Col(html.Div([
                _G(fig_heat),
                html.P("El heatmap agrupa por promedio de sentimiento. El filtro de Acción reduce el volumen disponible.",
                       style={"fontSize": "0.7rem", "color": COLOR_NEUTRAL_2, "marginTop": "4px"})
            ]), md=6),
        ], className="g-3"),
    ])


def render_retention(filters: dict) -> html.Div:
    esc_rate  = get_escalation_rate(filters)
    avg_cycle = get_average_behavior_cycle(filters)
    df_radar  = get_product_risk_radar(filters)
    df_topics = get_complaint_topics(filters)
    df_map    = get_state_intensity_map(filters)

    fig_radar = go.Figure()
    if not df_radar.empty:
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar["score"], theta=df_radar["product"], fill="toself",
            line_color=COLOR_PRIMARY, fillcolor="rgba(31, 78, 120, 0.12)",
            hovertemplate="<b>%{theta}</b><br>Nivel de Riesgo: %{r:.1f} / 100<extra></extra>",
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Productos con Mayor Riesgo de Deserción",
        font=dict(family=FONT_FAMILY, color=COLOR_NEUTRAL_DARK),
    )

    fig_topics = px.bar(
        df_topics, x="Frecuencia", y="Topic", orientation="h",
        title="Principales Motivos de Queja",
        labels={"Frecuencia": "Menciones en Reseñas", "Topic": "Categoría"},
    )
    fig_topics.update_traces(
        marker_color=COLOR_DANGER,
        hovertemplate="<b>%{y}</b><br>Menciones: %{x:,}<extra></extra>",
    )
    fig_topics = apply_corporate_layout(fig_topics, margin=dict(l=180), hide_x_title=True, hide_y_title=True)

    fig_map = px.bar(
        df_map, x="estado", y="quejas",
        title="Concentración Geográfica de Quejas",
        labels={"estado": "Estado / Región", "quejas": "Quejas Registradas"},
    )
    fig_map.update_traces(
        marker_color=COLOR_WARNING,
        hovertemplate="<b>%{x}</b><br>Quejas: %{y:,}<extra></extra>",
    )
    fig_map = apply_corporate_layout(fig_map, hide_x_title=True, hide_y_title=True)

    avg_cycle_str = f"{avg_cycle} días" if avg_cycle > 0 else "N/D"

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card(
                f"{esc_rate}%", "Tasa de Escalada", STYLE_KPI_DANGER,
                subtitle="Quejas que escalaron a reclamo formal — indica severidad del problema",
            ), md=6),
            dbc.Col(kpi_card(
                avg_cycle_str, "Tiempo Hasta la Deserción", STYLE_KPI_WARNING,
                subtitle="Días promedio entre la primera señal negativa y la cancelación",
            ), md=6),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(_G(fig_radar),  md=6),
            dbc.Col(_G(fig_topics), md=6),
        ], className="mb-4 g-3"),
        dbc.Row([dbc.Col(_G(fig_map), md=12)], className="g-3"),
    ])


def render_product_team(filters: dict) -> html.Div:
    device_filters = {k: v for k, v in filters.items() if k != "sources"}
    df_dev   = get_device_usage_comparison(device_filters)
    df_nlp   = get_app_reviews_nlp(device_filters)
    df_yoy   = get_yoy_volume_and_sentiment(device_filters)
    df_stars = get_rating_distribution(device_filters)

    banner = html.Div([
        html.Div([
            html.Div(
                html.I(className="bi bi-phone-fill", style={"fontSize": "1.5rem", "color": COLOR_ACCENT}),
                style={
                    "width": "54px", "height": "54px", "borderRadius": "14px",
                    "backgroundColor": "rgba(46,117,182,0.08)",
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "flexShrink": "0",
                },
            ),
            html.Div([
                html.Div([
                    html.Span("ANÁLISIS MÓVIL", style={
                        "fontSize": "0.62rem", "fontWeight": "700",
                        "letterSpacing": "0.12em", "color": COLOR_ACCENT,
                    }),
                    html.Span("  ·  Vista Exclusiva", style={
                        "fontSize": "0.62rem", "fontWeight": "500",
                        "color": COLOR_NEUTRAL_2,
                    }),
                ], style={"marginBottom": "3px"}),
                html.Div("Datos filtrados para App Store y Google Play", style={
                    "fontWeight": "700", "color": COLOR_PRIMARY, "fontSize": "0.95rem",
                }),
                html.Div(
                    "Las métricas de esta vista reflejan exclusivamente la experiencia en canales móviles",
                    style={"fontSize": "0.76rem", "color": COLOR_NEUTRAL_DARK, "marginTop": "2px"},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Span([html.I(className="bi bi-apple me-1"), "App Store"], style={
                    "display": "inline-flex", "alignItems": "center",
                    "backgroundColor": COLOR_PRIMARY, "color": "white",
                    "fontSize": "0.72rem", "fontWeight": "600",
                    "padding": "5px 12px", "borderRadius": "20px", "marginRight": "6px",
                }),
                html.Span([html.I(className="bi bi-google-play me-1"), "Google Play"], style={
                    "display": "inline-flex", "alignItems": "center",
                    "backgroundColor": "rgba(112,173,71,0.10)", "color": COLOR_SUCCESS,
                    "fontSize": "0.72rem", "fontWeight": "600",
                    "padding": "5px 12px", "borderRadius": "20px",
                    "border": "1.5px solid rgba(112,173,71,0.30)",
                }),
            ], style={"flexShrink": "0"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "18px"}),
    ], style={
        "backgroundColor": COLOR_BACKGROUND,
        "border": f"1px solid {COLOR_BORDER}",
        "borderLeft": f"4px solid {COLOR_ACCENT}",
        "borderRadius": "14px",
        "padding": "18px 22px",
        "marginBottom": "24px",
        "boxShadow": "0 2px 10px rgba(31,78,120,0.07)",
    })

    fig_dev = go.Figure()
    if not df_dev.empty:
        if "avg_rating" in df_dev.columns:
            sentiment_norm = ((df_dev["avg_sentiment"] + 1) / 2) * 5
            fig_dev.add_trace(go.Bar(
                y=df_dev["source"], x=sentiment_norm, name="Satisfacción (normalizada)",
                orientation="h", marker_color=COLOR_PRIMARY,
                customdata=df_dev["avg_sentiment"],
                hovertemplate="<b>%{y}</b><br>Índice de Satisfacción: %{customdata:.3f}<br>Equivalente: %{x:.2f}/5<extra></extra>",
            ))
            fig_dev.add_trace(go.Bar(
                y=df_dev["source"], x=df_dev["avg_rating"], name="Calificación Promedio",
                orientation="h", marker_color=COLOR_ACCENT,
                hovertemplate="<b>%{y}</b><br>Calificación: %{x:.2f} / 5 ★<extra></extra>",
            ))
            fig_dev.update_layout(
                title="Satisfacción vs Calificación por Plataforma (escala 0–5)",
                xaxis=dict(range=[0, 5]),
            )
        else:
            fig_dev.add_trace(go.Bar(
                y=df_dev["source"], x=df_dev["avg_sentiment"], name="Índice de Satisfacción",
                orientation="h", marker_color=COLOR_PRIMARY,
                hovertemplate="<b>%{y}</b><br>Índice de Satisfacción: %{x:.3f}<extra></extra>",
            ))
            fig_dev.update_layout(title="Satisfacción por Plataforma Móvil")
    else:
        fig_dev.add_annotation(
            text="No se encontraron datos de App Store o Google Play.<br>"
                 "Verificá que el dataset incluya la columna 'source' con valores 'AppStore' o 'Google Play'.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color=COLOR_NEUTRAL_2, family=FONT_FAMILY),
            align="center",
        )
        fig_dev.update_layout(title="Sin datos de plataformas móviles")
    fig_dev = apply_corporate_layout(fig_dev, barmode="group", margin=dict(l=120), hide_x_title=True, hide_y_title=True)

    fig_nlp = px.bar(
        df_nlp, x="Frecuencia", y="Problema", orientation="h",
        title="Problemas Técnicos Más Reportados",
        labels={"Frecuencia": "Cantidad de Reportes", "Problema": "Tipo de Problema"},
    )
    fig_nlp.update_traces(
        marker_color=COLOR_DANGER,
        hovertemplate="<b>%{y}</b><br>Reportes: %{x:,}<extra></extra>",
    )
    fig_nlp = apply_corporate_layout(fig_nlp, margin=dict(l=150), hide_x_title=True, hide_y_title=True)

    fig_yoy = go.Figure()
    if not df_yoy.empty:
        fig_yoy.add_trace(go.Bar(
            x=df_yoy["year"], y=df_yoy["volumen"], name="Volumen de Reseñas",
            marker_color=COLOR_BORDER, yaxis="y1",
            hovertemplate="<b>%{x}</b><br>Reseñas: %{y:,}<extra></extra>",
        ))
        fig_yoy.add_trace(go.Scatter(
            x=df_yoy["year"], y=df_yoy["avg_sentiment"], name="Índice de Satisfacción",
            mode="lines+markers", marker_color=COLOR_PRIMARY,
            line=dict(width=2.5), yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Índice de Satisfacción: %{y:.3f}<extra></extra>",
        ))
    fig_yoy.update_layout(
        title="Tendencia Anual: Volumen de Reseñas vs Satisfacción",
        yaxis=dict(title="Volumen de Reseñas", side="left"),
        yaxis2=dict(title="Índice de Satisfacción", side="right", overlaying="y", showgrid=False),
    )
    fig_yoy = apply_corporate_layout(fig_yoy, hide_x_title=True)

    # Star rating distribution
    fig_stars = go.Figure()
    if not df_stars.empty and df_stars["cantidad"].sum() > 0:
        star_colors = [COLOR_DANGER, COLOR_DANGER, COLOR_WARNING, COLOR_SUCCESS, COLOR_SUCCESS]
        fig_stars.add_trace(go.Bar(
            x=df_stars["estrella_label"],
            y=df_stars["cantidad"],
            marker_color=star_colors[:len(df_stars)],
            hovertemplate="<b>%{x}</b><br>Reseñas: %{y:,}<extra></extra>",
        ))
    else:
        fig_stars.add_annotation(
            text="Sin datos de calificaciones para este período",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLOR_NEUTRAL_2, size=13),
        )
    fig_stars.update_layout(title="Distribución de Calificaciones — App Store & Google Play")
    fig_stars = apply_corporate_layout(fig_stars, hide_x_title=True, hide_y_title=True)

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col(_G(fig_dev),   md=6),
            dbc.Col(_G(fig_stars), md=6),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(_G(fig_nlp), md=6),
            dbc.Col(_G(fig_yoy), md=6),
        ], className="g-3"),
    ])


if __name__ == "__main__":
    app.server.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
    app.run(debug=False, port=8050)
