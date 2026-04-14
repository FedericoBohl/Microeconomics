from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuración general
# ============================================================
st.set_page_config(
    page_title="Oferta, demanda y bienestar",
    layout="wide",
)

# ============================================================
# Estado persistente
# ============================================================
if "seccion_activa" not in st.session_state:
    st.session_state["seccion_activa"] = "Equilibrio"

if "p_control_store" not in st.session_state:
    st.session_state["p_control_store"] = None

if "t_tax_store" not in st.session_state:
    st.session_state["t_tax_store"] = 0.0

if "incidencia_legal_store" not in st.session_state:
    st.session_state["incidencia_legal_store"] = "Consumidor"


# ============================================================
# Curvas
# ============================================================
@dataclass
class InverseCurve:
    side: str                 # "demand" o "supply"
    kind: str                 # "finite", "horizontal", "vertical"
    intercept: float | None = None   # a o c
    slope: float | None = None       # b o d
    qbar: float | None = None        # posición de la vertical


def build_demand(a, b, qbar_d=None):
    if np.isinf(b):
        return InverseCurve(
            side="demand",
            kind="vertical",
            qbar=float(qbar_d),
        )
    if b == 0:
        return InverseCurve(
            side="demand",
            kind="horizontal",
            intercept=float(a),
        )
    return InverseCurve(
        side="demand",
        kind="finite",
        intercept=float(a),
        slope=float(b),
    )


def build_supply(c, d, qbar_s=None):
    if np.isinf(d):
        return InverseCurve(
            side="supply",
            kind="vertical",
            qbar=float(qbar_s),
        )
    if d == 0:
        return InverseCurve(
            side="supply",
            kind="horizontal",
            intercept=float(c),
        )
    return InverseCurve(
        side="supply",
        kind="finite",
        intercept=float(c),
        slope=float(d),
    )


# ============================================================
# Precios efectivos (sin precios negativos)
# ============================================================
def effective_demand_price(demand: InverseCurve, q):
    q = np.asarray(q, dtype=float)

    if demand.kind == "finite":
        return np.maximum(demand.intercept - demand.slope * q, 0.0)

    if demand.kind == "horizontal":
        return np.full_like(q, max(demand.intercept, 0.0), dtype=float)

    return np.full_like(q, np.nan, dtype=float)


def effective_supply_price(supply: InverseCurve, q):
    q = np.asarray(q, dtype=float)

    if supply.kind == "finite":
        return np.maximum(supply.intercept + supply.slope * q, 0.0)

    if supply.kind == "horizontal":
        return np.full_like(q, max(supply.intercept, 0.0), dtype=float)

    return np.full_like(q, np.nan, dtype=float)


# ============================================================
# Utilidades económicas
# ============================================================
def valid_result(q, p, tol=1e-10):
    if q is None or p is None:
        return False
    if not np.isfinite(q) or not np.isfinite(p):
        return False
    if q < -tol or p < -tol:
        return False
    return True


def solve_equilibrium(demand: InverseCurve, supply: InverseCurve, tol=1e-10):
    def ok_or_noeq(q, p, msg_noeq="No hay equilibrio en el primer cuadrante"):
        if valid_result(q, p, tol=tol):
            return {"status": "ok", "q": max(q, 0.0), "p": max(p, 0.0)}
        return {"status": "no_equilibrium", "msg": msg_noeq}

    # Ambas finitas
    if demand.kind == "finite" and supply.kind == "finite":
        if supply.intercept >= 0:
            denom = demand.slope + supply.slope
            if denom <= tol:
                return {"status": "indeterminate", "msg": "b + d = 0"}
            q = (demand.intercept - supply.intercept) / denom
            p = demand.intercept - demand.slope * q
            return ok_or_noeq(q, p)

        q_zero_supply_end = -supply.intercept / supply.slope
        q_zero_demand = demand.intercept / demand.slope

        if q_zero_demand <= q_zero_supply_end + tol:
            return ok_or_noeq(q_zero_demand, 0.0)

        denom = demand.slope + supply.slope
        q = (demand.intercept - supply.intercept) / denom
        p = demand.intercept - demand.slope * q
        return ok_or_noeq(q, p)

    # Demanda horizontal, oferta finita
    if demand.kind == "horizontal" and supply.kind == "finite":
        p = max(demand.intercept, 0.0)
        q = (p - supply.intercept) / supply.slope
        return ok_or_noeq(q, p)

    # Demanda finita, oferta horizontal
    if demand.kind == "finite" and supply.kind == "horizontal":
        p = max(supply.intercept, 0.0)
        q = (demand.intercept - p) / demand.slope
        return ok_or_noeq(q, p)

    # Demanda vertical, oferta finita
    if demand.kind == "vertical" and supply.kind == "finite":
        q = demand.qbar
        p = float(effective_supply_price(supply, np.array([q]))[0])
        return ok_or_noeq(
            q, p,
            "La demanda vertical no corta a la oferta finita en el primer cuadrante"
        )

    # Demanda finita, oferta vertical
    if demand.kind == "finite" and supply.kind == "vertical":
        q = supply.qbar
        q_zero_demand = demand.intercept / demand.slope
        if q > q_zero_demand + tol:
            return {
                "status": "no_equilibrium",
                "msg": "La oferta vertical fija una cantidad mayor que la demandada incluso a precio cero"
            }
        p = float(effective_demand_price(demand, np.array([q]))[0])
        return ok_or_noeq(
            q, p,
            "La oferta vertical no corta a la demanda finita en el primer cuadrante"
        )

    # Demanda horizontal, oferta vertical
    if demand.kind == "horizontal" and supply.kind == "vertical":
        q = supply.qbar
        p = max(demand.intercept, 0.0)
        return ok_or_noeq(q, p)

    # Demanda vertical, oferta horizontal
    if demand.kind == "vertical" and supply.kind == "horizontal":
        q = demand.qbar
        p = max(supply.intercept, 0.0)
        return ok_or_noeq(q, p)

    # Ambas horizontales
    if demand.kind == "horizontal" and supply.kind == "horizontal":
        p_d = max(demand.intercept, 0.0)
        p_s = max(supply.intercept, 0.0)

        if abs(p_d - p_s) <= tol:
            return {
                "status": "continuum",
                "msg": "Demanda y oferta horizontales al mismo precio: continuo de equilibrios"
            }
        return {
            "status": "indeterminate",
            "msg": "Ambas curvas son horizontales a distinto precio: no hay equilibrio único"
        }

    # Ambas verticales
    if demand.kind == "vertical" and supply.kind == "vertical":
        if abs(demand.qbar - supply.qbar) <= tol:
            return {
                "status": "continuum",
                "msg": "Demanda y oferta verticales en la misma cantidad: no hay precio único"
            }
        return {
            "status": "no_equilibrium",
            "msg": "Demanda y oferta verticales en cantidades distintas"
        }

    return {"status": "indeterminate", "msg": "Caso no manejado"}


def consumer_surplus(demand: InverseCurve, q, p):
    if q is None or p is None or q <= 0:
        return 0.0

    if demand.kind == "finite":
        q_cap = max((demand.intercept - p) / demand.slope, 0.0)
        q_eff = min(q, q_cap)
        if q_eff <= 0:
            return 0.0
        return (demand.intercept - p) * q_eff - 0.5 * demand.slope * q_eff**2

    if demand.kind == "horizontal":
        return max(demand.intercept - p, 0.0) * q

    if demand.kind == "vertical":
        return np.inf

    return 0.0


def producer_surplus(supply: InverseCurve, q, p):
    if q is None or p is None or q <= 0:
        return 0.0

    if supply.kind in ("finite", "horizontal"):
        q_grid = np.linspace(0, q, 500)
        p_supply = effective_supply_price(supply, q_grid)
        gap = np.maximum(p - p_supply, 0.0)
        return float(np.trapz(gap, q_grid))

    if supply.kind == "vertical":
        return np.inf

    return 0.0


def elasticity_display_demand(demand: InverseCurve, q, p):
    if q is None or p is None or q <= 0:
        return "N/D"
    if demand.kind == "finite":
        return f"{(-(1 / demand.slope) * (p / q)):.3f}"
    if demand.kind == "horizontal":
        return "−∞"
    if demand.kind == "vertical":
        return "0"
    return "N/D"


def elasticity_display_supply(supply: InverseCurve, q, p):
    if q is None or p is None or q <= 0:
        return "N/D"
    if supply.kind == "finite":
        return f"{((1 / supply.slope) * (p / q)):.3f}"
    if supply.kind == "horizontal":
        return "∞"
    if supply.kind == "vertical":
        return "0"
    return "N/D"


def fmt_metric(x):
    if x is None:
        return "N/D"
    if np.isinf(x):
        return "∞"
    return f"{x:.3f}"


# ============================================================
# Utilidades gráficas
# ============================================================
def add_polygon_fill(fig, x, y, color_rgba, name,group):
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=0),
        fill="toself",
        fillcolor=color_rgba,
        name=name,
        hoverinfo="skip",
        legendgroup=group
    ))


def add_consumer_surplus_fill(fig, demand: InverseCurve, q_eq, p_eq, p_max, tol=1e-10):
    if q_eq <= tol:
        return

    if demand.kind == "finite":
        q_cap = max((demand.intercept - p_eq) / demand.slope, 0.0)
        q_fill_max = min(q_eq, q_cap)
        if q_fill_max <= tol:
            return

        q_fill = np.linspace(0, q_fill_max, 300)
        p_top = demand.intercept - demand.slope * q_fill
        p_bottom = np.full_like(q_fill, p_eq)

        x_poly = np.concatenate([q_fill, q_fill[::-1]])
        y_poly = np.concatenate([p_top, p_bottom[::-1]])

        add_polygon_fill(fig, x_poly, y_poly, "rgba(0, 0, 255, 0.18)", "Excedente del consumidor",'D')
        return

    if demand.kind == "horizontal":
        if demand.intercept <= p_eq + tol:
            return

        x_poly = [0, q_eq, q_eq, 0]
        y_poly = [p_eq, p_eq, demand.intercept, demand.intercept]

        add_polygon_fill(fig, x_poly, y_poly, "rgba(0, 0, 255, 0.18)", "Excedente del consumidor",'D')
        return

    if demand.kind == "vertical":
        if p_max <= p_eq + tol:
            return

        x_poly = [0, q_eq, q_eq, 0]
        y_poly = [p_eq, p_eq, p_max, p_max]

        add_polygon_fill(fig, x_poly, y_poly, "rgba(0, 0, 255, 0.18)", "Excedente del consumidor",'D')


def add_producer_surplus_fill(fig, supply: InverseCurve, q_eq, p_eq, floor_price=0.0, tol=1e-10):
    if q_eq <= tol:
        return

    if supply.kind in ("finite", "horizontal"):
        q_fill = np.linspace(0, q_eq, 300)
        p_top = np.full_like(q_fill, p_eq)
        p_bottom = effective_supply_price(supply, q_fill)

        mask = p_top > p_bottom + tol
        if not np.any(mask):
            return

        q_fill = q_fill[mask]
        p_top = p_top[mask]
        p_bottom = p_bottom[mask]

        x_poly = np.concatenate([q_fill, q_fill[::-1]])
        y_poly = np.concatenate([p_top, p_bottom[::-1]])

        add_polygon_fill(fig, x_poly, y_poly, "rgba(255, 0, 0, 0.18)", "Excedente del productor",'S')
        return

    if supply.kind == "vertical":
        if p_eq <= floor_price + tol:
            return

        x_poly = [0, q_eq, q_eq, 0]
        y_poly = [floor_price, floor_price, p_eq, p_eq]

        add_polygon_fill(fig, x_poly, y_poly, "rgba(255, 0, 0, 0.18)", "Excedente del productor",'S')


def add_tax_revenue_fill(fig, q_tax, p_v, p_c, tol=1e-10):
    if q_tax is None or q_tax <= tol:
        return
    if p_v is None or p_c is None:
        return
    if p_c <= p_v + tol:
        return

    x_poly = [0, q_tax, q_tax, 0]
    y_poly = [p_v, p_v, p_c, p_c]

    add_polygon_fill(fig, x_poly, y_poly, "rgba(0, 128, 0, 0.22)", "Recaudación",'R')


def add_equilibrium_axis_labels(fig, q_eq, p_eq):
    fig.add_trace(go.Scatter(
        x=[q_eq],
        y=[0],
        mode="markers",
        marker=dict(color="black", size=7),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=[0],
        y=[p_eq],
        mode="markers",
        marker=dict(color="black", size=7),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_annotation(
        x=q_eq,
        y=0,
        text="Q*",
        showarrow=False,
        yshift=-18,
        bgcolor="white",
    )

    fig.add_annotation(
        x=0,
        y=p_eq,
        text="P*",
        showarrow=False,
        xshift=-18,
        bgcolor="white",
    )


# ============================================================
# Controles de precios
# ============================================================
def quantity_demanded_at_price(demand: InverseCurve, p, tol=1e-10):
    p = max(float(p), 0.0)

    if demand.kind == "finite":
        return max((demand.intercept - p) / demand.slope, 0.0)

    if demand.kind == "horizontal":
        return np.inf if p <= max(demand.intercept, 0.0) + tol else 0.0

    if demand.kind == "vertical":
        return demand.qbar

    return np.nan


def quantity_supplied_at_price(supply: InverseCurve, p, tol=1e-10):
    p = max(float(p), 0.0)

    if supply.kind == "finite":
        return max((p - supply.intercept) / supply.slope, 0.0)

    if supply.kind == "horizontal":
        return np.inf if p >= max(supply.intercept, 0.0) - tol else 0.0

    if supply.kind == "vertical":
        return supply.qbar

    return np.nan


def controlled_market_outcome(demand: InverseCurve, supply: InverseCurve, p_control, eq, tol=1e-10):
    if eq["status"] != "ok":
        return {
            "status": "no_equilibrium",
            "msg": "No se puede analizar el control de precios sin equilibrio competitivo de referencia"
        }

    p_control = max(float(p_control), 0.0)
    p_eq = eq["p"]

    if abs(p_control - p_eq) <= tol:
        regime = "No vinculante"
    elif p_control < p_eq:
        regime = "Precio máximo"
    else:
        regime = "Precio mínimo"

    qd = quantity_demanded_at_price(demand, p_control, tol=tol)
    qs = quantity_supplied_at_price(supply, p_control, tol=tol)

    if np.isinf(qd) and np.isinf(qs):
        return {
            "status": "indeterminate",
            "msg": "A ese precio, tanto demanda como oferta admiten cantidades no acotadas"
        }

    q_traded = min(qd, qs)

    if not np.isfinite(q_traded):
        return {
            "status": "indeterminate",
            "msg": "La cantidad transada no es finita"
        }

    if q_traded < 0:
        return {
            "status": "no_equilibrium",
            "msg": "La cantidad transada resultó negativa"
        }

    cs = consumer_surplus(demand, q_traded, p_control)
    ps = producer_surplus(supply, q_traded, p_control)

    if np.isinf(cs) or np.isinf(ps):
        ts = np.inf
    else:
        ts = cs + ps

    return {
        "status": "ok",
        "regime": regime,
        "p": p_control,
        "q": q_traded,
        "qd": qd,
        "qs": qs,
        "cs": cs,
        "ps": ps,
        "ts": ts,
    }


def add_dwl_fill(fig, demand: InverseCurve, supply: InverseCurve, q_control, q_eq, p_max, tol=1e-10):
    if q_control is None or q_eq is None:
        return

    if q_control >= q_eq - tol:
        return

    q_fill = np.linspace(q_control, q_eq, 300)

    if demand.kind == "finite":
        p_top = effective_demand_price(demand, q_fill)
    elif demand.kind == "horizontal":
        p_top = np.full_like(q_fill, max(demand.intercept, 0.0))
    elif demand.kind == "vertical":
        p_top = np.full_like(q_fill, p_max)
    else:
        return

    if supply.kind in ("finite", "horizontal"):
        p_bottom = effective_supply_price(supply, q_fill)
    elif supply.kind == "vertical":
        p_bottom = np.zeros_like(q_fill)
    else:
        return

    mask = p_top > p_bottom + tol
    if not np.any(mask):
        return

    q_fill = q_fill[mask]
    p_top = p_top[mask]
    p_bottom = p_bottom[mask]

    x_poly = np.concatenate([q_fill, q_fill[::-1]])
    y_poly = np.concatenate([p_top, p_bottom[::-1]])

    add_polygon_fill(fig, x_poly, y_poly, "rgba(139, 69, 19, 0.28)", "DWL",'DWL')


def make_distribution_figure(cs, ps, dwl, title):
    fig_shares = go.Figure()

    if cs is None or ps is None or dwl is None:
        return None

    if np.isinf(cs) or np.isinf(ps) or np.isinf(dwl):
        return None

    total_reference = cs + ps + dwl
    if total_reference <= 0:
        return None

    share_cs = 100 * cs / total_reference
    share_ps = 100 * ps / total_reference
    share_dwl = 100 * dwl / total_reference

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_cs],
        name="Consumidor",
        marker_color="blue",
        text=[f"{share_cs:.1f}%"],
        textposition="inside"
    ))

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_ps],
        name="Productor",
        marker_color="red",
        text=[f"{share_ps:.1f}%"],
        textposition="inside"
    ))

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_dwl],
        name="DWL",
        marker_color="brown",
        text=[f"{share_dwl:.1f}%"],
        textposition="inside"
    ))

    fig_shares.update_layout(
        barmode="stack",
        height=680,
        yaxis=dict(
            title="Porcentaje del excedente total potencial",
            range=[0, 100],
            ticksuffix="%"
        ),
        xaxis_title="",
        legend_title="Participación",
        title=title
    )

    return fig_shares


# ============================================================
# Impuestos
# ============================================================
def tax_gap_at_quantity(q, demand: InverseCurve, supply: InverseCurve, tax):
    p_c = float(effective_demand_price(demand, np.array([q]))[0])
    p_v = float(effective_supply_price(supply, np.array([q]))[0])
    return p_c - p_v - tax


def tax_market_outcome(demand: InverseCurve, supply: InverseCurve, tax, legal_incidence, eq, tol=1e-10):
    if eq["status"] != "ok":
        return {
            "status": "no_equilibrium",
            "msg": "No se puede analizar el impuesto sin equilibrio competitivo de referencia"
        }

    tax = max(float(tax), 0.0)
    p_star = eq["p"]
    q_star = eq["q"]

    if tax <= tol:
        cs = consumer_surplus(demand, q_star, p_star)
        ps = producer_surplus(supply, q_star, p_star)
        revenue = 0.0
        ts = np.inf if (np.isinf(cs) or np.isinf(ps)) else cs + ps
        return {
            "status": "ok",
            "legal_incidence": legal_incidence,
            "q": q_star,
            "p_c": p_star,
            "p_v": p_star,
            "tax": tax,
            "revenue": revenue,
            "cs": cs,
            "ps": ps,
            "ts": ts,
            "consumer_burden_total": 0.0,
            "producer_burden_total": 0.0,
            "consumer_burden_share": 0.0,
            "producer_burden_share": 0.0,
        }

    if demand.kind == "vertical" and supply.kind == "vertical":
        if abs(demand.qbar - supply.qbar) <= tol:
            return {
                "status": "indeterminate",
                "msg": "Con ambas curvas verticales en la misma cantidad, el impuesto no determina un par único de precios"
            }
        return {
            "status": "no_equilibrium",
            "msg": "Con ambas curvas verticales en cantidades distintas, no hay mercado antes ni después del impuesto"
        }

    if demand.kind == "vertical":
        q_tax = demand.qbar
        p_v = float(effective_supply_price(supply, np.array([q_tax]))[0])
        p_c = p_v + tax

        cs = consumer_surplus(demand, q_tax, p_c)
        ps = producer_surplus(supply, q_tax, p_v)
        revenue = tax * q_tax
        ts = np.inf if (np.isinf(cs) or np.isinf(ps)) else cs + ps + revenue

        cons_total = max(p_c - p_star, 0.0) * q_tax
        prod_total = max(p_star - p_v, 0.0) * q_tax
        burden_total = cons_total + prod_total

        return {
            "status": "ok",
            "legal_incidence": legal_incidence,
            "q": q_tax,
            "p_c": p_c,
            "p_v": p_v,
            "tax": tax,
            "revenue": revenue,
            "cs": cs,
            "ps": ps,
            "ts": ts,
            "consumer_burden_total": cons_total,
            "producer_burden_total": prod_total,
            "consumer_burden_share": 0.0 if burden_total <= tol else cons_total / burden_total,
            "producer_burden_share": 0.0 if burden_total <= tol else prod_total / burden_total,
        }

    if supply.kind == "vertical":
        q_tax = supply.qbar
        p_c = float(effective_demand_price(demand, np.array([q_tax]))[0])
        p_v = p_c - tax

        if p_v < -tol:
            return {
                "status": "no_trade",
                "msg": "El impuesto elimina el intercambio: el precio neto del vendedor sería negativo",
                "legal_incidence": legal_incidence,
                "q": 0.0,
                "p_c": None,
                "p_v": None,
                "tax": tax,
                "revenue": 0.0,
                "cs": 0.0,
                "ps": 0.0,
                "ts": 0.0,
                "consumer_burden_total": 0.0,
                "producer_burden_total": 0.0,
                "consumer_burden_share": 0.0,
                "producer_burden_share": 0.0,
            }

        cs = consumer_surplus(demand, q_tax, p_c)
        ps = producer_surplus(supply, q_tax, p_v)
        revenue = tax * q_tax
        ts = np.inf if (np.isinf(cs) or np.isinf(ps)) else cs + ps + revenue

        cons_total = max(p_c - p_star, 0.0) * q_tax
        prod_total = max(p_star - p_v, 0.0) * q_tax
        burden_total = cons_total + prod_total

        return {
            "status": "ok",
            "legal_incidence": legal_incidence,
            "q": q_tax,
            "p_c": p_c,
            "p_v": p_v,
            "tax": tax,
            "revenue": revenue,
            "cs": cs,
            "ps": ps,
            "ts": ts,
            "consumer_burden_total": cons_total,
            "producer_burden_total": prod_total,
            "consumer_burden_share": 0.0 if burden_total <= tol else cons_total / burden_total,
            "producer_burden_share": 0.0 if burden_total <= tol else prod_total / burden_total,
        }

    if q_star <= tol:
        return {
            "status": "no_trade",
            "msg": "El mercado competitivo ya no transa cantidad positiva",
            "legal_incidence": legal_incidence,
            "q": 0.0,
            "p_c": None,
            "p_v": None,
            "tax": tax,
            "revenue": 0.0,
            "cs": 0.0,
            "ps": 0.0,
            "ts": 0.0,
            "consumer_burden_total": 0.0,
            "producer_burden_total": 0.0,
            "consumer_burden_share": 0.0,
            "producer_burden_share": 0.0,
        }

    g0 = tax_gap_at_quantity(0.0, demand, supply, tax)
    if g0 <= tol:
        return {
            "status": "no_trade",
            "msg": "El impuesto elimina el intercambio",
            "legal_incidence": legal_incidence,
            "q": 0.0,
            "p_c": None,
            "p_v": None,
            "tax": tax,
            "revenue": 0.0,
            "cs": 0.0,
            "ps": 0.0,
            "ts": 0.0,
            "consumer_burden_total": 0.0,
            "producer_burden_total": 0.0,
            "consumer_burden_share": 0.0,
            "producer_burden_share": 0.0,
        }

    lo, hi = 0.0, q_star
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        gm = tax_gap_at_quantity(mid, demand, supply, tax)
        if gm > 0:
            lo = mid
        else:
            hi = mid

    q_tax = 0.5 * (lo + hi)
    p_c = float(effective_demand_price(demand, np.array([q_tax]))[0])
    p_v = float(effective_supply_price(supply, np.array([q_tax]))[0])

    cs = consumer_surplus(demand, q_tax, p_c)
    ps = producer_surplus(supply, q_tax, p_v)
    revenue = tax * q_tax
    ts = np.inf if (np.isinf(cs) or np.isinf(ps)) else cs + ps + revenue

    cons_total = max(p_c - p_star, 0.0) * q_tax
    prod_total = max(p_star - p_v, 0.0) * q_tax
    burden_total = cons_total + prod_total

    return {
        "status": "ok",
        "legal_incidence": legal_incidence,
        "q": q_tax,
        "p_c": p_c,
        "p_v": p_v,
        "tax": tax,
        "revenue": revenue,
        "cs": cs,
        "ps": ps,
        "ts": ts,
        "consumer_burden_total": cons_total,
        "producer_burden_total": prod_total,
        "consumer_burden_share": 0.0 if burden_total <= tol else cons_total / burden_total,
        "producer_burden_share": 0.0 if burden_total <= tol else prod_total / burden_total,
    }


def make_tax_distribution_figure(cs, ps, revenue, dwl, title):
    fig_shares = go.Figure()

    if cs is None or ps is None or revenue is None or dwl is None:
        return None

    if np.isinf(cs) or np.isinf(ps) or np.isinf(revenue) or np.isinf(dwl):
        return None

    total_reference = cs + ps + revenue + dwl
    if total_reference <= 0:
        return None

    share_cs = 100 * cs / total_reference
    share_ps = 100 * ps / total_reference
    share_rev = 100 * revenue / total_reference
    share_dwl = 100 * dwl / total_reference

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_cs],
        name="Consumidor",
        marker_color="blue",
        text=[f"{share_cs:.1f}%"],
        textposition="inside"
    ))

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_ps],
        name="Productor",
        marker_color="red",
        text=[f"{share_ps:.1f}%"],
        textposition="inside"
    ))

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_rev],
        name="Recaudación",
        marker_color="green",
        text=[f"{share_rev:.1f}%"],
        textposition="inside"
    ))

    fig_shares.add_trace(go.Bar(
        x=["Excedente total potencial"],
        y=[share_dwl],
        name="DWL",
        marker_color="brown",
        text=[f"{share_dwl:.1f}%"],
        textposition="inside"
    ))

    fig_shares.update_layout(
        barmode="stack",
        height=680,
        yaxis=dict(
            title="Porcentaje del excedente total potencial",
            range=[0, 100],
            ticksuffix="%"
        ),
        xaxis_title="",
        legend_title="Participación",
        title=title
    )

    return fig_shares


# ============================================================
# Helpers de rango y navegación
# ============================================================
def compute_plot_bounds(a, c, demand, supply, eq):
    q_candidates = [10.0]
    p_candidates = [0.0, a, max(c, 0.0)]

    if demand.kind == "finite":
        q_candidates.append(max(demand.intercept / demand.slope, 0.0))
    elif demand.kind == "vertical":
        q_candidates.append(max(demand.qbar, 0.0))

    if supply.kind == "vertical":
        q_candidates.append(max(supply.qbar, 0.0))
    elif supply.kind == "finite" and supply.intercept < 0:
        q_candidates.append(max(-supply.intercept / supply.slope, 0.0))

    if eq["status"] == "ok":
        q_candidates.append(max(eq["q"], 0.0))
        p_candidates.append(max(eq["p"], 0.0))

    q_max = max(q_candidates) * 1.15 + 1.0
    p_min = 0.0
    p_max = max(p_candidates) * 1.20 + 1.0
    return q_max, p_min, p_max

# ============================================================
# Texto principal
# ============================================================
st.title("Oferta, demanda y bienestar con controles de precios e impuestos")

with st.sidebar:
    st.subheader('Sección:')
    section_options = ["Equilibrio", "Precios Max/Min", "Impuestos"]
    seccion=st.pills('Sección',section_options,default=st.session_state["seccion_activa"],width='stretch',label_visibility='collapsed')
    st.session_state["seccion_activa"] = seccion



    st.caption(
        r"""
    Vamos a trabajar con oferta y demanda lineales en **forma inversa**:
    """
    )

    st.latex(r"P^d(Q)=a-bQ")
    st.latex(r"\varepsilon_d(Q,P)= -\frac{1}{b}\frac{P}{Q}")
    a = st.slider("$a$", 5.0, 20.0, value=12.0, step=0.5)
    b = st.select_slider("$b$", options=list(np.linspace(0, 5, 26)) + [np.inf], value=1.0)

    st.latex(r"P^s(Q)=c+dQ")
    st.latex(r"\varepsilon_s(Q,P)= \frac{1}{d}\frac{P}{Q}")
    c = st.slider("$c$", -5.0, float(a), value=2.0, step=0.5)
    d = st.select_slider("$d$", options=list(np.linspace(0, 5, 26)) + [np.inf], value=1.0)


# ============================================================
# Parámetros
# ============================================================
c1, c2, c3, c4 = st.columns(4)

qd_bar = None
qs_bar = None

if np.isinf(b):
    qd_bar = st.slider(r"$\bar Q_d$ (posición de la demanda vertical)", 0.0, 20.0, value=6.0, step=0.5)

if np.isinf(d):
    qs_bar = st.slider(r"$\bar Q_s$ (posición de la oferta vertical)", 0.0, 20.0, value=4.0, step=0.5)


# ============================================================
# TAB / SECCIÓN: EQUILIBRIO
# ============================================================
if seccion == "Equilibrio":
    demand = build_demand(a, b, qbar_d=qd_bar)
    supply = build_supply(c, d, qbar_s=qs_bar)
    eq = solve_equilibrium(demand, supply)

    q_max, p_min, p_max = compute_plot_bounds(a, c, demand, supply, eq)
    q_grid = np.linspace(0, q_max, 500)

    cs_base = None
    ps_base = None
    ts_base = None

    grafico, excendetes = st.columns([0.7, 0.3])
    fig = go.Figure()

    if demand.kind == "vertical":
        fig.add_trace(go.Scatter(
            x=[demand.qbar, demand.qbar],
            y=[p_min, p_max],
            mode="lines",
            name="Demanda",
            legendgroup='D',
            line=dict(color="blue", width=2),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=q_grid,
            y=effective_demand_price(demand, q_grid),
            mode="lines",
            name="Demanda",
            legendgroup='D',
            line=dict(color="blue", width=2),
        ))

    if supply.kind == "vertical":
        fig.add_trace(go.Scatter(
            x=[supply.qbar, supply.qbar],
            y=[p_min, p_max],
            mode="lines",
            name="Oferta",
            legendgroup='S',
            line=dict(color="red", width=2),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=q_grid,
            y=effective_supply_price(supply, q_grid),
            mode="lines",
            name="Oferta",
            legendgroup='S',
            line=dict(color="red", width=2),
        ))

    if eq["status"] == "ok":
        q_eq = eq["q"]
        p_eq = eq["p"]

        if q_eq >= 0 and p_eq >= 0:
            add_consumer_surplus_fill(fig, demand, q_eq, p_eq, p_max)
            add_producer_surplus_fill(fig, supply, q_eq, p_eq, floor_price=0.0)

            fig.add_vline(x=q_eq, line_dash="dot", line_color="gray")
            fig.add_hline(y=p_eq, line_dash="dot", line_color="gray")

            fig.add_trace(go.Scatter(
                x=[q_eq],
                y=[p_eq],
                mode="markers",
                name="Equilibrio",
                marker=dict(color="black", size=15),
            ))

            add_equilibrium_axis_labels(fig, q_eq, p_eq)

            cs_base = consumer_surplus(demand, q_eq, p_eq)
            ps_base = producer_surplus(supply, q_eq, p_eq)
            ts_base = np.inf if (np.isinf(cs_base) or np.isinf(ps_base)) else cs_base + ps_base

            c1, c2 = st.columns(2)
            c1.subheader("Equilibrio")
            m1, m2 = c1.columns(2)
            m1.metric("Cantidad de equilibrio", f"{q_eq:.3f}")
            m2.metric("Precio de equilibrio", f"{p_eq:.3f}")

            c2.subheader("Elasticidades en el equilibrio")
            e1, e2 = c2.columns(2)
            e1.metric("Elasticidad-precio de la demanda", elasticity_display_demand(demand, q_eq, p_eq))
            e2.metric("Elasticidad-precio de la oferta", elasticity_display_supply(supply, q_eq, p_eq))

            st.subheader("Excedentes")
            ccs, cps, et, dwl = st.columns(4)
            ccs.metric("Excedente del consumidor", fmt_metric(cs_base))
            cps.metric("Excedente del productor", fmt_metric(ps_base))
            et.metric("Excedente total", fmt_metric(ts_base))
            dwl.metric("Pérdida irrecuperable de eficiencia", "0.000")
    else:
        st.info(eq.get("msg", "No hay equilibrio único"))

    fig.update_layout(
        title="Oferta y demanda",
        xaxis_title="Cantidad",
        yaxis_title="Precio",
        hovermode="closest",
        height=680,
        xaxis=dict(range=[0, q_max]),
        yaxis=dict(range=[p_min, p_max]),
    )


    grafico.plotly_chart(fig, use_container_width=True)

    with excendetes:
        if cs_base is None or ps_base is None:
            st.info("No hay un equilibrio único para representar la distribución del excedente.")
        elif np.isinf(cs_base) or np.isinf(ps_base):
            st.info("No se puede graficar la participación porcentual cuando algún excedente es infinito.")
        else:
            total_surplus = cs_base + ps_base
            if total_surplus <= 0:
                st.info("No hay excedente total positivo para representar en porcentajes.")
            else:
                share_cs = 100 * cs_base / total_surplus
                share_ps = 100 * ps_base / total_surplus

                fig_shares = go.Figure()
                fig_shares.add_trace(go.Bar(
                    x=["Excedente total"],
                    y=[share_cs],
                    name="Consumidor",
                    marker_color="blue",
                    text=[f"{share_cs:.1f}%"],
                    textposition="inside"
                ))
                fig_shares.add_trace(go.Bar(
                    x=["Excedente total"],
                    y=[share_ps],
                    name="Productor",
                    marker_color="red",
                    text=[f"{share_ps:.1f}%"],
                    textposition="inside"
                ))

                fig_shares.update_layout(
                    barmode="stack",
                    height=680,
                    yaxis=dict(
                        title="Porcentaje del excedente total",
                        range=[0, 100],
                        ticksuffix="%"
                    ),
                    xaxis_title="",
                    legend_title="Participación",
                    title="Participación del excedente total"
                )

                st.plotly_chart(fig_shares, use_container_width=True)


# ============================================================
# TAB / SECCIÓN: PRECIOS MÁXIMOS / MÍNIMOS
# ============================================================
elif seccion == "Precios Max/Min":
    demand = build_demand(a, b, qbar_d=qd_bar)
    supply = build_supply(c, d, qbar_s=qs_bar)
    eq = solve_equilibrium(demand, supply)

    if eq["status"] != "ok":
        st.info(eq.get("msg", "No hay equilibrio competitivo de referencia"))
    else:
        q_eq = eq["q"]
        p_eq = eq["p"]

        cs_eq = consumer_surplus(demand, q_eq, p_eq)
        ps_eq = producer_surplus(supply, q_eq, p_eq)
        ts_eq = np.inf if (np.isinf(cs_eq) or np.isinf(ps_eq)) else cs_eq + ps_eq

        q_max, p_min, p_max = compute_plot_bounds(a, c, demand, supply, eq)
        q_grid = np.linspace(0, q_max, 500)

        st.subheader("Control de precios")

        if st.session_state["p_control_store"] is None:
            st.session_state["p_control_store"] = float(p_eq)

        control_slider_max = float(p_max)
        current_p_control = min(max(float(st.session_state["p_control_store"]), 0.0), control_slider_max)

        p_control_widget = st.slider(
            "Elegí el precio controlado",
            min_value=0.0,
            max_value=control_slider_max,
            value=current_p_control,
            step=0.1
        )

        st.session_state["p_control_store"] = float(p_control_widget)
        p_control = st.session_state["p_control_store"]

        outcome = controlled_market_outcome(demand, supply, p_control, eq)

        if outcome["status"] != "ok":
            st.info(outcome.get("msg", "No se pudo calcular el resultado con control de precios"))
        else:
            q_ctrl = outcome["q"]
            p_ctrl = outcome["p"]
            qd_ctrl = outcome["qd"]
            qs_ctrl = outcome["qs"]
            cs_ctrl = outcome["cs"]
            ps_ctrl = outcome["ps"]
            ts_ctrl = outcome["ts"]
            regime = outcome["regime"]

            dwl_value = np.inf if (np.isinf(ts_eq) or np.isinf(ts_ctrl)) else max(ts_eq - ts_ctrl, 0.0)

            grafico_precios, excedentes_precios = st.columns([0.7, 0.3])
            fig_precios = go.Figure()

            if demand.kind == "vertical":
                fig_precios.add_trace(go.Scatter(
                    x=[demand.qbar, demand.qbar],
                    y=[p_min, p_max],
                    mode="lines",
                    name="Demanda",
                    line=dict(color="blue", width=2),
                    legendgroup='D',
                ))
            else:
                fig_precios.add_trace(go.Scatter(
                    x=q_grid,
                    y=effective_demand_price(demand, q_grid),
                    mode="lines",
                    name="Demanda",
                    line=dict(color="blue", width=2),
                    legendgroup='D',
                ))

            if supply.kind == "vertical":
                fig_precios.add_trace(go.Scatter(
                    x=[supply.qbar, supply.qbar],
                    y=[p_min, p_max],
                    mode="lines",
                    name="Oferta",
                    line=dict(color="red", width=2),
                    legendgroup='S',
                ))
            else:
                fig_precios.add_trace(go.Scatter(
                    x=q_grid,
                    y=effective_supply_price(supply, q_grid),
                    mode="lines",
                    name="Oferta",
                    line=dict(color="red", width=2),
                    legendgroup='S',
                ))

            add_consumer_surplus_fill(fig_precios, demand, q_ctrl, p_ctrl, p_max)
            add_producer_surplus_fill(fig_precios, supply, q_ctrl, p_ctrl, floor_price=0.0)
            add_dwl_fill(fig_precios, demand, supply, q_ctrl, q_eq, p_max)

            fig_precios.add_hline(
                y=p_ctrl,
                line_dash="dash",
                line_color="black",
                annotation_text=regime,
                annotation_position="top left"
            )

            fig_precios.add_trace(go.Scatter(
                x=[q_eq],
                y=[p_eq],
                mode="markers",
                name="Equilibrio competitivo",
                marker=dict(color="black", size=11),
            ))

            fig_precios.add_trace(go.Scatter(
                x=[q_ctrl],
                y=[p_ctrl],
                mode="markers",
                name="Resultado con control",
                marker=dict(color="darkgreen", size=12, symbol="diamond"),
            ))

            fig_precios.add_vline(x=q_eq, line_dash="dot", line_color="gray")
            fig_precios.add_vline(x=q_ctrl, line_dash="dot", line_color="darkgreen")
            fig_precios.add_hline(y=p_eq, line_dash="dot", line_color="gray")

            fig_precios.update_layout(
                title="Mercado con precio máximo o mínimo",
                xaxis_title="Cantidad",
                yaxis_title="Precio",
                hovermode="closest",
                height=680,
                xaxis=dict(range=[0, q_max]),
                yaxis=dict(range=[p_min, p_max]),
            )

            grafico_precios.plotly_chart(fig_precios, use_container_width=True)

            with excedentes_precios:
                fig_dist = make_distribution_figure(
                    0.0 if np.isinf(cs_ctrl) else cs_ctrl,
                    0.0 if np.isinf(ps_ctrl) else ps_ctrl,
                    0.0 if np.isinf(dwl_value) else dwl_value,
                    "Distribución del excedente total potencial"
                )

                if fig_dist is None or np.isinf(cs_ctrl) or np.isinf(ps_ctrl) or np.isinf(dwl_value):
                    st.info("No se puede graficar la distribución porcentual cuando algún componente es infinito o no es finito.")
                else:
                    st.plotly_chart(fig_dist, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.subheader(regime)
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio controlado", f"{p_ctrl:.3f}")
                m2.metric("Cantidad transada", f"{q_ctrl:.3f}")
                m3.metric("Precio competitivo", f"{p_eq:.3f}")
            
            with c2.container(border=True):
                st.subheader("Cantidades al precio fijado")
                k1, k2, k3 = st.columns(3)
                k1.metric("Q demandada", "∞" if np.isinf(qd_ctrl) else f"{qd_ctrl:.3f}")
                k2.metric("Q ofrecida", "∞" if np.isinf(qs_ctrl) else f"{qs_ctrl:.3f}")
                k3.metric("Q competitiva", f"{q_eq:.3f}")

            with st.container(border=True):
                st.subheader("Bienestar con control de precios")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Excedente del consumidor", fmt_metric(cs_ctrl))
                b2.metric("Excedente del productor", fmt_metric(ps_ctrl))
                b3.metric("Excedente total", fmt_metric(ts_ctrl))
                b4.metric("DWL", fmt_metric(dwl_value))


# ============================================================
# TAB / SECCIÓN: IMPUESTOS
# ============================================================
elif seccion == "Impuestos":
    demand = build_demand(a, b, qbar_d=qd_bar)
    supply = build_supply(c, d, qbar_s=qs_bar)
    eq = solve_equilibrium(demand, supply)

    if eq["status"] != "ok":
        st.info(eq.get("msg", "No hay equilibrio competitivo de referencia para analizar impuestos"))
    else:
        q_eq = eq["q"]
        p_eq = eq["p"]

        cs_eq = consumer_surplus(demand, q_eq, p_eq)
        ps_eq = producer_surplus(supply, q_eq, p_eq)
        ts_eq = np.inf if (np.isinf(cs_eq) or np.isinf(ps_eq)) else cs_eq + ps_eq

        q_max, p_min, p_max_base = compute_plot_bounds(a, c, demand, supply, eq)

        incidencia_legal = st.pills(
            "¿Sobre quién recae legalmente el impuesto?",
            ["Consumidor", "Productor"],
            selection_mode="single",
            default=st.session_state["incidencia_legal_store"]
        )
        if incidencia_legal is None:
            incidencia_legal = st.session_state["incidencia_legal_store"]
        st.session_state["incidencia_legal_store"] = incidencia_legal

        tax_slider_max = max(
            float(max(p_eq * 1.5 + 2.0, 1.0)),
            float(st.session_state["t_tax_store"])
        )

        current_tax_value = min(float(st.session_state["t_tax_store"]), tax_slider_max)

        t_tax_widget = st.slider(
            "Monto del impuesto específico",
            min_value=0.0,
            max_value=tax_slider_max,
            value=current_tax_value,
            step=0.1
        )

        st.session_state["t_tax_store"] = float(t_tax_widget)
        t_tax = st.session_state["t_tax_store"]

        p_max = max(p_max_base, p_eq + t_tax) * 1.05
        q_grid = np.linspace(0, q_max, 500)

        st.subheader("Incidencia legal y precios con impuesto")
        if incidencia_legal == "Consumidor":
            st.latex(r"P_c = P_v + t")
        else:
            st.latex(r"P_v = P_c - t")

        st.caption("La incidencia legal cambia quién entrega formalmente el impuesto, pero el resultado económico final depende de la oferta y la demanda.")

        tax_outcome = tax_market_outcome(demand, supply, t_tax, incidencia_legal, eq)

        if tax_outcome["status"] not in ("ok", "no_trade"):
            st.info(tax_outcome.get("msg", "No se pudo calcular el equilibrio con impuesto"))
        else:
            q_tax = tax_outcome["q"]
            p_c_tax = tax_outcome["p_c"]
            p_v_tax = tax_outcome["p_v"]
            revenue_tax = tax_outcome["revenue"]
            cs_tax = tax_outcome["cs"]
            ps_tax = tax_outcome["ps"]
            ts_tax = tax_outcome["ts"]

            dwl_tax = np.inf if (np.isinf(ts_eq) or np.isinf(ts_tax)) else max(ts_eq - ts_tax, 0.0)

            cons_burden_total = tax_outcome["consumer_burden_total"]
            prod_burden_total = tax_outcome["producer_burden_total"]

            if revenue_tax > 0:
                cons_burden_pct = 100 * cons_burden_total / revenue_tax
                prod_burden_pct = 100 * prod_burden_total / revenue_tax
            else:
                cons_burden_pct = 0.0
                prod_burden_pct = 0.0

            grafico_impuestos, excedentes_impuestos = st.columns([0.7, 0.3])
            fig_impuestos = go.Figure()

            if demand.kind == "vertical":
                fig_impuestos.add_trace(go.Scatter(
                    x=[demand.qbar, demand.qbar],
                    y=[p_min, p_max],
                    mode="lines",
                    name="Demanda",
                    line=dict(color="blue", width=2),
                ))
            else:
                fig_impuestos.add_trace(go.Scatter(
                    x=q_grid,
                    y=effective_demand_price(demand, q_grid),
                    mode="lines",
                    name="Demanda",
                    line=dict(color="blue", width=2),
                ))

            if supply.kind == "vertical":
                fig_impuestos.add_trace(go.Scatter(
                    x=[supply.qbar, supply.qbar],
                    y=[p_min, p_max],
                    mode="lines",
                    name="Oferta",
                    line=dict(color="red", width=2),
                ))
            else:
                fig_impuestos.add_trace(go.Scatter(
                    x=q_grid,
                    y=effective_supply_price(supply, q_grid),
                    mode="lines",
                    name="Oferta",
                    line=dict(color="red", width=2),
                ))

            if q_tax > 0 and p_c_tax is not None:
                add_consumer_surplus_fill(fig_impuestos, demand, q_tax, p_c_tax, p_max)
            if q_tax > 0 and p_v_tax is not None:
                add_producer_surplus_fill(fig_impuestos, supply, q_tax, p_v_tax, floor_price=0.0)
            add_tax_revenue_fill(fig_impuestos, q_tax, p_v_tax, p_c_tax)
            add_dwl_fill(fig_impuestos, demand, supply, q_tax, q_eq, p_max)

            fig_impuestos.add_trace(go.Scatter(
                x=[q_eq],
                y=[p_eq],
                mode="markers",
                name="Equilibrio competitivo",
                marker=dict(color="black", size=11),
            ))

            if q_tax > 0 and p_c_tax is not None:
                fig_impuestos.add_trace(go.Scatter(
                    x=[q_tax],
                    y=[p_c_tax],
                    mode="markers",
                    name="Precio al consumidor",
                    marker=dict(color="darkgreen", size=12, symbol="diamond"),
                ))
            if q_tax > 0 and p_v_tax is not None:
                fig_impuestos.add_trace(go.Scatter(
                    x=[q_tax],
                    y=[p_v_tax],
                    mode="markers",
                    name="Precio al vendedor",
                    marker=dict(color="green", size=12, symbol="diamond-open"),
                ))

            fig_impuestos.add_vline(x=q_eq, line_dash="dot", line_color="gray")
            fig_impuestos.add_hline(y=p_eq, line_dash="dot", line_color="gray")

            if q_tax > 0:
                fig_impuestos.add_vline(x=q_tax, line_dash="dot", line_color="darkgreen")
            if p_c_tax is not None:
                fig_impuestos.add_hline(
                    y=p_c_tax,
                    line_dash="dash",
                    line_color="darkgreen",
                    annotation_text="Pc",
                    annotation_position="top left"
                )
            if p_v_tax is not None:
                fig_impuestos.add_hline(
                    y=p_v_tax,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Pv",
                    annotation_position="bottom left"
                )

            fig_impuestos.update_layout(
                title="Mercado con impuesto específico",
                xaxis_title="Cantidad",
                yaxis_title="Precio",
                hovermode="closest",
                height=680,
                xaxis=dict(range=[0, q_max]),
                yaxis=dict(range=[p_min, p_max]),
            )

            grafico_impuestos.plotly_chart(fig_impuestos, use_container_width=True)

            with excedentes_impuestos:
                fig_dist_tax = make_tax_distribution_figure(
                    0.0 if np.isinf(cs_tax) else cs_tax,
                    0.0 if np.isinf(ps_tax) else ps_tax,
                    revenue_tax,
                    0.0 if np.isinf(dwl_tax) else dwl_tax,
                    "Distribución del excedente total potencial"
                )

                if fig_dist_tax is None or np.isinf(cs_tax) or np.isinf(ps_tax) or np.isinf(dwl_tax):
                    st.info("No se puede graficar la distribución porcentual cuando algún componente es infinito o no es finito.")
                else:
                    st.plotly_chart(fig_dist_tax, use_container_width=True)

            if p_c_tax is None or p_v_tax is None:
                st.latex(r"P_c=\text{N/D}\qquad P_v=\text{N/D}\qquad t=" + f"{t_tax:.3f}")
            else:
                st.latex(
                    rf"P_c={p_c_tax:.3f}\qquad P_v={p_v_tax:.3f}\qquad t={t_tax:.3f}\qquad P_c-P_v={p_c_tax-p_v_tax:.3f}"
                )

            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.subheader("Resultado con impuesto")
                m1, m2, m3 = st.columns(3)
                m1.metric("Cantidad transada", f"{q_tax:.3f}")
                m2.metric("Precio al consumidor", fmt_metric(p_c_tax))
                m3.metric("Precio al vendedor", fmt_metric(p_v_tax))
            with c2.container(border=True):
                st.subheader(r"Elasticidades en $(P^*;Q^*)$ ")
                e1, e2 = st.columns(2)
                e1.metric("Elasticidad demanda", elasticity_display_demand(demand, q_eq, p_eq))
                e2.metric("Elasticidad oferta", elasticity_display_supply(supply, q_eq, p_eq))
            with st.container(border=True):
                st.subheader("Bienestar y recaudación")
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Excedente del consumidor", fmt_metric(cs_tax))
                b2.metric("Excedente del productor", fmt_metric(ps_tax))
                b3.metric("Recaudación", fmt_metric(revenue_tax))
                b4.metric("Excedente total", fmt_metric(ts_tax))
                b5.metric("DWL", fmt_metric(dwl_tax))

            with st.container(border=True):
                st.subheader("¿Quién soporta la carga del impuesto?")
                i1, i2 = st.columns(2)
                i1.metric(
                    "Carga sobre consumidores",
                    f"{cons_burden_total:.3f}",
                    f"{cons_burden_pct:.1f}% de la recaudación" if revenue_tax > 0 else "0.0%"
                )
                i2.metric(
                    "Carga sobre productores",
                    f"{prod_burden_total:.3f}",
                    f"{prod_burden_pct:.1f}% de la recaudación" if revenue_tax > 0 else "0.0%"
                )

            st.caption("Se suele recomendar gravar al lado más inelástico porque ese lado ajusta menos su cantidad y termina absorbiendo una mayor parte de la carga tributaria.")