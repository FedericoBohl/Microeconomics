from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# Configuración general
# ============================================================
st.set_page_config(
    page_title="Oferta, demanda y bienestar con controles e impuestos",
    layout="wide",
)

TOL = 1e-9


# ============================================================
# Estructuras y utilidades
# ============================================================
@dataclass
class MarketParams:
    p_star: float
    q_star: float
    eps_d: float   # magnitud |epsilon_d| >= 0 o np.inf
    eps_s: float   # epsilon_s >= 0 o np.inf

    @property
    def beta_d(self) -> float:
        # Pendiente en demanda inversa: P_d(Q) = a_d - beta_d Q
        if self.eps_d == 0:
            return np.inf
        if np.isinf(self.eps_d):
            return 0.0
        return self.p_star / (self.eps_d * self.q_star)

    @property
    def beta_s(self) -> float:
        # Pendiente en oferta inversa: P_s(Q) = a_s + beta_s Q
        if self.eps_s == 0:
            return np.inf
        if np.isinf(self.eps_s):
            return 0.0
        return self.p_star / (self.eps_s * self.q_star)

    @property
    def a_d(self) -> float:
        if np.isinf(self.beta_d):
            return np.inf
        return self.p_star + self.beta_d * self.q_star

    @property
    def a_s(self) -> float:
        if np.isinf(self.beta_s):
            return -np.inf
        return self.p_star - self.beta_s * self.q_star


def is_close(x: float, y: float, tol: float = TOL) -> bool:
    return abs(x - y) <= tol


def safe_positive(x: float) -> float:
    return max(float(x), 0.0)


def safe_sum(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if any(np.isinf(v) for v in values):
        return np.inf
    if len(finite) != len(values):
        return np.nan
    return float(sum(finite))


def parse_elasticity(mode: str, finite_value: float) -> float:
    if mode.startswith("0"):
        return 0.0
    if mode.startswith("∞"):
        return np.inf
    return max(float(finite_value), 1e-6)


# ============================================================
# Curvas: precios y cantidades
# ============================================================
def demand_price(q: float, params: MarketParams) -> float:
    """Precio sobre la demanda inversa para una cantidad q."""
    if q <= 0:
        if params.eps_d == 0:
            return np.inf
        if np.isinf(params.eps_d):
            return params.p_star
    if params.eps_d == 0:
        if is_close(q, params.q_star):
            return params.p_star
        return np.inf
    if np.isinf(params.eps_d):
        return params.p_star
    return params.a_d - params.beta_d * q



def supply_price(q: float, params: MarketParams) -> float:
    """Precio sobre la oferta inversa para una cantidad q."""
    if q <= 0:
        if params.eps_s == 0:
            return -np.inf
        if np.isinf(params.eps_s):
            return params.p_star
    if params.eps_s == 0:
        if is_close(q, params.q_star):
            return params.p_star
        return -np.inf
    if np.isinf(params.eps_s):
        return params.p_star
    return params.a_s + params.beta_s * q



def demand_quantity(p: float, params: MarketParams) -> float:
    """
    Cantidad demandada a precio p.

    Devuelve:
    - np.inf si la demanda es perfectamente elástica y p < p*
    - np.nan si la demanda es perfectamente elástica y p = p* (cualquier cantidad)
    """
    if params.eps_d == 0:
        return params.q_star
    if np.isinf(params.eps_d):
        if p < params.p_star - TOL:
            return np.inf
        if is_close(p, params.p_star):
            return np.nan
        return 0.0

    q = (params.a_d - p) / params.beta_d
    return safe_positive(q)



def supply_quantity(p: float, params: MarketParams) -> float:
    """
    Cantidad ofrecida a precio p.

    Devuelve:
    - np.inf si la oferta es perfectamente elástica y p > p*
    - np.nan si la oferta es perfectamente elástica y p = p* (cualquier cantidad)
    """
    if params.eps_s == 0:
        return params.q_star
    if np.isinf(params.eps_s):
        if p > params.p_star + TOL:
            return np.inf
        if is_close(p, params.p_star):
            return np.nan
        return 0.0

    q = (p - params.a_s) / params.beta_s
    return safe_positive(q)


# ============================================================
# Bienestar
# ============================================================
def consumer_surplus(q: float, buyer_price: float, params: MarketParams) -> float:
    if q <= TOL:
        return 0.0

    if params.eps_d == 0:
        # En el límite lineal, el excedente se vuelve no acotado.
        return np.inf

    if np.isinf(params.eps_d):
        return max(params.p_star - buyer_price, 0.0) * q

    area = params.a_d * q - 0.5 * params.beta_d * q * q
    return max(area - buyer_price * q, 0.0)



def producer_surplus(q: float, seller_price: float, params: MarketParams) -> float:
    if q <= TOL:
        return 0.0

    if params.eps_s == 0:
        # En el límite lineal, el excedente se vuelve no acotado.
        return np.inf

    if np.isinf(params.eps_s):
        return max(seller_price - params.p_star, 0.0) * q

    area = params.a_s * q + 0.5 * params.beta_s * q * q
    return max(seller_price * q - area, 0.0)



def deadweight_loss(q_traded: float, params: MarketParams) -> float:
    """
    DWL como pérdida de ganancias de intercambio respecto del equilibrio competitivo base.

    Con curvas lineales calibradas en (Q*, P*):
        DWL = 1/2 * (beta_d + beta_s) * (Q* - Q_t)^2

    Si alguna elasticidad es exactamente 0 y la cantidad cae por debajo de Q*, el costo de bienestar
    deja de ser finito en el límite lineal, por lo que reportamos infinito.
    """
    gap_q = max(params.q_star - q_traded, 0.0)
    if gap_q <= TOL:
        return 0.0

    if params.eps_d == 0 or params.eps_s == 0:
        return np.inf

    return 0.5 * (params.beta_d + params.beta_s) * gap_q * gap_q


# ============================================================
# Resultados de políticas
# ============================================================
def price_control_outcome(p_control: float, params: MarketParams) -> Dict[str, float | str]:
    qd_raw = demand_quantity(p_control, params)
    qs_raw = supply_quantity(p_control, params)

    # Resolver indeterminaciones en el caso perfectamente elástico y precio exactamente igual a P*
    if np.isnan(qd_raw) and np.isnan(qs_raw):
        q_traded = params.q_star
        qd = params.q_star
        qs = params.q_star
    elif np.isnan(qd_raw):
        q_traded = min(np.inf if np.isnan(qd_raw) else qd_raw, qs_raw)
        q_traded = float(qs_raw)
        qd = q_traded
        qs = float(qs_raw)
    elif np.isnan(qs_raw):
        q_traded = float(qd_raw)
        qd = float(qd_raw)
        qs = q_traded
    else:
        q_traded = min(qd_raw, qs_raw)
        qd = qd_raw
        qs = qs_raw

    shortage = np.nan
    excess = np.nan
    if np.isfinite(qd) and np.isfinite(qs):
        shortage = max(qd - qs, 0.0)
        excess = max(qs - qd, 0.0)
    elif np.isinf(qd) and np.isfinite(qs):
        shortage = np.inf
        excess = 0.0
    elif np.isfinite(qd) and np.isinf(qs):
        shortage = 0.0
        excess = np.inf

    cs = consumer_surplus(q_traded, p_control, params)
    ps = producer_surplus(q_traded, p_control, params)
    ts = safe_sum([cs, ps])
    dwl = deadweight_loss(q_traded, params)

    if p_control < params.p_star - TOL:
        policy = "Precio máximo"
    elif p_control > params.p_star + TOL:
        policy = "Precio mínimo"
    else:
        policy = "Sin distorsión"

    return {
        "Política": policy,
        "Precio regulado": p_control,
        "Q_d": qd,
        "Q_s": qs,
        "Q_t": q_traded,
        "Escasez": shortage,
        "Exceso de oferta": excess,
        "Precio comprador": p_control,
        "Precio vendedor": p_control,
        "CS": cs,
        "PS": ps,
        "Recaudación": 0.0,
        "TS": ts,
        "DWL": dwl,
    }



def tax_outcome(tax: float, params: MarketParams) -> Dict[str, float | str]:
    # Caso 1: ambas curvas no verticales
    if params.eps_d != 0 and params.eps_s != 0:
        denom = params.beta_d + params.beta_s

        if denom <= TOL:
            # Oferta y demanda perfectamente elásticas en el mismo precio.
            if tax <= TOL:
                q_traded = params.q_star
                p_c = params.p_star
                p_p = params.p_star
            else:
                q_traded = 0.0
                p_c = np.nan
                p_p = np.nan
        else:
            q_traded = max(params.q_star - tax / denom, 0.0)
            p_c = demand_price(q_traded, params)
            p_p = p_c - tax

    # Caso 2: demanda perfectamente inelástica
    elif params.eps_d == 0 and params.eps_s != 0:
        q_traded = params.q_star
        p_p = supply_price(q_traded, params)
        p_c = p_p + tax

    # Caso 3: oferta perfectamente inelástica
    elif params.eps_s == 0 and params.eps_d != 0:
        q_traded = params.q_star
        p_c = demand_price(q_traded, params)
        p_p = p_c - tax

    # Caso 4: ambas perfectamente inelásticas
    else:
        q_traded = params.q_star
        p_p = params.p_star
        p_c = params.p_star + tax

    cs = consumer_surplus(q_traded, p_c, params) if np.isfinite(q_traded) and np.isfinite(p_c) else 0.0
    ps = producer_surplus(q_traded, p_p, params) if np.isfinite(q_traded) and np.isfinite(p_p) else 0.0
    revenue = tax * q_traded
    ts = safe_sum([cs, ps, revenue])
    dwl = deadweight_loss(q_traded, params)

    return {
        "Impuesto": tax,
        "Q_t": q_traded,
        "Precio consumidor": p_c,
        "Precio productor": p_p,
        "CS": cs,
        "PS": ps,
        "Recaudación": revenue,
        "TS": ts,
        "DWL": dwl,
    }


# ============================================================
# Formato para tablas y texto
# ============================================================
def fmt_value(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "N/D"
    if np.isposinf(x):
        return "∞"
    if np.isneginf(x):
        return "-∞"
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")



def display_df(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: fmt_value(x, digits=digits))
    return out



def elasticidad_texto(eps: float, lado: str) -> str:
    if eps == 0:
        return f"{lado}: 0 (perfectamente inelástica)"
    if np.isinf(eps):
        return f"{lado}: ∞ (perfectamente elástica)"
    return f"{lado}: {eps:.3f}"


# ============================================================
# Gráficos
# ============================================================
def add_series(fig, df, x_col: str, y_col: str, name: str, secondary_y: bool = False):
    series = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=series[x_col],
            y=series[y_col],
            mode="lines",
            name=name,
        ),
        secondary_y=secondary_y,
    )



def build_control_chart(df: pd.DataFrame, params: MarketParams) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Eje monetario
    add_series(fig, df, "Precio regulado", "CS", "CS")
    add_series(fig, df, "Precio regulado", "PS", "PS")
    add_series(fig, df, "Precio regulado", "TS", "TS")
    add_series(fig, df, "Precio regulado", "DWL", "DWL")

    # Eje real
    add_series(fig, df, "Precio regulado", "Q_t", "Cantidad transada", secondary_y=True)
    add_series(fig, df, "Precio regulado", "Escasez", "Escasez", secondary_y=True)
    add_series(fig, df, "Precio regulado", "Exceso de oferta", "Exceso de oferta", secondary_y=True)

    fig.add_vline(x=params.p_star, line_dash="dash", annotation_text="P*", annotation_position="top")
    fig.update_layout(
        title="Controles de precios: bienestar y cantidad según el precio regulado",
        xaxis_title="Precio regulado",
        yaxis_title="Montos",
        yaxis2_title="Cantidades",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig



def build_tax_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Eje monetario
    add_series(fig, df, "Impuesto", "CS", "CS")
    add_series(fig, df, "Impuesto", "PS", "PS")
    add_series(fig, df, "Impuesto", "Recaudación", "Recaudación")
    add_series(fig, df, "Impuesto", "TS", "TS")
    add_series(fig, df, "Impuesto", "DWL", "DWL")

    # Eje real / precios
    add_series(fig, df, "Impuesto", "Q_t", "Cantidad transada", secondary_y=True)
    add_series(fig, df, "Impuesto", "Precio consumidor", "Precio consumidor", secondary_y=True)
    add_series(fig, df, "Impuesto", "Precio productor", "Precio productor", secondary_y=True)

    fig.add_vline(x=0.0, line_dash="dash", annotation_text="t = 0", annotation_position="top")
    fig.update_layout(
        title="Impuesto específico: bienestar, precios y cantidad según t",
        xaxis_title="Impuesto específico t",
        yaxis_title="Montos",
        yaxis2_title="Cantidades / precios",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("Parámetros")

st.sidebar.subheader("Equilibrio competitivo base")
p_star = st.sidebar.number_input("Precio competitivo base P*", min_value=0.01, value=100.0, step=1.0)
q_star = st.sidebar.number_input("Cantidad competitiva base Q*", min_value=0.01, value=100.0, step=1.0)

st.sidebar.subheader("Elasticidades")
mode_d = st.sidebar.selectbox(
    "Elasticidad-precio de la demanda |ε_d|",
    ["Finita", "0 (perfectamente inelástica)", "∞ (perfectamente elástica)"],
    index=0,
)
value_d = 1.50
if mode_d == "Finita":
    value_d = st.sidebar.number_input("Valor finito de |ε_d|", min_value=0.0001, value=1.50, step=0.10, format="%.4f")

mode_s = st.sidebar.selectbox(
    "Elasticidad-precio de la oferta ε_s",
    ["Finita", "0 (perfectamente inelástica)", "∞ (perfectamente elástica)"],
    index=0,
)
value_s = 1.00
if mode_s == "Finita":
    value_s = st.sidebar.number_input("Valor finito de ε_s", min_value=0.0001, value=1.00, step=0.10, format="%.4f")

eps_d = parse_elasticity(mode_d, value_d)
eps_s = parse_elasticity(mode_s, value_s)
params = MarketParams(p_star=float(p_star), q_star=float(q_star), eps_d=eps_d, eps_s=eps_s)

st.sidebar.subheader("Barrido de precios máximos y mínimos")
control_min_factor = st.sidebar.slider("Mínimo como fracción de P*", min_value=0.0, max_value=0.99, value=0.50, step=0.01)
control_max_factor = st.sidebar.slider("Máximo como fracción de P*", min_value=1.01, max_value=3.00, value=1.50, step=0.01)
n_controls = st.sidebar.slider("Cantidad de niveles de precio regulado", min_value=9, max_value=61, value=21, step=2)

st.sidebar.subheader("Barrido de impuestos")
tax_max_factor = st.sidebar.slider("Impuesto máximo como fracción de P*", min_value=0.0, max_value=2.00, value=0.80, step=0.01)
n_taxes = st.sidebar.slider("Cantidad de niveles de impuesto", min_value=9, max_value=61, value=21, step=2)

show_notes = st.sidebar.checkbox("Mostrar notas metodológicas", value=True)


# ============================================================
# Texto principal
# ============================================================
st.title("Oferta, demanda y bienestar con controles de precios e impuestos")

st.markdown(
    r"""
Esta app parte de un equilibrio competitivo base \((P^*,Q^*)\) y reconstruye una oferta y una demanda
lineales en **forma inversa** para que la elasticidad de cada curva en ese punto coincida con la que se vio en clase.
A partir de ahí, compara bienestar bajo **precio máximo**, **precio mínimo** e **impuesto específico**.
"""
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("P*", fmt_value(params.p_star))
col_b.metric("Q*", fmt_value(params.q_star))
col_c.metric("|ε_d|", "∞" if np.isinf(params.eps_d) else fmt_value(params.eps_d, 3))
col_d.metric("ε_s", "∞" if np.isinf(params.eps_s) else fmt_value(params.eps_s, 3))

with st.expander("Ecuaciones y relación con las definiciones de clase", expanded=True):
    st.markdown("**1. Curvas inversas calibradas en el equilibrio competitivo base**")
    st.latex(r"P_d(Q)=a_d-b_dQ")
    st.latex(r"P_s(Q)=a_s+b_sQ")
    st.markdown("donde, para elasticidades finitas,")
    st.latex(r"b_d=\frac{P^*}{|\varepsilon_d|Q^*}")
    st.latex(r"b_s=\frac{P^*}{\varepsilon_sQ^*}")
    st.latex(r"a_d=P^*+b_dQ^*")
    st.latex(r"a_s=P^*-b_sQ^*")

    st.markdown("**2. Relación con la elasticidad-precio en el equilibrio**")
    st.latex(r"|\varepsilon_d|=-\frac{dQ_d}{dP}\frac{P^*}{Q^*}")
    st.latex(r"\varepsilon_s=\frac{dQ_s}{dP}\frac{P^*}{Q^*}")
    st.markdown(
        """
- Si la elasticidad es **0**, la curva es perfectamente inelástica (vertical)
- Si la elasticidad es **∞**, la curva es perfectamente elástica (horizontal)
"""
    )

    st.markdown("**3. Controles de precios**")
    st.latex(r"Q_t(\bar P)=\min\{Q_d(\bar P),Q_s(\bar P)\}")
    st.latex(r"\bar P<P^* \Rightarrow \text{precio máximo}")
    st.latex(r"\bar P>P^* \Rightarrow \text{precio mínimo}")
    st.latex(r"\text{Escasez}=Q_d(\bar P)-Q_s(\bar P)")
    st.latex(r"\text{Exceso de oferta}=Q_s(\bar P)-Q_d(\bar P)")

    st.markdown("**4. Impuesto específico**")
    st.latex(r"P_c=P_p+t")
    st.latex(r"P_d(Q_t)=P_s(Q_t)+t")

    st.markdown("**5. Excedentes y pérdida de eficiencia**")
    st.latex(r"CS=\int_0^{Q_t}P_d(Q)dQ-P_cQ_t")
    st.latex(r"PS=P_pQ_t-\int_0^{Q_t}P_s(Q)dQ")
    st.latex(r"GR=tQ_t")
    st.latex(r"TS=CS+PS+GR")
    st.latex(r"DWL=\int_{Q_t}^{Q^*}\left[P_d(Q)-P_s(Q)\right]dQ")

    st.markdown(
        r"""
En el caso lineal calibrado en \((P^*,Q^*)\), cuando las elasticidades son finitas o infinitas,
la DWL también puede escribirse como
"""
    )
    st.latex(r"DWL=\frac{1}{2}(b_d+b_s)(Q^*-Q_t)^2")

    st.info(
        "Con elasticidad exactamente 0 en alguno de los lados, algunos excedentes dejan de ser finitos en el límite lineal. "
        "La app lo marca como ∞ o N/D en vez de forzar un número artificial."
    )

if show_notes:
    with st.expander("Notas metodológicas", expanded=False):
        st.markdown(
            fr"""
- {elasticidad_texto(params.eps_d, 'Demanda')}
- {elasticidad_texto(params.eps_s, 'Oferta')}
- La app usa un **equilibrio base fijo** \((P^*,Q^*)\). Al cambiar elasticidades, las curvas rotan alrededor de ese punto.
- Esto ayuda a que los alumnos vean cómo cambian la incidencia, la escasez, el exceso de oferta y la DWL
  **sin mover el benchmark competitivo**.
"""
        )


# ============================================================
# Cálculos
# ============================================================
control_prices = np.linspace(control_min_factor * params.p_star, control_max_factor * params.p_star, n_controls)
control_rows = [price_control_outcome(float(p), params) for p in control_prices]
control_df = pd.DataFrame(control_rows)

# Asegurar que P* esté en la tabla para la lectura del equilibrio
if not np.any(np.isclose(control_df["Precio regulado"].values, params.p_star, atol=1e-8)):
    control_df = pd.concat([control_df, pd.DataFrame([price_control_outcome(params.p_star, params)])], ignore_index=True)
    control_df = control_df.sort_values("Precio regulado").reset_index(drop=True)

tax_values = np.linspace(0.0, tax_max_factor * params.p_star, n_taxes)
tax_rows = [tax_outcome(float(t), params) for t in tax_values]
tax_df = pd.DataFrame(tax_rows)

# Fila de equilibrio competitivo de referencia para mostrar arriba
base_control = price_control_outcome(params.p_star, params)
base_tax = tax_outcome(0.0, params)


# ============================================================
# Gráficos en dos columnas
# ============================================================
left, right = st.columns(2)

with left:
    st.plotly_chart(build_control_chart(control_df, params), use_container_width=True)
    st.markdown(
        f"**Equilibrio base en el gráfico:** P* = {fmt_value(base_control['Precio regulado'])}, "
        f"Q* = {fmt_value(base_control['Q_t'])}"
    )

with right:
    st.plotly_chart(build_tax_chart(tax_df), use_container_width=True)
    st.markdown(
        f"**Equilibrio base en el gráfico:** t = 0, "
        f"P_c = {fmt_value(base_tax['Precio consumidor'])}, "
        f"P_p = {fmt_value(base_tax['Precio productor'])}, "
        f"Q* = {fmt_value(base_tax['Q_t'])}"
    )


# ============================================================
# Reportes
# ============================================================
st.subheader("Reporte resumido")

rep1, rep2 = st.columns(2)
with rep1:
    st.markdown("**Precio máximo / mínimo**")
    control_display_cols = [
        "Política", "Precio regulado", "Q_d", "Q_s", "Q_t", "Escasez", "Exceso de oferta",
        "CS", "PS", "TS", "DWL"
    ]
    st.dataframe(display_df(control_df[control_display_cols]), use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar reporte de precios (CSV)",
        control_df.to_csv(index=False).encode("utf-8"),
        file_name="reporte_precios_maximos_minimos.csv",
        mime="text/csv",
        use_container_width=True,
    )

with rep2:
    st.markdown("**Impuesto específico**")
    tax_display_cols = [
        "Impuesto", "Precio consumidor", "Precio productor", "Q_t",
        "CS", "PS", "Recaudación", "TS", "DWL"
    ]
    st.dataframe(display_df(tax_df[tax_display_cols]), use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar reporte de impuestos (CSV)",
        tax_df.to_csv(index=False).encode("utf-8"),
        file_name="reporte_impuestos.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Lectura económica automática
# ============================================================
st.subheader("Lectura económica rápida")

# Incidencia del impuesto a partir del primer tax positivo con datos finitos
finite_tax = tax_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Impuesto", "Precio consumidor", "Precio productor"])
incidence_text = "No pudo calcularse con los parámetros elegidos."
if len(finite_tax) >= 2:
    ref = finite_tax.iloc[0]
    last = finite_tax.iloc[-1]
    total_tax = last["Impuesto"] - ref["Impuesto"]
    if total_tax > TOL:
        burden_consumers = (last["Precio consumidor"] - ref["Precio consumidor"]) / total_tax
        burden_producers = (ref["Precio productor"] - last["Precio productor"]) / total_tax
        incidence_text = (
            f"A lo largo del barrido de impuestos, aproximadamente {100*burden_consumers:.1f}% de la carga "
            f"recae sobre consumidores y {100*burden_producers:.1f}% sobre productores."
        )

notes = []
if params.eps_d == 0 or params.eps_s == 0:
    notes.append(
        "Con una elasticidad exactamente 0, algunos excedentes pasan a ser no finitos en el límite lineal. "
        "Eso no es un error numérico: es una propiedad del caso extremo."
    )
if np.isinf(params.eps_d) or np.isinf(params.eps_s):
    notes.append(
        "Con una elasticidad infinita, esa curva es horizontal: el precio queda fijado y todo el ajuste recae del otro lado o en la cantidad."
    )
if not notes:
    notes.append(
        "Con elasticidades finitas, la app entrega excedentes, recaudación y DWL completamente comparables en todo el barrido."
    )

st.markdown(f"- {incidence_text}")
for note in notes:
    st.markdown(f"- {note}")

st.divider()
st.caption(
    "Sugerencia didáctica: pedile a tus alumnos que mantengan fijo (P*, Q*) y comparen qué pasa con la incidencia, "
    "la escasez, el exceso de oferta y la DWL cuando una de las curvas se vuelve más inelástica."
)
