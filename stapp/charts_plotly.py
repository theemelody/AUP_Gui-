"""
charts_plotly.py — Plotly versions of all SECAP charts.

Each function accepts pre-loaded DataFrames and returns a plotly.graph_objects.Figure.
Call st.plotly_chart(fig, use_container_width=True) in app.py.

Data conventions (same as charts.ipynb):
  df    — annual per-building demand  (Total_demand.csv, indexed by building name)
  dfh   — hourly district totals      (Total_demand_hourly.csv, datetime index)
  EF    — emission factor dict        {carrier: tCO2/MWh}
  PALETTE — optional colour overrides
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Default emission factors (override by passing EF kwarg) ─────────────────
_DEFAULT_EF = {
    "NG":    0.202,
    "COAL":  0.341,
    "OIL":   0.266,
    "WOOD":  0.017,
    "DH":    0.070,
    "GRID":  0.366,
    "SOLAR": 0.000,
    "PV":    0.000,
}

_COLORS = {
    "Grid":          "#4472C4",
    "Natural Gas":   "#FF7043",
    "District Heat": "#42A5F5",
    "Solar Thermal": "#FDD835",
    "Oil":           "#8D6E63",
    "Coal":          "#616161",
    "Wood/Biomass":  "#66BB6A",
    "Space Heating": "#EF5350",
    "DHW":           "#FF8A65",
    "Cooling":       "#29B6F6",
    "Appliances":    "#AB47BC",
    "Lighting":      "#FFEE58",
}

_MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Chart 1 — Energy Mix & CO₂ Baseline ─────────────────────────────────────

def chart_energy_mix(df: pd.DataFrame, EF: dict = None) -> go.Figure:
    """Bar chart of energy by carrier + CO₂ secondary axis. [§3 BEI]"""
    ef = EF or _DEFAULT_EF

    carriers = {
        "Grid":          df["GRID_MWhyr"].sum(),
        "Natural Gas":  (df["NG_hs_MWhyr"]   + df["NG_ww_MWhyr"]).sum(),
        "District Heat":(df["DH_hs_MWhyr"]   + df["DH_ww_MWhyr"]).sum(),
        "Solar Thermal":(df["SOLAR_hs_MWhyr"] + df["SOLAR_ww_MWhyr"]).sum(),
        "Oil":          (df["OIL_hs_MWhyr"]  + df["OIL_ww_MWhyr"]).sum(),
        "Coal":         (df["COAL_hs_MWhyr"] + df["COAL_ww_MWhyr"]).sum(),
        "Wood/Biomass": (df["WOOD_hs_MWhyr"] + df["WOOD_ww_MWhyr"]).sum(),
    }
    ef_map = {
        "Grid": ef["GRID"], "Natural Gas": ef["NG"], "District Heat": ef["DH"],
        "Solar Thermal": 0.0, "Oil": ef["OIL"], "Coal": ef["COAL"], "Wood/Biomass": ef["WOOD"],
    }
    s = pd.Series(carriers)
    s = s[s > 0]
    co2 = s * pd.Series(ef_map).reindex(s.index).fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=s.index, y=s.values,
        marker_color=[_COLORS.get(k, "#9E9E9E") for k in s.index],
        name="Final Energy (MWh/yr)",
        text=[f"{v:,.0f}" for v in s.values], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=co2.index, y=co2.values, mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#B71C1C"),
        name="CO₂ (tCO₂/yr)",
        text=[f"{v:,.0f} t" for v in co2.values], textposition="top center",
        textfont=dict(color="#B71C1C", size=11),
    ), secondary_y=True)

    fig.update_layout(
        title="District Energy Mix & CO₂ Baseline  [§3 BEI]",
        yaxis_title="Final Energy (MWh/yr)",
        yaxis2_title="Estimated CO₂ Emissions (tCO₂/yr)",
        yaxis2=dict(tickfont=dict(color="#B71C1C"), title_font=dict(color="#B71C1C")),
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white", height=420,
    )
    return fig


# ── Chart A — End-Use Breakdown ──────────────────────────────────────────────

def chart_end_use_breakdown(df: pd.DataFrame) -> go.Figure:
    """Donut + horizontal bar — delivered energy by end-use. [§3 BEI]"""
    end_uses = pd.Series({
        "Space Heating": df["Qhs_sys_MWhyr"].sum(),
        "DHW":           df["Qww_sys_MWhyr"].sum(),
        "Cooling":       df["QC_sys_MWhyr"].sum(),
        "Appliances":    df["Ea_MWhyr"].sum(),
        "Lighting":      df["El_MWhyr"].sum(),
    })
    end_uses = end_uses[end_uses > 0]
    colors = [_COLORS.get(k, "#9E9E9E") for k in end_uses.index]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=["Share of Delivered Energy", "Absolute Demand by End-Use"],
    )
    fig.add_trace(go.Pie(
        labels=end_uses.index, values=end_uses.values,
        hole=0.55, marker_colors=colors,
        textinfo="label+percent", hovertemplate="%{label}: %{value:,.1f} MWh/yr",
    ), row=1, col=1)
    eu_s = end_uses.sort_values()
    fig.add_trace(go.Bar(
        x=eu_s.values, y=eu_s.index, orientation="h",
        marker_color=[_COLORS.get(k, "#9E9E9E") for k in eu_s.index],
        text=[f"{v:,.1f}" for v in eu_s.values], textposition="outside",
        hovertemplate="%{y}: %{x:,.1f} MWh/yr",
    ), row=1, col=2)

    fig.update_layout(
        title="End-Use Breakdown — Delivered Energy  [§3 BEI]",
        showlegend=False, template="plotly_white", height=420,
    )
    fig.update_xaxes(title_text="MWh/yr", row=1, col=2)
    return fig


# ── Chart B — Emissions vs 2030 Target Gauge ────────────────────────────────

def chart_target_gauge(total_co2: float, eui_now: float) -> go.Figure:
    """Two bullet/indicator gauges: CO₂ and EUI vs 2030 target. [§2.2]"""
    target_co2 = total_co2 * 0.60
    target_eui = eui_now * 0.60

    # Each indicator gets its own domain so titles don't overlap the gauge arc
    gauge_domain_left  = {"x": [0.02, 0.46], "y": [0.05, 0.95]}
    gauge_domain_right = {"x": [0.54, 0.98], "y": [0.05, 0.95]}

    kw = dict(
        mode="gauge+number+delta",
        delta={"reference": 0, "valueformat": ",.0f"},
    )
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=total_co2,
        title={"text": "Total CO₂ Baseline<br><span style='font-size:0.85em;color:gray'>tCO₂/yr</span>",
               "font": {"size": 16}},
        domain=gauge_domain_left,
        gauge={
            "axis": {"range": [0, total_co2 * 1.15]},
            "bar": {"color": "#EF5350"},
            "threshold": {"line": {"color": "#1565C0", "width": 4}, "value": target_co2},
            "steps": [{"range": [0, target_co2], "color": "#C8E6C9"},
                      {"range": [target_co2, total_co2 * 1.15], "color": "#FFCDD2"}],
        },
        **kw,
    ))
    fig.add_trace(go.Indicator(
        value=eui_now,
        title={"text": "District EUI<br><span style='font-size:0.85em;color:gray'>kWh/m²/yr</span>",
               "font": {"size": 16}},
        domain=gauge_domain_right,
        gauge={
            "axis": {"range": [0, eui_now * 1.15]},
            "bar": {"color": "#EF5350"},
            "threshold": {"line": {"color": "#1565C0", "width": 4}, "value": target_eui},
            "steps": [{"range": [0, target_eui], "color": "#C8E6C9"},
                      {"range": [target_eui, eui_now * 1.15], "color": "#FFCDD2"}],
        },
        **kw,
    ))

    fig.update_layout(
        title=dict(
            text="Baseline vs SECAP 2030 Target (−40%)  [§2.2 Commitments]",
            y=0.97, x=0.5, xanchor="center", yanchor="top",
        ),
        template="plotly_white",
        height=420,
        margin=dict(t=60, b=50, l=20, r=20),
        annotations=[
            dict(text=f"Target: {target_co2:,.0f}", x=0.24, y=0.03, showarrow=False,
                 xref="paper", yref="paper", font=dict(color="#1565C0", size=11)),
            dict(text=f"Target: {target_eui:,.0f}", x=0.76, y=0.03, showarrow=False,
                 xref="paper", yref="paper", font=dict(color="#1565C0", size=11)),
        ],
    )
    return fig


# ── Chart C — Building Sector Mix ────────────────────────────────────────────

def chart_sector_mix(df: pd.DataFrame) -> go.Figure | None:
    """3-panel: GFA pie, energy pie, EUI bar — by building use type. [§3 BEI]"""
    if "1ST_USE" not in df.columns or df["1ST_USE"].isna().all():
        return None

    sector_cols = ["GFA_m2", "QH_sys_MWhyr", "QC_sys_MWhyr", "GRID_MWhyr"]
    by_sector = df.groupby("1ST_USE")[sector_cols].sum()
    by_sector["Total_MWh"] = by_sector[["QH_sys_MWhyr", "QC_sys_MWhyr", "GRID_MWhyr"]].sum(axis=1)
    by_sector["EUI"] = by_sector["Total_MWh"] * 1000 / by_sector["GFA_m2"]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "xy"}]],
        subplot_titles=["GFA by Sector (m²)", "Total Energy by Sector (MWh/yr)", "EUI by Sector (kWh/m²/yr)"],
    )
    pie_kw = dict(textinfo="label+percent", hovertemplate="%{label}: %{value:,.0f}")
    fig.add_trace(go.Pie(labels=by_sector.index, values=by_sector["GFA_m2"], hole=0.4, **pie_kw), row=1, col=1)
    fig.add_trace(go.Pie(labels=by_sector.index, values=by_sector["Total_MWh"], hole=0.4, **pie_kw), row=1, col=2)
    fig.add_trace(go.Bar(
        x=by_sector.index, y=by_sector["EUI"],
        marker_color="#5C6BC0",
        text=[f"{v:.0f}" for v in by_sector["EUI"]], textposition="outside",
    ), row=1, col=3)

    fig.update_layout(
        title="Building Sector Analysis  [§3 BEI — residential / tertiary]",
        showlegend=False, template="plotly_white", height=420,
    )
    fig.update_yaxes(title_text="kWh/m²/yr", row=1, col=3)
    return fig


# ── Chart 2 — Load Duration Curves ───────────────────────────────────────────

def chart_load_duration(dfh: pd.DataFrame) -> go.Figure:
    """3 LDCs: heating, cooling, grid electricity. [§5 Mitigation]"""
    specs = [
        ("Qhs_kWh",  "Space Heating",   "#EF5350"),
        ("Qcs_kWh",  "Cooling",         "#29B6F6"),
        ("GRID_kWh", "Grid Electricity","#4472C4"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[s[1] for s in specs])

    for col_idx, (col, title, color) in enumerate(specs, start=1):
        if col not in dfh.columns or dfh[col].fillna(0).max() == 0:
            fig.add_annotation(
                text=f"No {title.lower()} demand<br>in this scenario",
                xref=f"x{col_idx} domain" if col_idx > 1 else "x domain",
                yref=f"y{col_idx} domain" if col_idx > 1 else "y domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#9E9E9E", size=12),
                row=1, col=col_idx,
            )
            continue
        vals = np.sort(dfh[col].fillna(0).values)[::-1]
        hours = np.arange(1, len(vals) + 1)
        p95 = float(np.percentile(vals[vals > 0], 95)) if (vals > 0).any() else 0

        fig.add_trace(go.Scatter(
            x=hours, y=vals, mode="lines",
            line=dict(color=color, width=1.5), fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            name=title, hovertemplate="Hour %{x}: %{y:,.0f} kWh/h",
        ), row=1, col=col_idx)
        fig.add_hline(y=p95, line_dash="dash", line_color="grey", line_width=1,
                      annotation_text=f"P95: {p95:,.0f}", annotation_position="top right",
                      row=1, col=col_idx)

    fig.update_layout(
        title="Load Duration Curves — District Peak Demand  [§5 Mitigation]",
        showlegend=False, template="plotly_white", height=400,
    )
    for i in range(1, 4):
        fig.update_xaxes(title_text="Hours ranked (1 = peak)", row=1, col=i)
        fig.update_yaxes(title_text="kWh/h", row=1, col=i)
    return fig


# ── Chart 3 — Monthly Demand Profile ─────────────────────────────────────────

def chart_monthly_demand(dfh: pd.DataFrame) -> go.Figure:
    """Stacked bar of monthly heating / cooling / electricity. [§4 RVA]"""
    heat_col = "QH_sys_kWh" if "QH_sys_kWh" in dfh.columns else "Qhs_kWh"
    cool_col = "QC_sys_kWh" if "QC_sys_kWh" in dfh.columns else "Qcs_kWh"

    monthly = dfh[[heat_col, cool_col, "GRID_kWh"]].resample("ME").sum() / 1e3
    monthly.columns = ["Heating", "Cooling", "Electricity"]

    fig = go.Figure()
    palette = {"Heating": "#EF5350", "Cooling": "#29B6F6", "Electricity": "#4472C4"}
    for col in ["Heating", "Cooling", "Electricity"]:
        fig.add_trace(go.Bar(
            x=_MONTH_LABELS, y=monthly[col].values,
            name=col, marker_color=palette[col],
            hovertemplate=f"{col}: %{{y:,.1f}} MWh<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Monthly Energy Demand Profile — District Total  [§4 RVA]",
        yaxis_title="Energy (MWh/month)",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white", height=400,
    )
    return fig


# ── Chart 4 — Heat Stress Heatmap ────────────────────────────────────────────

def chart_heat_stress(dfh: pd.DataFrame) -> go.Figure | None:
    """Hour×Month heatmap of cooling demand + outdoor temp. [§4 RVA + §6]"""
    cooling_candidates = [c for c in ["QC_sys_kWh", "Qcs_kWh"] if c in dfh.columns]
    cool_col = next((c for c in cooling_candidates if dfh[c].sum() > 0), None)
    has_tout = "theta_o_C" in dfh.columns

    if not cool_col and not has_tout:
        return None

    dfh_h = dfh.copy()
    dfh_h["month"] = dfh_h.index.month
    dfh_h["hour"] = dfh_h.index.hour

    n_panels = (1 if cool_col else 0) + (1 if has_tout else 0)
    titles = []
    if cool_col:
        titles.append("Cooling Demand: Hour × Month (mean kWh/h)")
    if has_tout:
        titles.append("Mean Operative Temperature: Hour × Month (mean °C)")

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles)
    col_idx = 1

    if cool_col:
        pivot = dfh_h.pivot_table(values=cool_col, index="hour", columns="month", aggfunc="mean")
        pivot.columns = _MONTH_LABELS
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=_MONTH_LABELS, y=list(range(24)),
            colorscale="YlOrRd", colorbar=dict(title="kWh/h", x=0.45 if n_panels == 2 else 1.0),
            hovertemplate="Month: %{x}<br>Hour: %{y}:00<br>%{z:.1f} kWh/h<extra></extra>",
        ), row=1, col=col_idx)
        col_idx += 1

    if has_tout:
        pivot_t = dfh_h.pivot_table(values="theta_o_C", index="hour", columns="month", aggfunc="mean")
        pivot_t.columns = _MONTH_LABELS
        fig.add_trace(go.Heatmap(
            z=pivot_t.values, x=_MONTH_LABELS, y=list(range(24)),
            colorscale="RdBu_r", colorbar=dict(title="°C"),
            hovertemplate="Month: %{x}<br>Hour: %{y}:00<br>%{z:.1f} °C<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Heat Stress Risk  [§4 RVA — heatwave + §6 Adaptation]",
        template="plotly_white", height=450,
    )
    for i in range(1, n_panels + 1):
        fig.update_yaxes(title_text="Hour of Day", row=1, col=i)
    return fig


# ── Chart 5 — EUI Distribution & Outlier Flagging ────────────────────────────

def chart_eui_distribution(df: pd.DataFrame, threshold_eui: float = 200) -> go.Figure:
    """Histogram of EUI with NZEB threshold and annotated outliers. [§3 + §5]"""
    df_eui = df.copy()
    df_eui["total_final_MWh"] = (
        df["GRID_MWhyr"]
        + df[["NG_hs_MWhyr", "NG_ww_MWhyr", "COAL_hs_MWhyr", "COAL_ww_MWhyr",
              "OIL_hs_MWhyr", "OIL_ww_MWhyr", "WOOD_hs_MWhyr", "WOOD_ww_MWhyr",
              "DH_hs_MWhyr", "DH_ww_MWhyr", "SOLAR_hs_MWhyr", "SOLAR_ww_MWhyr"]]
             .sum(axis=1)
    )
    df_eui["EUI"] = df_eui["total_final_MWh"] * 1000 / df_eui["GFA_m2"]
    outliers = df_eui[df_eui["EUI"] > threshold_eui]
    mean_eui = df_eui["EUI"].mean()

    n_bins = max(5, min(int(np.ceil(np.log2(len(df_eui)) + 1)), 30))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_eui["EUI"], nbinsx=n_bins,
        marker_color="#5C6BC0", opacity=0.75, name="Buildings",
        hovertemplate="EUI: %{x:.0f} kWh/m²/yr<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=threshold_eui, line_dash="dash", line_color="#EF5350",
                  annotation_text=f"NZEB threshold ({threshold_eui} kWh/m²/yr)",
                  annotation_position="top right")
    fig.add_vline(x=mean_eui, line_dash="dot", line_color="#FFA726",
                  annotation_text=f"Mean ({mean_eui:.0f} kWh/m²/yr)",
                  annotation_position="top left")

    # Annotate outliers
    for bname, row in outliers.iterrows():
        fig.add_annotation(
            x=row["EUI"], y=0,
            text=f"⚑ {bname}<br>({row['EUI']:.0f})",
            showarrow=True, arrowhead=2, arrowcolor="#EF5350",
            font=dict(size=9, color="#EF5350"), ax=0, ay=-50,
        )

    fig.update_layout(
        title="EUI Distribution — Building Stock Overview  [§3 BEI + §5 Targeting]",
        xaxis_title="Energy Use Intensity (kWh/m²/yr)",
        yaxis_title="Building Count",
        template="plotly_white", height=400,
        showlegend=False,
    )
    return fig


# ── Chart 6 — Solar PV Potential vs Demand ───────────────────────────────────

def chart_solar_vs_demand(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: total demand vs PV potential per building. [§5 Mitigation]"""
    # Compute total final energy (reuse EUI logic)
    df2 = df.copy()
    df2["total_final_MWh"] = (
        df["GRID_MWhyr"]
        + df[["NG_hs_MWhyr", "NG_ww_MWhyr", "COAL_hs_MWhyr", "COAL_ww_MWhyr",
              "OIL_hs_MWhyr", "OIL_ww_MWhyr", "WOOD_hs_MWhyr", "WOOD_ww_MWhyr",
              "DH_hs_MWhyr", "DH_ww_MWhyr", "SOLAR_hs_MWhyr", "SOLAR_ww_MWhyr"]]
             .sum(axis=1)
    )
    df2["gap"] = df2["total_final_MWh"] - df2["PV_MWhyr"]
    df2["coverage_pct"] = (df2["PV_MWhyr"] / df2["total_final_MWh"] * 100).clip(0, 100)
    df2 = df2.sort_values("gap", ascending=False)

    if df2["PV_MWhyr"].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠ PV_MWhyr = 0 for all buildings.<br>Run <b>cea photovoltaic</b> to generate this chart.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#9E9E9E"),
        )
        fig.update_layout(title="Solar PV Potential vs Total Demand  [§5 Mitigation]",
                          template="plotly_white", height=300)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df2.index, y=df2["total_final_MWh"],
        name="Total Demand (MWh/yr)", marker_color="#EF5350",
        hovertemplate="%{x}<br>Demand: %{y:,.1f} MWh/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df2.index, y=df2["PV_MWhyr"],
        name="PV Potential (MWh/yr)", marker_color="#FDD835",
        hovertemplate="%{x}<br>PV: %{y:,.1f} MWh/yr<br>Coverage: %{customdata:.0f}%<extra></extra>",
        customdata=df2["coverage_pct"],
    ))

    # Coverage % annotations above each pair
    for bname, row in df2.iterrows():
        fig.add_annotation(
            x=bname, y=max(row["total_final_MWh"], row["PV_MWhyr"]) * 1.05,
            text=f"{row['coverage_pct']:.0f}%",
            showarrow=False, font=dict(size=10, color="#5C6BC0"),
        )

    fig.update_layout(
        barmode="group",
        title="Solar PV Potential vs Total Demand  [§5 Mitigation — solar action]",
        yaxis_title="MWh/yr",
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white", height=420,
    )
    return fig


# ── Chart 7 — District Energy Loop Opportunity ───────────────────────────────

def chart_district_loop(dfh: pd.DataFrame) -> go.Figure | None:
    """Scatter of hourly heating vs cooling + waste-heat bar. [§5 Mitigation]"""
    heat_col = "Qhs_kWh"
    cool_col = next((c for c in ["QC_sys_kWh", "Qcs_kWh"] if c in dfh.columns), None)

    if cool_col is None or heat_col not in dfh.columns or dfh[cool_col].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠ Cooling demand is zero.<br>Run <b>cea cooling-demand</b> to enable this chart.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#9E9E9E"),
        )
        fig.update_layout(title="District Energy Loop Opportunity  [§5 Mitigation]",
                          template="plotly_white", height=300)
        return fig

    loop = dfh[[heat_col, cool_col]].fillna(0).copy()
    loop.columns = ["Heating", "Cooling"]
    mask_both  = (loop["Heating"] > 0) & (loop["Cooling"] > 0)
    mask_honly = (loop["Heating"] > 0) & (loop["Cooling"] == 0)
    mask_conly = (loop["Heating"] == 0) & (loop["Cooling"] > 0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Hourly Load Scatter", "Waste Heat Recovery Potential"])
    for mask, name, color in [
        (mask_honly, f"Heating only ({mask_honly.sum():,} h)", "#EF5350"),
        (mask_conly, f"Cooling only ({mask_conly.sum():,} h)", "#29B6F6"),
        (mask_both,  f"Simultaneous ★ ({mask_both.sum():,} h)", "#7B1FA2"),
    ]:
        sub = loop[mask]
        fig.add_trace(go.Scatter(
            x=sub["Heating"], y=sub["Cooling"], mode="markers",
            marker=dict(size=3, color=color, opacity=0.4 if name.startswith("Simultaneous") else 0.15),
            name=name,
        ), row=1, col=1)

    opp_heat = loop.loc[mask_both, "Heating"].sum() / 1000
    opp_cool = loop.loc[mask_both, "Cooling"].sum() / 1000
    fig.add_trace(go.Bar(
        x=["Recoverable Heat", "Offset Cooling"], y=[opp_heat, opp_cool],
        marker_color=["#EF5350", "#29B6F6"],
        text=[f"{opp_heat:,.1f}", f"{opp_cool:,.1f}"],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        title="District Energy Loop Opportunity  [§5 Mitigation — heat exchange loop]",
        template="plotly_white", height=420,
        legend=dict(orientation="h", y=-0.25),
    )
    fig.update_xaxes(title_text="Heating Load (kWh/h)", row=1, col=1)
    fig.update_yaxes(title_text="Cooling Load (kWh/h)", row=1, col=1)
    fig.update_yaxes(title_text="MWh/yr", row=1, col=2)
    return fig


# ── Chart 8 — CO₂ Reduction Waterfall ────────────────────────────────────────

def chart_co2_waterfall(df: pd.DataFrame, total_co2: float, EF: dict = None,
                         COP: float = 3.0, grid_2030_ef: float = 0.200) -> go.Figure:
    """Waterfall showing cumulative CO₂ reduction per scenario. [§5 Mitigation]"""
    ef = EF or _DEFAULT_EF
    ng_mwh   = (df["NG_hs_MWhyr"]   + df["NG_ww_MWhyr"]).sum()
    oil_mwh  = (df["OIL_hs_MWhyr"]  + df["OIL_ww_MWhyr"]).sum()
    coal_mwh = (df["COAL_hs_MWhyr"] + df["COAL_ww_MWhyr"]).sum()
    grid_mwh = df["GRID_MWhyr"].sum()
    pv_mwh   = df["PV_MWhyr"].sum()
    co2_fossil = ng_mwh * ef["NG"] + oil_mwh * ef["OIL"] + coal_mwh * ef["COAL"]

    scenarios = []
    if co2_fossil > 0:
        s1 = -(co2_fossil) + (ng_mwh + oil_mwh + coal_mwh) / COP * ef["GRID"]
        scenarios.append(("Fossil→HP<br>Electrification", s1))
    s2 = grid_mwh * (grid_2030_ef - ef["GRID"])
    scenarios.append(("Grid<br>Decarbonization", s2))
    if pv_mwh > 0:
        s3 = -pv_mwh * ef["GRID"]
        scenarios.append(("Rooftop<br>Solar PV", s3))

    cumulative = [total_co2]
    for _, d in scenarios:
        cumulative.append(cumulative[-1] + d)
    final = cumulative[-1]
    target = total_co2 * 0.60

    x  = ["Baseline"] + [lbl for lbl, _ in scenarios] + ["Combined<br>Scenario"]
    y  = [total_co2] + [abs(d) for _, d in scenarios] + [final]
    measures = ["absolute"] + ["relative"] * len(scenarios) + ["total"]
    texts = [f"{total_co2:,.0f}t"] + [f"{d:+,.0f}t" for _, d in scenarios] + [f"{final:,.0f}t"]

    fig = go.Figure(go.Waterfall(
        x=x, y=[total_co2] + [d for _, d in scenarios] + [0],
        measure=measures,
        text=texts, textposition="outside",
        connector={"line": {"color": "#9E9E9E", "dash": "dot"}},
        decreasing={"marker": {"color": "#66BB6A"}},
        increasing={"marker": {"color": "#EF5350"}},
        totals={"marker": {"color": "#5C6BC0"}},
    ))
    fig.add_hline(y=target, line_dash="dash", line_color="#1565C0", line_width=2,
                  annotation_text=f"2030 Target −40% ({target:,.0f} tCO₂/yr)",
                  annotation_position="top right")

    fig.update_layout(
        title="CO₂ Reduction Pathway — Scenario Waterfall  [§5 Mitigation]",
        yaxis_title="tCO₂/yr", yaxis_range=[0, total_co2 * 1.25],
        template="plotly_white", height=450,
    )
    return fig


# ── Chart D — Fossil Phase-Out Timeline ──────────────────────────────────────

def chart_fossil_timeline(fossil_mwh: float, co2_fossil: float,
                           EF: dict = None, COP: float = 3.0) -> go.Figure | None:
    """Multi-line: fossil energy + CO₂ decline under different retrofit rates. [§5 + §2.7]"""
    if fossil_mwh == 0:
        return None

    ef = EF or _DEFAULT_EF
    years = np.arange(2025, 2036)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Fossil Energy Consumption (MWh/yr)",
                                        "Net CO₂ from Heating Systems (tCO₂/yr)"])
    colors_rate = ["#B0BEC5", "#81C784", "#4CAF50", "#1B5E20"]

    for rate_pct, color in zip([5, 10, 15, 20], colors_rate):
        remaining = np.array([max(1.0 - rate_pct / 100 * i, 0) for i in range(len(years))])
        fossil_mwh_yr  = fossil_mwh * remaining
        fossil_co2_yr  = co2_fossil * remaining
        hp_extra       = fossil_mwh * (1 - remaining) / COP * ef["GRID"]
        net_co2        = fossil_co2_yr + hp_extra
        lbl = f"{rate_pct}%/yr"

        fig.add_trace(go.Scatter(
            x=years, y=fossil_mwh_yr, mode="lines+markers",
            name=lbl, line=dict(color=color), marker=dict(size=5),
            legendgroup=lbl, showlegend=True,
            hovertemplate=f"{lbl}<br>%{{x}}: %{{y:,.0f}} MWh/yr<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=years, y=net_co2, mode="lines+markers",
            name=lbl, line=dict(color=color), marker=dict(size=5),
            legendgroup=lbl, showlegend=False,
            hovertemplate=f"{lbl}<br>%{{x}}: %{{y:,.0f}} tCO₂/yr<extra></extra>",
        ), row=1, col=2)

    # SECAP target lines
    fig.add_hline(y=fossil_mwh * 0.60, line_dash="dash", line_color="#1565C0",
                  annotation_text="SECAP target", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=co2_fossil * 0.60, line_dash="dash", line_color="#1565C0",
                  annotation_text="SECAP target", annotation_position="top right", row=1, col=2)
    fig.add_vline(x=2030, line_dash="dot", line_color="#B71C1C", row=1, col=1)
    fig.add_vline(x=2030, line_dash="dot", line_color="#B71C1C", row=1, col=2)

    fig.update_layout(
        title="Fossil Fuel Phase-Out Timeline  [§5 Mitigation — timing + §2.7 Monitoring]",
        template="plotly_white", height=420,
        legend=dict(title="Annual retrofit rate", orientation="v"),
        yaxis=dict(rangemode="tozero"),
        yaxis2=dict(rangemode="tozero"),
    )
    fig.update_xaxes(tickvals=list(years[::2]))
    fig.update_yaxes(title_text="MWh/yr", row=1, col=1)
    fig.update_yaxes(title_text="tCO₂/yr", row=1, col=2)
    return fig
