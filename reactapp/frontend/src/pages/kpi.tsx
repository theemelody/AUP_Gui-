import { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import ChartCard from '../components/ChartCard';
import { useScenarioContext } from '../context/ScenarioContext';
import { fetchKpiData, type KpiData, type KpiAnnualRow } from '../services/api';
import { KPI_COLORS, removeZeroSeries, FIELD_LABELS } from '../utils/kpiColors';

// ── Types ────────────────────────────────────────────────────────────────────

type TabId = 'demand' | 'secap' | 'lifecycle' | 'renewables' | 'network';

const TABS: { id: TabId; label: string; requires: string }[] = [
  { id: 'demand',     label: 'Demand',            requires: 'demand' },
  { id: 'secap',      label: 'SECAP',             requires: 'demand' },
  { id: 'lifecycle',  label: 'Emissions & Costs', requires: 'lifecycle' },
  { id: 'renewables', label: 'Tech Potentials',   requires: 'renewables' },
  { id: 'network',    label: 'Thermal Networks',  requires: 'network' },
];

const CHART_H = 340;
const CHART_H_WIDE = 300;

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number | undefined, decimals = 0) {
  if (n === undefined || n === null) return '—';
  return n.toLocaleString('en-US', { maximumFractionDigits: decimals });
}

function sortByTotal(rows: KpiAnnualRow[], fields: (keyof KpiAnnualRow)[]) {
  return [...rows].sort((a, b) => {
    const ta = fields.reduce((s, f) => s + (Number(a[f]) || 0), 0);
    const tb = fields.reduce((s, f) => s + (Number(b[f]) || 0), 0);
    return tb - ta;
  });
}

// ── Demand Tab ───────────────────────────────────────────────────────────────

function DemandTab({ data }: { data: KpiData }) {
  const annual = data.annual ?? [];

  // Chart 1: Annual end-use per building
  const sorted1 = sortByTotal(annual, ['Qhs_sys_MWhyr', 'Qww_sys_MWhyr', 'Qcs_sys_MWhyr', 'E_sys_MWhyr']);
  const names1 = sorted1.map((r) => r.name);
  const endUseOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: names1, axisLabel: { rotate: 35, fontSize: 10 } },
    yAxis: { type: 'value', name: 'MWh/yr', nameTextStyle: { fontSize: 10 } },
    series: removeZeroSeries([
      { name: FIELD_LABELS.Qhs_sys_MWhyr, type: 'bar', stack: 'total', data: sorted1.map((r) => r.Qhs_sys_MWhyr || 0), itemStyle: { color: KPI_COLORS.heating } },
      { name: FIELD_LABELS.Qww_sys_MWhyr,  type: 'bar', stack: 'total', data: sorted1.map((r) => r.Qww_sys_MWhyr || 0),  itemStyle: { color: KPI_COLORS.dhw } },
      { name: FIELD_LABELS.Qcs_sys_MWhyr, type: 'bar', stack: 'total', data: sorted1.map((r) => r.Qcs_sys_MWhyr || 0), itemStyle: { color: KPI_COLORS.cooling } },
      { name: FIELD_LABELS.E_sys_MWhyr,   type: 'bar', stack: 'total', data: sorted1.map((r) => r.E_sys_MWhyr || 0),   itemStyle: { color: KPI_COLORS.electricity } },
    ]),
  };

  // Chart 2: EUI per building
  const sorted2 = sortByTotal(annual, ['Qhs_sys_MWhyr', 'Qww_sys_MWhyr', 'Qcs_sys_MWhyr', 'E_sys_MWhyr']);
  const totalEui = sorted2.map((r) => r.GFA_m2 > 0 ? (((r.Qhs_sys_MWhyr || 0) + (r.Qww_sys_MWhyr || 0) + (r.Qcs_sys_MWhyr || 0) + (r.E_sys_MWhyr || 0)) * 1000 / r.GFA_m2) : 0);
  const meanEui = totalEui.length ? totalEui.reduce((a, b) => a + b, 0) / totalEui.length : 0;
  const euiOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: sorted2.map((r) => r.name), axisLabel: { rotate: 35, fontSize: 10 } },
    yAxis: { type: 'value', name: 'kWh/m²/yr', nameTextStyle: { fontSize: 10 } },
    series: removeZeroSeries([
      { name: 'Space Heating', type: 'bar', stack: 'eui', data: sorted2.map((r) => r.GFA_m2 > 0 ? +((r.Qhs_sys_MWhyr || 0) * 1000 / r.GFA_m2).toFixed(1) : 0), itemStyle: { color: KPI_COLORS.heating } },
      { name: 'Hot Water',     type: 'bar', stack: 'eui', data: sorted2.map((r) => r.GFA_m2 > 0 ? +((r.Qww_sys_MWhyr || 0) * 1000 / r.GFA_m2).toFixed(1) : 0),  itemStyle: { color: KPI_COLORS.dhw } },
      { name: 'Cooling',       type: 'bar', stack: 'eui', data: sorted2.map((r) => r.GFA_m2 > 0 ? +((r.Qcs_sys_MWhyr || 0) * 1000 / r.GFA_m2).toFixed(1) : 0), itemStyle: { color: KPI_COLORS.cooling } },
      { name: 'Electricity',   type: 'bar', stack: 'eui', data: sorted2.map((r) => r.GFA_m2 > 0 ? +((r.E_sys_MWhyr || 0) * 1000 / r.GFA_m2).toFixed(1) : 0),   itemStyle: { color: KPI_COLORS.electricity },
        markLine: { silent: true, data: [
          { yAxis: 200, name: 'NZEB 200', lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: 'NZEB 200' } },
          { yAxis: meanEui, name: 'Mean', lineStyle: { color: '#f97316', type: 'dotted' }, label: { formatter: `Mean ${fmt(meanEui, 0)}` } },
        ]},
      },
    ]),
  };

  // Chart 3: Peak loads
  const sorted3 = sortByTotal(annual, ['E_sys0_kW', 'Qhs_sys0_kW', 'Qcs_sys0_kW'] as (keyof KpiAnnualRow)[]);
  const peakOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: sorted3.map((r) => r.name), axisLabel: { rotate: 35, fontSize: 10 } },
    yAxis: { type: 'value', name: 'kW', nameTextStyle: { fontSize: 10 } },
    series: removeZeroSeries([
      { name: 'Peak Electricity', type: 'bar', data: sorted3.map((r) => r.E_sys0_kW || 0),   itemStyle: { color: KPI_COLORS.electricity } },
      { name: 'Peak Heating',     type: 'bar', data: sorted3.map((r) => r.Qhs_sys0_kW || 0), itemStyle: { color: KPI_COLORS.heating } },
      { name: 'Peak Cooling',     type: 'bar', data: sorted3.map((r) => r.Qcs_sys0_kW || 0), itemStyle: { color: KPI_COLORS.cooling } },
    ]),
  };

  // Chart 4: Monthly demand
  const monthly = data.monthly;
  const monthlyOption = monthly ? {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: monthly.labels },
    yAxis: { type: 'value', name: 'MWh', nameTextStyle: { fontSize: 10 } },
    series: removeZeroSeries([
      { name: 'Heating', type: 'bar', stack: 'total', data: monthly.heating_MWh,     itemStyle: { color: KPI_COLORS.heating } },
      { name: 'Cooling', type: 'bar', stack: 'total', data: monthly.cooling_MWh,     itemStyle: { color: KPI_COLORS.cooling } },
      { name: 'Electricity', type: 'bar', stack: 'total', data: monthly.electricity_MWh, itemStyle: { color: KPI_COLORS.electricity } },
    ]),
  } : null;

  // Chart 5: Load duration curves — 3 panels
  const ldc = data.load_duration;
  const ldcPanels: ReadonlyArray<{ key: 'heating_kWh' | 'cooling_kWh' | 'electricity_kWh'; label: string; color: string }> = [
    { key: 'heating_kWh',     label: 'Heating',     color: KPI_COLORS.heating },
    { key: 'cooling_kWh',     label: 'Cooling',     color: KPI_COLORS.cooling },
    { key: 'electricity_kWh', label: 'Electricity', color: KPI_COLORS.electricity },
  ];

  // Chart 6: Hourly load curve
  const hs = data.hourly_sample;
  const hourlyOption = hs ? {
    tooltip: { trigger: 'axis', axisPointer: { type: 'line' } },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 60, top: 16, bottom: 56 },
    xAxis: { type: 'category', data: hs.dates as string[], axisLabel: { interval: Math.floor((hs.dates as string[]).length / 12), fontSize: 10 } },
    yAxis: [
      { type: 'value', name: 'kWh/h', nameTextStyle: { fontSize: 10 } },
      { type: 'value', name: '°C', nameTextStyle: { fontSize: 10 }, position: 'right' },
    ],
    series: removeZeroSeries([
      ...[
        { name: FIELD_LABELS.Qhs_sys_kWh, key: 'Qhs_sys_kWh', color: KPI_COLORS.heating },
        { name: FIELD_LABELS.Qww_sys_kWh, key: 'Qww_sys_kWh', color: KPI_COLORS.dhw },
        { name: FIELD_LABELS.Qcs_sys_kWh, key: 'Qcs_sys_kWh', color: KPI_COLORS.cooling },
        { name: FIELD_LABELS.E_sys_kWh,   key: 'E_sys_kWh',   color: KPI_COLORS.electricity },
      ].filter((s) => hs[s.key]).map((s) => ({
        name: s.name, type: 'bar', stack: 'load',
        data: (hs[s.key] as number[]).map((v, i) => s.key === 'Qcs_sys_kWh' ? -Math.abs(v) : v),
        itemStyle: { color: s.color },
      })),
      ...(hs['theta_op_avg_C'] ? [{
        name: FIELD_LABELS.theta_op_avg_C, type: 'line', yAxisIndex: 1,
        data: hs['theta_op_avg_C'] as number[],
        lineStyle: { color: KPI_COLORS.outdoor, width: 1 },
        itemStyle: { color: KPI_COLORS.outdoor },
        showSymbol: false,
        areaStyle: undefined,
      }] : []),
    ]),
  } : null;

  // Chart 7: Thermal balance
  const bal = data.monthly_balance;
  const balanceOption = bal ? (() => {
    const positiveKeys = ['I_sol_kWh', 'Q_gain_sen_peop_kWh', 'Q_gain_sen_light_kWh', 'Q_gain_sen_app_kWh', 'Q_gain_sen_wall_kWh'] as const;
    const negativeKeys = ['Q_loss_sen_ref_kWh', 'I_rad_kWh'] as const;
    const colorMap: Record<string, string> = {
      I_sol_kWh: KPI_COLORS.I_sol, Q_gain_sen_peop_kWh: KPI_COLORS.Q_gain_sen_peop,
      Q_gain_sen_light_kWh: KPI_COLORS.Q_gain_sen_light, Q_gain_sen_app_kWh: KPI_COLORS.Q_gain_sen_app,
      Q_gain_sen_wall_kWh: KPI_COLORS.Q_gain_sen_wall, Q_loss_sen_ref_kWh: KPI_COLORS.Q_loss_sen_ref,
      I_rad_kWh: KPI_COLORS.I_rad,
    };
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 10 }, type: 'scroll' },
      grid: { left: 60, right: 20, top: 16, bottom: 72 },
      xAxis: { type: 'category', data: bal.labels as string[] },
      yAxis: { type: 'value', name: 'kWh', nameTextStyle: { fontSize: 10 } },
      series: removeZeroSeries([
        ...positiveKeys.filter((k) => bal[k]).map((k) => ({
          name: FIELD_LABELS[k] || k, type: 'bar', stack: 'gains',
          data: bal[k] as number[], itemStyle: { color: colorMap[k] },
        })),
        ...negativeKeys.filter((k) => bal[k]).map((k) => ({
          name: FIELD_LABELS[k] || k, type: 'bar', stack: 'losses',
          data: (bal[k] as number[]).map((v) => -Math.abs(v)),
          itemStyle: { color: colorMap[k] },
        })),
      ]),
    };
  })() : null;

  // Chart 8: Solar radiation by orientation
  const sol = data.solar_radiation;
  const solarKeys = ['roofs_top', 'walls_south', 'walls_east', 'walls_west', 'walls_north',
                     'windows_south', 'windows_east', 'windows_west', 'windows_north'] as const;
  const solarOption = sol ? {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { fontSize: 10 }, type: 'scroll' },
    grid: { left: 60, right: 20, top: 16, bottom: 72 },
    xAxis: { type: 'category', data: sol.labels as string[] },
    yAxis: { type: 'value', name: 'MWh/month', nameTextStyle: { fontSize: 10 } },
    series: removeZeroSeries(
      solarKeys.filter((k) => sol[k]).map((k) => ({
        name: FIELD_LABELS[k] || k, type: 'bar', stack: 'solar',
        data: sol[k] as number[],
        itemStyle: { color: (KPI_COLORS as Record<string, string>)[k] || '#ccc' },
      }))
    ),
  } : null;

  return (
    <div className="kpi-grid">
      <ChartCard title="Annual End-Use per Building" subtitle="MWh/yr">
        <ReactECharts option={endUseOption} style={{ height: CHART_H }} />
      </ChartCard>

      <ChartCard title="Energy Use Intensity (EUI)" subtitle="kWh/m²/yr">
        <ReactECharts option={euiOption} style={{ height: CHART_H }} />
      </ChartCard>

      <ChartCard title="Peak Loads" subtitle="kW">
        <ReactECharts option={peakOption} style={{ height: CHART_H }} />
      </ChartCard>

      <ChartCard title="Monthly Demand Profile" subtitle="MWh/month">
        {monthlyOption ? <ReactECharts option={monthlyOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No monthly data</div>}
      </ChartCard>

      {ldcPanels.map((panel) => (
        <ChartCard key={panel.key} title={`Load Duration — ${panel.label}`} subtitle="kWh/h">
          {ldc && ldc[panel.key].length > 0 ? (
            <ReactECharts option={{
              tooltip: { trigger: 'axis' },
              grid: { left: 60, right: 20, top: 16, bottom: 40 },
              xAxis: { type: 'value', name: 'Hours', nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
              yAxis: { type: 'value', name: 'kWh/h', nameTextStyle: { fontSize: 10 } },
              series: [{
                type: 'line', data: ldc[panel.key].map((v, i) => [i * 9, v]),
                lineStyle: { color: panel.color, width: 1.5 },
                itemStyle: { color: panel.color }, showSymbol: false,
                areaStyle: { color: panel.color, opacity: 0.15 },
              }],
            }} style={{ height: CHART_H }} />
          ) : <div className="kpi-empty">No data</div>}
        </ChartCard>
      ))}

      <ChartCard title="Hourly Load Curve — Full Year" subtitle="kWh/h · °C" wide>
        {hourlyOption ? <ReactECharts option={hourlyOption} style={{ height: CHART_H_WIDE }} /> : <div className="kpi-empty">No hourly data</div>}
      </ChartCard>

      <ChartCard title="Monthly Thermal Gains & Losses" subtitle="kWh/month">
        {balanceOption ? <ReactECharts option={balanceOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No balance data</div>}
      </ChartCard>

      <ChartCard title="Solar Radiation by Orientation" subtitle="MWh/month">
        {solarOption ? <ReactECharts option={solarOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No radiation data</div>}
      </ChartCard>
    </div>
  );
}

// ── SECAP Tab ────────────────────────────────────────────────────────────────

function SecapTab({ data }: { data: KpiData }) {
  const annual = data.annual ?? [];
  const totalGfa = data.meta.total_gfa_m2;
  const buildingCount = data.meta.building_count;
  const totalDemand = annual.reduce((s, r) => s + (r.Qhs_sys_MWhyr || 0) + (r.Qww_sys_MWhyr || 0) + (r.Qcs_sys_MWhyr || 0) + (r.E_sys_MWhyr || 0), 0);
  const meanEui = totalGfa > 0 ? totalDemand * 1000 / totalGfa : 0;
  const peakElec = Math.max(...annual.map((r) => r.E_sys0_kW || 0));

  const donutData = [
    { value: annual.reduce((s, r) => s + (r.Qhs_sys_MWhyr || 0), 0), name: 'Space Heating', itemStyle: { color: KPI_COLORS.heating } },
    { value: annual.reduce((s, r) => s + (r.Qww_sys_MWhyr || 0), 0),  name: 'Hot Water',     itemStyle: { color: KPI_COLORS.dhw } },
    { value: annual.reduce((s, r) => s + (r.Qcs_sys_MWhyr || 0), 0), name: 'Cooling',       itemStyle: { color: KPI_COLORS.cooling } },
    { value: annual.reduce((s, r) => s + (r.E_sys_MWhyr || 0), 0),   name: 'Electricity',   itemStyle: { color: KPI_COLORS.electricity } },
  ].filter((d) => d.value > 0);

  const donutOption = {
    tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    series: [{ type: 'pie', radius: ['40%', '68%'], data: donutData, label: { fontSize: 11 } }],
  };

  // EUI histogram
  const euis = annual.map((r) => r.GFA_m2 > 0 ? (((r.Qhs_sys_MWhyr || 0) + (r.Qww_sys_MWhyr || 0) + (r.Qcs_sys_MWhyr || 0) + (r.E_sys_MWhyr || 0)) * 1000 / r.GFA_m2) : 0);
  const bins = [0, 50, 100, 150, 200, Infinity];
  const binLabels = ['0–50', '50–100', '100–150', '150–200', '200+'];
  const binCounts = bins.slice(0, -1).map((lo, i) => euis.filter((v) => v >= lo && v < bins[i + 1]).length);
  const histOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 16, bottom: 40 },
    xAxis: { type: 'category', data: binLabels, name: 'kWh/m²/yr', nameLocation: 'middle', nameGap: 28 },
    yAxis: { type: 'value', name: 'Buildings', nameTextStyle: { fontSize: 10 } },
    series: [{
      type: 'bar', data: binCounts, itemStyle: { color: '#5c6bc0' },
      markLine: { silent: true, data: [
        { xAxis: 4, name: 'NZEB 200', lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: 'NZEB 200' } },
        { yAxis: binCounts.reduce((a, b) => a + b, 0) / binLabels.length, name: 'Mean', lineStyle: { color: '#f97316', type: 'dotted' } },
      ]},
    }],
  };

  // Monthly 100% stacked
  const monthly = data.monthly;
  const monthly100Option = monthly ? (() => {
    const totals = monthly.labels.map((_, i) => (monthly.heating_MWh[i] || 0) + (monthly.cooling_MWh[i] || 0) + (monthly.electricity_MWh[i] || 0));
    const pct = (arr: number[]) => arr.map((v, i) => totals[i] > 0 ? +(v * 100 / totals[i]).toFixed(1) : 0);
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 60, right: 20, top: 16, bottom: 56 },
      xAxis: { type: 'category', data: monthly.labels },
      yAxis: { type: 'value', name: '%', max: 100, nameTextStyle: { fontSize: 10 } },
      series: removeZeroSeries([
        { name: 'Heating',     type: 'bar', stack: 'pct', data: pct(monthly.heating_MWh),     itemStyle: { color: KPI_COLORS.heating } },
        { name: 'Cooling',     type: 'bar', stack: 'pct', data: pct(monthly.cooling_MWh),     itemStyle: { color: KPI_COLORS.cooling } },
        { name: 'Electricity', type: 'bar', stack: 'pct', data: pct(monthly.electricity_MWh), itemStyle: { color: KPI_COLORS.electricity } },
      ]),
    };
  })() : null;

  // LDC (same as demand tab but in SECAP section)
  const ldc = data.load_duration;

  return (
    <>
      <div className="kpi-metric-row">
        <div className="kpi-metric-card">
          <div className="kpi-metric-label">Buildings</div>
          <div className="kpi-metric-value">{buildingCount}</div>
        </div>
        <div className="kpi-metric-card">
          <div className="kpi-metric-label">Total GFA</div>
          <div className="kpi-metric-value">{fmt(totalGfa / 1000, 1)}<span className="kpi-metric-unit">k m²</span></div>
        </div>
        <div className="kpi-metric-card">
          <div className="kpi-metric-label">Mean EUI</div>
          <div className="kpi-metric-value">{fmt(meanEui, 0)}<span className="kpi-metric-unit">kWh/m²/yr</span></div>
        </div>
        <div className="kpi-metric-card">
          <div className="kpi-metric-label">Peak Electricity</div>
          <div className="kpi-metric-value">{fmt(peakElec, 0)}<span className="kpi-metric-unit">kW</span></div>
        </div>
      </div>

      <div className="kpi-grid">
        <ChartCard title="District End-Use Breakdown" subtitle="Annual MWh">
          <ReactECharts option={donutOption} style={{ height: CHART_H }} />
        </ChartCard>

        <ChartCard title="EUI Distribution" subtitle="Building count per bin">
          <ReactECharts option={histOption} style={{ height: CHART_H }} />
        </ChartCard>

        <ChartCard title="Monthly Share of Demand (100%)" subtitle="% per carrier/month">
          {monthly100Option ? <ReactECharts option={monthly100Option} style={{ height: CHART_H }} /> : <div className="kpi-empty">No monthly data</div>}
        </ChartCard>

        {((['heating_kWh', 'cooling_kWh', 'electricity_kWh'] as const).map((key) => {
          const labels: Record<'heating_kWh' | 'cooling_kWh' | 'electricity_kWh', string> = { heating_kWh: 'Heating', cooling_kWh: 'Cooling', electricity_kWh: 'Electricity' };
          const colors: Record<'heating_kWh' | 'cooling_kWh' | 'electricity_kWh', string> = { heating_kWh: KPI_COLORS.heating, cooling_kWh: KPI_COLORS.cooling, electricity_kWh: KPI_COLORS.electricity };
          return (
            <ChartCard key={key} title={`LDC — ${labels[key]}`} subtitle="kWh/h">
              {ldc && ldc[key].length > 0 ? (
                <ReactECharts option={{
                  tooltip: { trigger: 'axis' },
                  grid: { left: 60, right: 20, top: 16, bottom: 40 },
                  xAxis: { type: 'value', name: 'Hours', nameTextStyle: { fontSize: 10 } },
                  yAxis: { type: 'value', name: 'kWh/h', nameTextStyle: { fontSize: 10 } },
                  series: [{
                    type: 'line', data: ldc[key].map((v: number, i: number) => [i * 9, v]),
                    lineStyle: { color: colors[key], width: 1.5 }, itemStyle: { color: colors[key] },
                    showSymbol: false, areaStyle: { color: colors[key], opacity: 0.15 },
                  }],
                }} style={{ height: CHART_H }} />
              ) : <div className="kpi-empty">No data</div>}
            </ChartCard>
          );
        }))}
      </div>
    </>
  );
}

// ── Emissions & Costs Tab (Profile B) ────────────────────────────────────────

function LifecycleTab({ data }: { data: KpiData }) {
  const locked = !data.available.includes('lifecycle');

  const emissionsOption = (() => {
    if (!data.emissions) return null;
    // CEA 4.x: flat rows with type='building'|'plant', values in kgCO2e → convert to tCO2e
    type EmRow = { name?: string; type?: string; operation_kgCO2e?: number; production_kgCO2e?: number; biogenic_kgCO2e?: number; demolition_kgCO2e?: number };
    const allRows = data.emissions as EmRow[];
    const buildings = allRows.filter((r) => r.type !== 'plant');
    const plantMap = Object.fromEntries(
      allRows.filter((r) => r.type === 'plant').map((r) => [r.name || '', r])
    );
    const sorted = [...buildings].sort((a, b) =>
      ((b.operation_kgCO2e || 0) - (a.operation_kgCO2e || 0))
    );
    const names = sorted.map((r) => r.name || '');
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 60, right: 20, top: 16, bottom: 56 },
      xAxis: { type: 'category', data: names, axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'tCO₂e/yr', nameTextStyle: { fontSize: 10 } },
      series: removeZeroSeries([
        { name: 'Operational (building)', type: 'bar', stack: 'ghg',
          data: sorted.map((r) => +((r.operation_kgCO2e || 0) / 1000).toFixed(3)),
          itemStyle: { color: '#f97474' } },
        { name: 'Operational (district plant)', type: 'bar', stack: 'ghg',
          data: names.map((n) => +((plantMap[n]?.operation_kgCO2e || 0) / 1000).toFixed(3)),
          itemStyle: { color: KPI_COLORS.ghg_op } },
        { name: 'Embodied (production)', type: 'bar', stack: 'ghg',
          data: sorted.map((r) => +((r.production_kgCO2e || 0) / 1000).toFixed(3)),
          itemStyle: { color: KPI_COLORS.ghg_emb } },
        { name: 'Biogenic', type: 'bar', stack: 'ghg',
          data: sorted.map((r) => +((r.biogenic_kgCO2e || 0) / 1000).toFixed(3)),
          itemStyle: { color: '#7ec78f' } },
        { name: 'Demolition', type: 'bar', stack: 'ghg',
          data: sorted.map((r) => +((r.demolition_kgCO2e || 0) / 1000).toFixed(3)),
          itemStyle: { color: '#b0a090' } },
      ]),
    };
  })();

  const costsOption = (() => {
    if (!data.costs) return null;
    // CEA 4.x: rows with type='building'|'plant', capex_a_USD, opex_fixed_a_USD, opex_var_a_USD, TAC_USD
    type CostRow = { name?: string; type?: string; capex_a_USD?: number; opex_fixed_a_USD?: number; opex_var_a_USD?: number; TAC_USD?: number };
    const allRows = data.costs as CostRow[];
    const buildings = allRows.filter((r) => r.type !== 'plant');
    const plantMap = Object.fromEntries(
      allRows.filter((r) => r.type === 'plant').map((r) => [r.name || '', r])
    );
    const sorted = [...buildings].sort((a, b) =>
      ((b.capex_a_USD || 0) + (b.opex_fixed_a_USD || 0) + (b.opex_var_a_USD || 0))
      - ((a.capex_a_USD || 0) + (a.opex_fixed_a_USD || 0) + (a.opex_var_a_USD || 0))
    );
    const names = sorted.map((r) => r.name || '');
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 70, right: 20, top: 16, bottom: 56 },
      xAxis: { type: 'category', data: names, axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'USD/yr', nameTextStyle: { fontSize: 10 } },
      series: removeZeroSeries([
        { name: 'CAPEX (building)',        type: 'bar', stack: 'cost',
          data: sorted.map((r) => +(r.capex_a_USD || 0).toFixed(0)),
          itemStyle: { color: '#fde68a' } },
        { name: 'CAPEX (district plant)',  type: 'bar', stack: 'cost',
          data: names.map((n) => +((plantMap[n]?.capex_a_USD || 0)).toFixed(0)),
          itemStyle: { color: KPI_COLORS.capex } },
        { name: 'OPEX (building)',         type: 'bar', stack: 'cost',
          data: sorted.map((r) => +((r.opex_fixed_a_USD || 0) + (r.opex_var_a_USD || 0)).toFixed(0)),
          itemStyle: { color: '#93c5fd' } },
        { name: 'OPEX (district plant)',   type: 'bar', stack: 'cost',
          data: names.map((n) => +((plantMap[n]?.opex_fixed_a_USD || 0) + (plantMap[n]?.opex_var_a_USD || 0)).toFixed(0)),
          itemStyle: { color: KPI_COLORS.opex } },
      ]),
    };
  })();

  const tacMetrics = (() => {
    if (!data.costs) return null;
    type CostRow = { type?: string; TAC_USD?: number };
    const rows = data.costs as CostRow[];
    const district  = rows.filter((r) => r.type === 'plant').reduce((s, r) => s + (r.TAC_USD || 0), 0);
    const buildings = rows.filter((r) => r.type !== 'plant').reduce((s, r) => s + (r.TAC_USD || 0), 0);
    return { district, buildings, total: district + buildings };
  })();

  return (
    <>
      {tacMetrics && (
        <div className="kpi-metric-row">
          <div className="kpi-metric-card"><span className="kpi-metric-label">TAC — District</span><span className="kpi-metric-value">${fmt(tacMetrics.district)}/yr</span></div>
          <div className="kpi-metric-card"><span className="kpi-metric-label">TAC — Buildings</span><span className="kpi-metric-value">${fmt(tacMetrics.buildings)}/yr</span></div>
          <div className="kpi-metric-card"><span className="kpi-metric-label">TAC — Combined</span><span className="kpi-metric-value">${fmt(tacMetrics.total)}/yr</span></div>
        </div>
      )}
      <div className="kpi-grid">
        <ChartCard title="Annual GHG Emissions" subtitle="tCO₂-eq/yr per building" locked={locked} lockMessage="Run Profile B — Lifecycle Assessment to unlock">
          {emissionsOption ? <ReactECharts option={emissionsOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No emissions data</div>}
        </ChartCard>
        <ChartCard title="Annual System Costs (CAPEX / OPEX)" subtitle="USD/yr per building" locked={locked} lockMessage="Run Profile B — Lifecycle Assessment to unlock">
          {costsOption ? <ReactECharts option={costsOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No cost data</div>}
        </ChartCard>
      </div>
    </>
  );
}

// ── Tech Potentials Tab (Profile C) ─────────────────────────────────────────

function RenewablesTab({ data }: { data: KpiData }) {
  const locked = !data.available.includes('renewables');
  const pot = data.potentials;

  type PotRow = { Name?: string; name?: string; [key: string]: unknown };

  const pvDemandOption = (() => {
    if (!pot?.pv || !data.annual) return null;
    const pvMap = Object.fromEntries((pot.pv as PotRow[]).map((r) => [(r.Name || r.name || ''), r]));
    const sorted = sortByTotal(data.annual, ['Qhs_sys_MWhyr', 'Qww_sys_MWhyr', 'Qcs_sys_MWhyr', 'E_sys_MWhyr']);
    const names = sorted.map((r) => r.name);
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 60, right: 20, top: 16, bottom: 56 },
      xAxis: { type: 'category', data: names, axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'MWh/yr', nameTextStyle: { fontSize: 10 } },
      series: [
        { name: 'Total Demand', type: 'bar', data: sorted.map((r) => +((r.Qhs_sys_MWhyr||0)+(r.Qww_sys_MWhyr||0)+(r.Qcs_sys_MWhyr||0)+(r.E_sys_MWhyr||0)).toFixed(1)), itemStyle: { color: '#cbd5e1' } },
        { name: 'PV Potential', type: 'bar', data: names.map((n) => { const row = pvMap[n]; return row ? +(Number(row.E_PV_gen_kWh || row.E_PV_MWhyr || 0) / 1000).toFixed(2) : 0; }), itemStyle: { color: KPI_COLORS.solar } },
      ],
    };
  })();

  const pvtScOption = (() => {
    if (!pot?.pvt && !pot?.sc) return null;
    const pvtRows = (pot.pvt || []) as PotRow[];
    const scRows  = (pot.sc  || []) as PotRow[];
    const names = pvtRows.map((r) => r.Name || r.name || '');
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 60, right: 20, top: 16, bottom: 56 },
      xAxis: { type: 'category', data: names, axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'MWh/yr', nameTextStyle: { fontSize: 10 } },
      series: removeZeroSeries([
        { name: 'PVT Thermal', type: 'bar', stack: 'pvtsc', data: pvtRows.map((r) => +(Number(r.Q_PVT_gen_kWh || 0) / 1000).toFixed(2)), itemStyle: { color: KPI_COLORS.pvt } },
        { name: 'SC Thermal',  type: 'bar', stack: 'pvtsc', data: scRows.map((r) => +(Number(r.Q_SC_gen_kWh || 0) / 1000).toFixed(2)),  itemStyle: { color: KPI_COLORS.sc } },
      ]),
    };
  })();

  const mixDonutData = (() => {
    if (!pot) return [];
    const pv  = pot.pv  ? (pot.pv  as PotRow[]).reduce((s, r) => s + Number(r.E_PV_gen_kWh   || r.E_PV_MWhyr   || 0), 0) : 0;
    const pvt = pot.pvt ? (pot.pvt as PotRow[]).reduce((s, r) => s + Number(r.Q_PVT_gen_kWh  || 0), 0) : 0;
    const sc  = pot.sc  ? (pot.sc  as PotRow[]).reduce((s, r) => s + Number(r.Q_SC_gen_kWh   || 0), 0) : 0;
    const geo = pot.geothermal ? (pot.geothermal as PotRow[]).reduce((s, r) => s + Number(r.QGHP_kW || 0), 0) : 0;
    const sew = pot.sewage     ? (pot.sewage     as PotRow[]).reduce((s, r) => s + Number(r.Qsw_kW  || 0), 0) : 0;
    return [
      { value: +(pv / 1000).toFixed(1),  name: 'PV',         itemStyle: { color: KPI_COLORS.solar } },
      { value: +(pvt / 1000).toFixed(1), name: 'PVT',        itemStyle: { color: KPI_COLORS.pvt } },
      { value: +(sc / 1000).toFixed(1),  name: 'SC',         itemStyle: { color: KPI_COLORS.sc } },
      { value: +(geo / 1000).toFixed(1), name: 'Geothermal', itemStyle: { color: KPI_COLORS.geothermal } },
      { value: +(sew / 1000).toFixed(1), name: 'Sewage',     itemStyle: { color: KPI_COLORS.sewage } },
    ].filter((d) => d.value > 0);
  })();

  const geoOption = (() => {
    if (!pot?.geothermal) return null;
    const rows = pot.geothermal as PotRow[];
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 60, right: 20, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: rows.map((r) => r.Name || r.name || ''), axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'kW', nameTextStyle: { fontSize: 10 } },
      series: [{ type: 'bar', data: rows.map((r) => Number(r.QGHP_kW || 0)), itemStyle: { color: KPI_COLORS.geothermal } }],
    };
  })();

  const sewageOption = (() => {
    if (!pot?.sewage) return null;
    const rows = pot.sewage as PotRow[];
    const names = rows.map((r) => r.Name || r.name || '');
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 60, right: 20, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: names, axisLabel: { rotate: 35, fontSize: 10 } },
      yAxis: { type: 'value', name: 'kW', nameTextStyle: { fontSize: 10 } },
      series: [{ type: 'line', data: rows.map((r) => Number(r.Qsw_kW || 0)), lineStyle: { color: KPI_COLORS.sewage }, itemStyle: { color: KPI_COLORS.sewage }, showSymbol: false, areaStyle: { color: KPI_COLORS.sewage, opacity: 0.2 } }],
    };
  })();

  return (
    <div className="kpi-grid">
      <ChartCard title="PV Potential vs Total Demand" subtitle="MWh/yr" locked={locked} lockMessage="Run Profile C — Renewable Potentials to unlock">
        {pvDemandOption ? <ReactECharts option={pvDemandOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No PV data</div>}
      </ChartCard>

      <ChartCard title="PVT + SC Thermal Output" subtitle="MWh/yr" locked={locked} lockMessage="Run Profile C — Renewable Potentials to unlock">
        {pvtScOption ? <ReactECharts option={pvtScOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No PVT/SC data</div>}
      </ChartCard>

      <ChartCard title="Renewable Mix Summary" subtitle="Annual potential share" locked={locked} lockMessage="Run Profile C — Renewable Potentials to unlock">
        {mixDonutData.length > 0 ? (
          <ReactECharts option={{
            tooltip: { trigger: 'item', formatter: '{b}: {d}% ({c} MWh)' },
            legend: { bottom: 0, textStyle: { fontSize: 11 } },
            series: [{ type: 'pie', radius: ['40%', '68%'], data: mixDonutData, label: { fontSize: 11 } }],
          }} style={{ height: CHART_H }} />
        ) : <div className="kpi-empty">No data</div>}
      </ChartCard>

      <ChartCard title="Geothermal Capacity" subtitle="kW per building" locked={locked} lockMessage="Run Profile C — Renewable Potentials to unlock">
        {geoOption ? <ReactECharts option={geoOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No geothermal data</div>}
      </ChartCard>

      <ChartCard title="Sewage Heat Availability" subtitle="kW" locked={locked} lockMessage="Run Profile C — Renewable Potentials to unlock">
        {sewageOption ? <ReactECharts option={sewageOption} style={{ height: CHART_H }} /> : <div className="kpi-empty">No sewage data</div>}
      </ChartCard>
    </div>
  );
}

// ── Thermal Networks Tab (Profile D) ─────────────────────────────────────────

function NetworkTab({ data }: { data: KpiData }) {
  const locked = !data.available.includes('network');
  const net = data.network as Record<string, unknown> | null;

  const plantCols = net ? Object.keys(net).filter((k) => k !== 'labels' && k !== 'name' && Array.isArray(net[k])) : [];

  const ldcOption = (() => {
    if (!net || !plantCols.length) return null;
    const col = plantCols[0];
    const vals = net[col] as number[];
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 60, right: 20, top: 16, bottom: 40 },
      xAxis: { type: 'value', name: 'Hours ranked', nameTextStyle: { fontSize: 10 } },
      yAxis: { type: 'value', name: 'kW', nameTextStyle: { fontSize: 10 } },
      series: [{ type: 'line', data: vals.map((v, i) => [i * 9, v]), lineStyle: { color: KPI_COLORS.heating, width: 1.5 }, itemStyle: { color: KPI_COLORS.heating }, showSymbol: false, areaStyle: { color: KPI_COLORS.heating, opacity: 0.15 } }],
    };
  })();

  return (
    <div className="kpi-grid">
      <ChartCard title="Network Demand Duration Curve" subtitle="kW" locked={locked} lockMessage="Run Profile D — District Network to unlock" wide>
        {ldcOption ? <ReactECharts option={ldcOption} style={{ height: CHART_H_WIDE }} /> : <div className="kpi-empty">No network data</div>}
      </ChartCard>

      {plantCols.slice(1).map((col) => (
        <ChartCard key={col} title={col} locked={locked} lockMessage="Run Profile D — District Network to unlock">
          {net ? <ReactECharts option={{
            tooltip: { trigger: 'axis' },
            grid: { left: 60, right: 20, top: 16, bottom: 40 },
            xAxis: { type: 'value', name: 'Hours ranked', nameTextStyle: { fontSize: 10 } },
            yAxis: { type: 'value', name: 'kW', nameTextStyle: { fontSize: 10 } },
            series: [{ type: 'line', data: (net[col] as number[]).map((v, i) => [i * 9, v]), lineStyle: { color: KPI_COLORS.heating, width: 1.5 }, itemStyle: { color: KPI_COLORS.heating }, showSymbol: false }],
          }} style={{ height: CHART_H }} /> : null}
        </ChartCard>
      ))}
    </div>
  );
}

// ── Main KPI Page ─────────────────────────────────────────────────────────────

function KPI() {
  const { selectedScenarioForSim, simulationStatus } = useScenarioContext();
  const [activeTab, setActiveTab] = useState<TabId>('demand');
  const [kpiData, setKpiData] = useState<KpiData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedScenarioForSim) { setKpiData(null); return; }
    setLoading(true);
    setError(null);
    fetchKpiData(selectedScenarioForSim)
      .then((d) => { setKpiData(d); setLoading(false); })
      .catch((e) => { setError(String(e?.message || e)); setLoading(false); });
  }, [selectedScenarioForSim, simulationStatus]);

  const available = kpiData?.available ?? [];

  return (
    <div className="kpi-page">
      <div className="kpi-header">
        <h1>KPI Dashboard</h1>
        {selectedScenarioForSim && (
          <span className="kpi-scenario-badge">{selectedScenarioForSim}</span>
        )}
      </div>

      {!selectedScenarioForSim && (
        <div className="kpi-empty">Select a scenario in the Building Workspace to view KPIs.</div>
      )}

      {selectedScenarioForSim && (
        <>
          <div className="kpi-tabs">
            {TABS.map((tab) => {
              const isLocked = !available.includes(tab.requires);
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  className={['kpi-tab-btn', isActive ? 'active' : '', isLocked ? 'locked' : ''].filter(Boolean).join(' ')}
                  onClick={() => { if (!isLocked) setActiveTab(tab.id); }}
                >
                  {isLocked && <span className="kpi-tab-lock-icon">🔒</span>}
                  {tab.label}
                </button>
              );
            })}
          </div>

          {loading && <div className="kpi-loading">Loading KPI data…</div>}
          {error && <div className="kpi-loading" style={{ color: '#ef4444' }}>Error: {error}</div>}

          {kpiData && !loading && (
            <>
              {activeTab === 'demand'     && <DemandTab data={kpiData} />}
              {activeTab === 'secap'      && <SecapTab data={kpiData} />}
              {activeTab === 'lifecycle'  && <LifecycleTab data={kpiData} />}
              {activeTab === 'renewables' && <RenewablesTab data={kpiData} />}
              {activeTab === 'network'    && <NetworkTab data={kpiData} />}
            </>
          )}
        </>
      )}
    </div>
  );
}

export default KPI;
