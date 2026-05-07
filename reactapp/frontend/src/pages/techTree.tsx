import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  useViewport,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import ELK from 'elkjs/lib/elk.bundled.js';
import { fetchTechTreeGraph } from '../services/api.js';
import '../styles/techtree.css';

// ── types ─────────────────────────────────────────────────────────────────────

interface RawNode {
  id: string;
  label: string;
  layer: number;
  sublayer: string;
  description?: string;
  building_count?: number;
}

interface RawEdge {
  id: string;
  source: string;
  target: string;
  field?: string;
}

interface GraphData {
  active_const_types: string[];
  nodes: RawNode[];
  edges: RawEdge[];
  total_buildings?: number;
  has_assembly_data?: boolean;
}

interface TechTreeProps {
  activeConstTypes: string[];
  scenarioName?: string;
  isActive?: boolean;
  buildingCountByConstType?: Record<string, number>;
  totalBuildings?: number;
}

interface FilterSectionData {
  id: string;
  label: string;
  nodes: RawNode[];
}

// ── constants ─────────────────────────────────────────────────────────────────

const SUBLAYER_LABELS: Record<string, string> = {
  'l2.1': 'Envelope',
  'l2.2': 'HVAC',
  'l2.3': 'Supply',
  'l3.1': 'Conversion',
  'l3.2': 'Feedstocks',
};

const SUBLAYER_COLORS: Record<string, string> = {
  'l2.1': '#2d6a4f',
  'l2.2': '#1d3557',
  'l2.3': '#c46b00',
  'l3.1': '#7b2d2d',
  'l3.2': '#7a6000',
};

function sublayerRank(sublayer: string): number {
  const map: Record<string, number> = {
    l0: 0, l1: 1, 'l2.1': 2, 'l2.2': 3, 'l2.3': 4, 'l3.1': 5, 'l3.2': 6,
  };
  return map[sublayer] ?? 1;
}

const LAYER_LABELS: Record<number, string> = {
  0: 'L0 · Typology',
  1: 'L1 · Archetypes',
  2: 'L2.1 · Envelope',
  3: 'L2.2 · HVAC',
  4: 'L2.3 · Supply',
  5: 'L3.1 · Conversion',
  6: 'L3.2 · Feedstocks',
};

const BAND_BG: Record<string, string> = {
  'L0 · Typology':     'rgba(56,  189, 248, 0.05)',
  'L1 · Archetypes':   'rgba(167, 139, 250, 0.05)',
  'L2.1 · Envelope':   'rgba(52,  211, 153, 0.06)',
  'L2.2 · HVAC':       'rgba(29,   53,  87, 0.08)',
  'L2.3 · Supply':     'rgba(196, 107,   0, 0.06)',
  'L3.1 · Conversion': 'rgba(239,  68,  68, 0.05)',
  'L3.2 · Feedstocks': 'rgba(251, 191,  36, 0.05)',
};
const BAND_BORDER: Record<string, string> = {
  'L0 · Typology':     'rgba(56,  189, 248, 0.25)',
  'L1 · Archetypes':   'rgba(167, 139, 250, 0.25)',
  'L2.1 · Envelope':   'rgba(52,  211, 153, 0.30)',
  'L2.2 · HVAC':       'rgba(29,   53,  87, 0.50)',
  'L2.3 · Supply':     'rgba(196, 107,   0, 0.30)',
  'L3.1 · Conversion': 'rgba(239,  68,  68, 0.25)',
  'L3.2 · Feedstocks': 'rgba(251, 191,  36, 0.25)',
};
const BAND_TEXT: Record<string, string> = {
  'L0 · Typology':     'rgba(56,  189, 248, 0.75)',
  'L1 · Archetypes':   'rgba(167, 139, 250, 0.75)',
  'L2.1 · Envelope':   'rgba(52,  211, 153, 0.75)',
  'L2.2 · HVAC':       'rgba(96,  165, 250, 0.75)',
  'L2.3 · Supply':     'rgba(251, 146,  60, 0.75)',
  'L3.1 · Conversion': 'rgba(248, 113, 113, 0.75)',
  'L3.2 · Feedstocks': 'rgba(251, 191,  36, 0.75)',
};

const EXPANDED_NODE_HEIGHT = 130;
const NORMAL_NODE_HEIGHT   = 34;
const PILL_HEIGHT          = 44;

const elk = new ELK();

// ── pure helpers ──────────────────────────────────────────────────────────────

function computeActiveNodes(data: GraphData, activeConstTypes: string[]): Set<string> {
  const active = new Set<string>([...data.active_const_types, ...activeConstTypes]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const e of data.edges) {
      if (active.has(e.source) && !active.has(e.target)) {
        active.add(e.target);
        changed = true;
      }
    }
  }
  return active;
}

/** BFS from a single node in both directions through raw edges. */
function computePathNodes(nodeId: string, edges: RawEdge[]): Set<string> {
  const visited = new Set([nodeId]);
  const queue = [nodeId];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const e of edges) {
      if (e.source === cur && !visited.has(e.target)) { visited.add(e.target); queue.push(e.target); }
      if (e.target === cur && !visited.has(e.source)) { visited.add(e.source); queue.push(e.source); }
    }
  }
  return visited;
}

/**
 * Downward-only BFS from filter-selected nodes.
 * Bidirectional BFS causes the entire graph to be included via shared L1 assemblies
 * (e.g. selecting one L0 typology walks up through shared assemblies to all other
 * typologies, then cascades down again → 1000+ nodes → ELK freeze).
 */
function computeFilterExpanded(filterIds: Set<string>, edges: RawEdge[]): Set<string> {
  const visited = new Set(filterIds);
  const queue = [...filterIds];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const e of edges) {
      if (e.source === cur && !visited.has(e.target)) { visited.add(e.target); queue.push(e.target); }
    }
  }
  return visited;
}

function buildVisibleGraph(
  data: GraphData,
  collapsed: Set<string>,
  activeIds: Set<string>,
  expandedNodeId: string | null,
  filterExpanded: Set<string> | null,
  buildingCounts: Record<string, number> = {},
  totalBuildings = 0,
  hasAssemblyData = false,
): { nodes: Node[]; edges: Edge[] } {
  // When filter is active, show the filter-expanded set; otherwise show activeIds only
  const visibleSet = filterExpanded ?? activeIds;

  const idMap = new Map<string, string>();
  const visibleNodes: Node[] = [];
  const pillActiveCounts: Record<string, number> = {};

  for (const n of data.nodes) {
    if (!visibleSet.has(n.id)) continue;
    const buildingCount = n.building_count ?? buildingCounts[n.id] ?? 0;
    if (hasAssemblyData && buildingCount === 0) continue;
    if (collapsed.has(n.sublayer)) {
      const pillId = `__pill_${n.sublayer}`;
      idMap.set(n.id, pillId);
      pillActiveCounts[pillId] = (pillActiveCounts[pillId] ?? 0) + 1;
    } else {
      const expanded = n.id === expandedNodeId;
      idMap.set(n.id, n.id);
      visibleNodes.push({
        id: n.id,
        type: 'techNode',
        position: { x: 0, y: 0 },
        data: {
          label: n.label,
          description: n.description ?? '',
          active: activeIds.has(n.id),
          sublayer: n.sublayer,
          layer: n.layer,
          expanded,
          buildingCount,
          totalBuildings,
        },
      });
    }
  }

  for (const [pillId, activeCount] of Object.entries(pillActiveCounts)) {
    const sublayer = pillId.replace('__pill_', '');
    visibleNodes.push({
      id: pillId,
      type: 'sublayerPill',
      position: { x: 0, y: 0 },
      data: {
        label: `${SUBLAYER_LABELS[sublayer] ?? sublayer} × ${activeCount}`,
        sublayer,
        active: true,
        layer: sublayerRank(sublayer),
        expanded: false,
      },
    });
  }

  const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));
  const edgeSet = new Set<string>();
  const visibleEdges: Edge[] = [];

  for (const e of data.edges) {
    const src = idMap.get(e.source);
    const tgt = idMap.get(e.target);
    if (!src || !tgt || src === tgt) continue;
    if (!visibleNodeIds.has(src) || !visibleNodeIds.has(tgt)) continue;
    const key = `${src}--${tgt}`;
    if (edgeSet.has(key)) continue;
    edgeSet.add(key);
    visibleEdges.push({
      id: `ve_${key}`,
      source: src,
      target: tgt,
      style: { stroke: 'var(--accent, #4fc)', strokeWidth: 1.5 },
    });
  }

  return { nodes: visibleNodes, edges: visibleEdges };
}

async function runElkLayout(nodes: Node[], edges: Edge[]): Promise<Node[]> {
  const elkGraph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.layered.spacing.nodeNodeBetweenLayers': '60',
      'elk.spacing.nodeNode': '18',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.partitioning.activate': 'true',
    },
    children: nodes.map((n) => ({
      id: n.id,
      width: n.type === 'sublayerPill' ? 164 : 132,
      height: n.type === 'sublayerPill'
        ? PILL_HEIGHT
        : n.data.expanded ? EXPANDED_NODE_HEIGHT : NORMAL_NODE_HEIGHT,
      layoutOptions: {
        'elk.partitioning.partition': String(n.data.layer as number),
      },
    })),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };

  const layout = await elk.layout(elkGraph);
  const posMap = new Map<string, { x: number; y: number }>();
  for (const c of layout.children ?? []) posMap.set(c.id, { x: c.x ?? 0, y: c.y ?? 0 });
  return nodes.map((n) => ({ ...n, position: posMap.get(n.id) ?? n.position }));
}

// ── custom node components ────────────────────────────────────────────────────

function TechNode({ data }: { data: Record<string, unknown> }) {
  const expanded      = data.expanded      as boolean;
  const dimmed        = data.dimmed        as boolean | undefined;
  const desc          = (data.description  as string) || '';
  const buildingCount = (data.buildingCount as number) ?? 0;
  const total         = (data.totalBuildings as number) ?? 0;

  const cls = [
    'tech-node',
    data.active ? 'tech-node--active' : 'tech-node--inactive',
    expanded ? 'tech-node--expanded' : '',
    dimmed   ? 'tech-node--dimmed'   : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={cls} title={expanded ? '' : (data.label as string)}>
      <Handle type="target" position={Position.Top} />
      <span className="tech-node__label">{data.label as string}</span>
      {expanded && (
        <>
          {buildingCount > 0 && (
            <span className="tech-node__count">
              {buildingCount} / {total} buildings
            </span>
          )}
          <div className={`tech-node__desc${!desc ? ' tech-node__desc--empty' : ''}`}>
            {desc || 'No description available.'}
          </div>
        </>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function SublayerPill({ data }: { data: Record<string, unknown> }) {
  const sublayer = data.sublayer as string;
  const color = SUBLAYER_COLORS[sublayer] ?? '#444';
  return (
    <div
      className={`sublayer-pill${data.active ? ' sublayer-pill--active' : ''}`}
      style={{ borderColor: color }}
      title={`Click to expand ${SUBLAYER_LABELS[sublayer] ?? sublayer}`}
    >
      <Handle type="target" position={Position.Top} />
      <span className="sublayer-pill__chevron">▶</span>
      <span>{data.label as string}</span>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const NODE_TYPES = { techNode: TechNode, sublayerPill: SublayerPill };

// ── layer overlays (viewport-aware bands) ─────────────────────────────────────

interface Band { label: string; minY: number; maxY: number }

function LayerOverlays({ bands }: { bands: Band[] }) {
  const { y: vpY, zoom } = useViewport();
  if (!bands.length) return null;
  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
      {bands.map((band) => {
        const top    = band.minY * zoom + vpY;
        const height = Math.max(0, (band.maxY - band.minY) * zoom);
        return (
          <div
            key={band.label}
            className="layer-band"
            style={{
              top, height,
              background: BAND_BG[band.label]   ?? 'transparent',
              borderTop: `1px solid ${BAND_BORDER[band.label] ?? 'rgba(255,255,255,0.06)'}`,
            }}
          >
            <span
              className="layer-band__label"
              style={{ color: BAND_TEXT[band.label] ?? 'rgba(255,255,255,0.4)' }}
            >
              {band.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── filter panel ──────────────────────────────────────────────────────────────

function TechFilterPanel({
  sections,
  selection,
  onToggle,
  onClear,
}: {
  sections: FilterSectionData[];
  selection: Set<string>;
  onToggle: (id: string) => void;
  onClear: () => void;
}) {
  const [openSections, setOpenSections] = useState<Set<string>>(() => new Set(['l0']));
  const [searches, setSearches] = useState<Record<string, string>>({});

  const toggleSection = useCallback((id: string) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const totalSelected = selection.size;

  return (
    <div className="tt-filter-panel">
      <div className="tt-filter-header">
        <span className="tt-filter-title">Filter graph</span>
        {totalSelected > 0 && (
          <button className="tt-filter-clear" onClick={onClear} title="Clear all filters">
            ✕ {totalSelected}
          </button>
        )}
      </div>

      <div className="tt-filter-body">
        {sections.map((sec) => {
          const isOpen = openSections.has(sec.id);
          const search = searches[sec.id] ?? '';
          const filtered = search
            ? sec.nodes.filter(
                (n) =>
                  n.id.toLowerCase().includes(search.toLowerCase()) ||
                  n.label.toLowerCase().includes(search.toLowerCase()) ||
                  (n.description ?? '').toLowerCase().includes(search.toLowerCase()),
              )
            : sec.nodes;
          const selectedInSection = sec.nodes.filter((n) => selection.has(n.id)).length;

          return (
            <div key={sec.id} className="tt-filter-section">
              <button
                className={`tt-filter-section-header${isOpen ? ' tt-filter-section-header--open' : ''}`}
                onClick={() => toggleSection(sec.id)}
              >
                <span className="tt-filter-chevron">{isOpen ? '▼' : '▶'}</span>
                <span className="tt-filter-section-label">{sec.label}</span>
                <span className="tt-filter-section-count">
                  {selectedInSection > 0 && <span className="tt-filter-badge">{selectedInSection}</span>}
                  {sec.nodes.length}
                </span>
              </button>

              {isOpen && (
                <div className="tt-filter-section-body">
                  {sec.nodes.length > 8 && (
                    <input
                      className="tt-filter-search"
                      type="text"
                      placeholder="Search…"
                      value={search}
                      onChange={(e) =>
                        setSearches((prev) => ({ ...prev, [sec.id]: e.target.value }))
                      }
                    />
                  )}
                  <div className="tt-filter-list">
                    {filtered.map((n) => (
                      <label key={n.id} className="tt-filter-item" title={n.description ?? ''}>
                        <input
                          type="checkbox"
                          checked={selection.has(n.id)}
                          onChange={() => onToggle(n.id)}
                        />
                        <span className="tt-filter-item-id">{n.id}</span>
                        {n.description && (
                          <span className="tt-filter-item-desc">{n.description}</span>
                        )}
                      </label>
                    ))}
                    {filtered.length === 0 && (
                      <span className="tt-filter-empty">No matches</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

function TechTree({
  activeConstTypes,
  scenarioName = '',
  isActive = false,
  buildingCountByConstType = {},
  totalBuildings = 0,
}: TechTreeProps) {
  const [graphData, setGraphData]   = useState<GraphData | null>(null);
  const [loadError, setLoadError]   = useState<string | null>(null);
  const [collapsed, setCollapsed]   = useState<Set<string>>(
    () => new Set(['l2.1', 'l2.2', 'l2.3', 'l3.1', 'l3.2']),
  );
  const [expandedNode, setExpandedNode]       = useState<string | null>(null);
  const [highlightedPath, setHighlightedPath] = useState<Set<string> | null>(null);
  const [filterSelection, setFilterSelection] = useState<Set<string>>(() => new Set());
  const [filterOpen, setFilterOpen]           = useState(false);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutReady, setLayoutReady]   = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    fetchTechTreeGraph(scenarioName)
      .then((data: GraphData) => { if (!cancelled) setGraphData(data); })
      .catch((e: Error)       => { if (!cancelled) setLoadError(e.message); });
    return () => { cancelled = true; };
  }, [scenarioName]);

  const activeIds = useMemo(() => {
    if (!graphData) return new Set<string>();
    return computeActiveNodes(graphData, activeConstTypes);
  }, [graphData, activeConstTypes]);

  // Pre-compute filter expanded set (multi-source BFS through full graph)
  const filterExpanded = useMemo(() => {
    if (filterSelection.size === 0 || !graphData) return null;
    return computeFilterExpanded(filterSelection, graphData.edges);
  }, [filterSelection, graphData]);

  // Build filter panel sections — scoped to activeIds so only nodes present in the
  // current building selection appear as filter options.
  const filterSections = useMemo((): FilterSectionData[] => {
    if (!graphData) return [];
    const nodeById = new Map(graphData.nodes.map((n) => [n.id, n]));
    const hasBuildings = (n: RawNode) => !graphData.has_assembly_data || (n.building_count ?? 0) > 0;

    // Map L1 node → its connected L2 family
    const l1ToFamily = new Map<string, string>();
    for (const e of graphData.edges) {
      const src = nodeById.get(e.source);
      const tgt = nodeById.get(e.target);
      if (src?.sublayer === 'l1' && tgt?.sublayer?.startsWith('l2')) {
        if (!l1ToFamily.has(e.source)) l1ToFamily.set(e.source, e.target);
      }
    }

    const l0Nodes  = graphData.nodes.filter((n) => n.sublayer === 'l0'          && activeIds.has(n.id) && hasBuildings(n));
    const l2Nodes  = graphData.nodes.filter((n) => n.sublayer.startsWith('l2')  && activeIds.has(n.id) && hasBuildings(n));
    const l31Nodes = graphData.nodes.filter((n) => n.sublayer === 'l3.1'        && activeIds.has(n.id) && hasBuildings(n));
    const l32Nodes = graphData.nodes.filter((n) => n.sublayer === 'l3.2'        && activeIds.has(n.id) && hasBuildings(n));

    const sections: FilterSectionData[] = [
      { id: 'l0', label: 'L0 · Typology', nodes: l0Nodes },
      { id: 'l2', label: 'L2 · Assembly families', nodes: l2Nodes },
    ];

    // L1 grouped by L2 family — only active L1 nodes with buildings
    for (const familyNode of l2Nodes) {
      const l1InFamily = graphData.nodes.filter(
        (n) => n.sublayer === 'l1' && activeIds.has(n.id) && hasBuildings(n) && l1ToFamily.get(n.id) === familyNode.id,
      );
      if (l1InFamily.length > 0) {
        sections.push({ id: `l1_${familyNode.id}`, label: `L1 · ${familyNode.label}`, nodes: l1InFamily });
      }
    }

    sections.push(
      { id: 'l3_1', label: 'L3.1 · Conversion',  nodes: l31Nodes },
      { id: 'l3_2', label: 'L3.2 · Feedstocks',  nodes: l32Nodes },
    );
    return sections;
  }, [graphData, activeIds]);

  // ELK layout — re-runs when structure, expansion, or filter changes
  useEffect(() => {
    if (!graphData) return;
    const effectiveTotalBuildings = graphData.total_buildings || totalBuildings;
    const { nodes: vNodes, edges: vEdges } = buildVisibleGraph(
      graphData, collapsed, activeIds, expandedNode, filterExpanded,
      buildingCountByConstType, effectiveTotalBuildings, graphData.has_assembly_data,
    );
    setLayoutReady(false);
    runElkLayout(vNodes, vEdges)
      .then((laidOut) => { setNodes(laidOut); setEdges(vEdges); setLayoutReady(true); })
      .catch((err) => {
        console.error('[TechTree] ELK layout failed:', err);
        setNodes(vNodes); setEdges(vEdges); setLayoutReady(true);
      });
  }, [graphData, collapsed, activeIds, expandedNode, filterExpanded, buildingCountByConstType, totalBuildings, setNodes, setEdges]);

  // Apply highlight styles without re-running ELK
  useEffect(() => {
    if (!layoutReady) return;
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, dimmed: highlightedPath ? !highlightedPath.has(n.id) : false },
      })),
    );
    setEdges((eds) =>
      eds.map((e) => {
        const inPath =
          !highlightedPath ||
          (highlightedPath.has(e.source) && highlightedPath.has(e.target));
        return {
          ...e,
          style: {
            stroke:      inPath ? 'var(--accent, #4fc)' : 'rgba(255,255,255,0.06)',
            strokeWidth: inPath ? 2 : 1,
            opacity:     inPath ? 1 : 0.25,
          },
        };
      }),
    );
  }, [highlightedPath, layoutReady, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (node.type === 'sublayerPill') {
        const sublayer = node.data.sublayer as string;
        setCollapsed((prev) => {
          const next = new Set(prev);
          if (next.has(sublayer)) next.delete(sublayer); else next.add(sublayer);
          return next;
        });
        return;
      }
      if (node.type === 'techNode') {
        const id = node.id;
        if (expandedNode === id) {
          setExpandedNode(null);
          setHighlightedPath(null);
        } else {
          setExpandedNode(id);
          if (graphData) setHighlightedPath(computePathNodes(id, graphData.edges));
        }
      }
    },
    [expandedNode, graphData],
  );

  const handleFilterToggle = useCallback((id: string) => {
    setFilterSelection((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const handleFilterClear = useCallback(() => {
    setFilterSelection(new Set());
  }, []);

  // Layer bands per ELK rank (0-6)
  const layerBands = useMemo(() => {
    if (!layoutReady || !nodes.length) return [];
    const bandMap = new Map<number, { minY: number; maxY: number; label: string }>();
    for (const n of nodes) {
      const rank  = n.data.layer as number;
      const label = LAYER_LABELS[rank] ?? '';
      if (!label) continue;
      const nh = (n.data.expanded as boolean) ? EXPANDED_NODE_HEIGHT : PILL_HEIGHT;
      const ny = n.position.y;
      const existing = bandMap.get(rank);
      if (!existing) bandMap.set(rank, { minY: ny, maxY: ny + nh, label });
      else bandMap.set(rank, {
        minY: Math.min(existing.minY, ny),
        maxY: Math.max(existing.maxY, ny + nh),
        label,
      });
    }
    return [...bandMap.values()].sort((a, b) => a.minY - b.minY);
  }, [nodes, layoutReady]);

  if (loadError) {
    return (
      <div className="techtree-root techtree-error">
        <p>Failed to load tech tree: {loadError}</p>
      </div>
    );
  }

  return (
    <div className="techtree-root">
      {isActive && (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          minZoom={0.05}
          nodesDraggable={false}
          nodesConnectable={false}
        >
          <LayerOverlays bands={layerBands} />
          <Background gap={24} size={1} color="var(--color-bg-3, #2a2a2a)" />
          <Controls />
        </ReactFlow>
      )}

      {/* Filter toggle button */}
      <button
        className={`tt-filter-toggle${filterOpen ? ' tt-filter-toggle--open' : ''}${filterSelection.size > 0 ? ' tt-filter-toggle--active' : ''}`}
        onClick={() => setFilterOpen((v) => !v)}
        title="Filter graph"
      >
        ⚗{filterSelection.size > 0 && <span className="tt-filter-toggle-badge">{filterSelection.size}</span>}
      </button>

      {filterOpen && (
        <TechFilterPanel
          sections={filterSections}
          selection={filterSelection}
          onToggle={handleFilterToggle}
          onClear={handleFilterClear}
        />
      )}

      {!layoutReady && !loadError && (
        <div className="techtree-loading">Computing layout…</div>
      )}
    </div>
  );
}

export default TechTree;
