import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
}

interface TechTreeProps {
  activeConstTypes: string[];
  scenarioName?: string;
  isActive?: boolean;
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

const LAYER_LABELS: Record<number, string> = {
  0: 'L0 · Typology',
  1: 'L1 · Archetypes',
  2: 'L2 · Assemblies',
  5: 'L3 · Energy',
};

// ELK rank per sublayer — L2 sub-layers stacked 2/3/4, L3 stacked 5/6
function sublayerRank(sublayer: string): number {
  const map: Record<string, number> = {
    l0: 0, l1: 1, 'l2.1': 2, 'l2.2': 3, 'l2.3': 4, 'l3.1': 5, 'l3.2': 6,
  };
  return map[sublayer] ?? 1;
}

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

function buildVisibleGraph(
  data: GraphData,
  collapsed: Set<string>,
  activeIds: Set<string>,
): { nodes: Node[]; edges: Edge[] } {
  // Only include active nodes — inactive nodes are excluded entirely from layout
  const idMap = new Map<string, string>();
  const visibleNodes: Node[] = [];
  const pillActiveCounts: Record<string, number> = {};

  for (const n of data.nodes) {
    if (!activeIds.has(n.id)) continue; // skip inactive nodes completely
    if (collapsed.has(n.sublayer)) {
      const pillId = `__pill_${n.sublayer}`;
      idMap.set(n.id, pillId);
      pillActiveCounts[pillId] = (pillActiveCounts[pillId] ?? 0) + 1;
    } else {
      idMap.set(n.id, n.id);
      visibleNodes.push({
        id: n.id,
        type: 'techNode',
        position: { x: 0, y: 0 },
        data: { label: n.label, active: true, sublayer: n.sublayer, layer: n.layer },
      });
    }
  }

  // Add pill nodes only for sub-layers that have at least one active member
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
      },
    });
  }

  // Build edges — only between nodes that are both visible
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
      'elk.layered.spacing.nodeNodeBetweenLayers': '80',
      'elk.spacing.nodeNode': '18',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.partitioning.activate': 'true',
    },
    children: nodes.map((n) => ({
      id: n.id,
      width: n.type === 'sublayerPill' ? 164 : 132,
      height: n.type === 'sublayerPill' ? 44 : 34,
      layoutOptions: {
        'elk.partitioning.partition': String(n.data.layer as number),
      },
    })),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };

  console.debug('[TechTree] ELK input', { nodes: elkGraph.children.length, edges: elkGraph.edges.length });

  const layout = await elk.layout(elkGraph);

  console.debug('[TechTree] ELK layout done', { children: layout.children?.length });

  const posMap = new Map<string, { x: number; y: number }>();
  for (const c of layout.children ?? []) {
    posMap.set(c.id, { x: c.x ?? 0, y: c.y ?? 0 });
  }

  return nodes.map((n) => ({ ...n, position: posMap.get(n.id) ?? n.position }));
}

// ── custom node components ────────────────────────────────────────────────────

function TechNode({ data }: { data: Record<string, unknown> }) {
  const cls = `tech-node${data.active ? ' tech-node--active' : ' tech-node--inactive'}`;
  return (
    <div className={cls} title={data.label as string}>
      <Handle type="target" position={Position.Top} />
      <span>{data.label as string}</span>
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

// ── layer overlay (bands + labels, viewport-aware) ────────────────────────────

const BAND_BG: Record<string, string> = {
  'L0 · Typology':   'rgba(56, 189, 248, 0.05)',
  'L1 · Archetypes': 'rgba(167, 139, 250, 0.05)',
  'L2 · Assemblies': 'rgba(52, 211, 153, 0.05)',
  'L3 · Energy':     'rgba(251, 146, 60,  0.05)',
};
const BAND_BORDER: Record<string, string> = {
  'L0 · Typology':   'rgba(56, 189, 248, 0.25)',
  'L1 · Archetypes': 'rgba(167, 139, 250, 0.25)',
  'L2 · Assemblies': 'rgba(52, 211, 153, 0.25)',
  'L3 · Energy':     'rgba(251, 146, 60,  0.25)',
};
const BAND_TEXT: Record<string, string> = {
  'L0 · Typology':   'rgba(56, 189, 248, 0.75)',
  'L1 · Archetypes': 'rgba(167, 139, 250, 0.75)',
  'L2 · Assemblies': 'rgba(52, 211, 153, 0.75)',
  'L3 · Energy':     'rgba(251, 146, 60,  0.75)',
};

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
            style={{
              position: 'absolute',
              left: 0, right: 0, top, height,
              background:  BAND_BG[band.label]     ?? 'transparent',
              borderTop:  `1px solid ${BAND_BORDER[band.label] ?? 'rgba(255,255,255,0.06)'}`,
            }}
          >
            <span style={{
              position: 'absolute',
              left: 10,
              top: 6,
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: BAND_TEXT[band.label] ?? 'rgba(255,255,255,0.4)',
              fontFamily: 'var(--font-stack, monospace)',
              whiteSpace: 'nowrap',
              userSelect: 'none',
            }}>
              {band.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

function TechTree({ activeConstTypes, scenarioName = '', isActive = false }: TechTreeProps) {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(
    () => new Set(['l2.1', 'l2.2', 'l2.3', 'l3.1', 'l3.2']),
  );
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutReady, setLayoutReady] = useState(false);
  const prevCollapsedRef = useRef<string>('');

  // Fetch graph data once (or when scenario changes)
  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    fetchTechTreeGraph(scenarioName)
      .then((data: GraphData) => {
        if (cancelled) return;
        console.debug('[TechTree] graph loaded', { nodes: data.nodes.length, edges: data.edges.length });
        setGraphData(data);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        console.error('[TechTree] fetch failed', e);
        setLoadError(e.message);
      });
    return () => { cancelled = true; };
  }, [scenarioName]);

  // Active node set derived from graphData + activeConstTypes prop
  const activeIds = useMemo(() => {
    if (!graphData) return new Set<string>();
    const ids = computeActiveNodes(graphData, activeConstTypes);
    console.debug('[TechTree] active nodes', ids.size, [...ids].slice(0, 10));
    return ids;
  }, [graphData, activeConstTypes]);

  // Re-run ELK layout when graphData, collapsed, or activeIds change
  useEffect(() => {
    if (!graphData) return;
    const collapsedKey = [...collapsed].sort().join(',');
    if (collapsedKey === prevCollapsedRef.current && !graphData) return;
    prevCollapsedRef.current = collapsedKey;

    const { nodes: vNodes, edges: vEdges } = buildVisibleGraph(graphData, collapsed, activeIds);
    setLayoutReady(false);

    runElkLayout(vNodes, vEdges)
      .then((laidOut) => {
        setNodes(laidOut);
        setEdges(vEdges);
        setLayoutReady(true);
      })
      .catch((err) => {
        console.error('[TechTree] ELK layout failed:', err);
        setNodes(vNodes);
        setEdges(vEdges);
        setLayoutReady(true);
      });
  }, [graphData, collapsed, activeIds, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (node.type !== 'sublayerPill') return;
      const sublayer = node.data.sublayer as string;
      console.debug('[TechTree] toggle sublayer', sublayer, 'was collapsed:', collapsed.has(sublayer));
      setCollapsed((prev) => {
        const next = new Set(prev);
        if (next.has(sublayer)) next.delete(sublayer);
        else next.add(sublayer);
        return next;
      });
    },
    [collapsed],
  );

  // Layer band Y positions derived from laid-out nodes
  const layerBands = useMemo(() => {
    if (!layoutReady || !nodes.length) return [];
    const bandMap = new Map<number, { minY: number; maxY: number; label: string }>();
    for (const n of nodes) {
      const mainLayer = (n.data.layer as number) <= 1
        ? (n.data.layer as number)
        : (n.data.layer as number) <= 4 ? 2 : 5;
      const label = LAYER_LABELS[mainLayer] ?? '';
      if (!label) continue;
      const existing = bandMap.get(mainLayer);
      const ny = n.position.y;
      if (!existing) bandMap.set(mainLayer, { minY: ny, maxY: ny + 44, label });
      else bandMap.set(mainLayer, { minY: Math.min(existing.minY, ny), maxY: Math.max(existing.maxY, ny + 44), label });
    }
    return [...bandMap.values()];
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

      {!layoutReady && !loadError && (
        <div className="techtree-loading">Computing layout…</div>
      )}
    </div>
  );
}

export default TechTree;
