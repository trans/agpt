// microGPT Construction Kit — Graph Editor with Hierarchical Zoom
// Vanilla JS + SVG, D3 zoom for pan/zoom

'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const starSvg = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>';

// ── Helpers ──────────────────────────────────────────────────────────────────

async function apiPost(endpoint, data, method = 'POST') {
  const resp = await fetch(endpoint, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return await resp.json();
}

async function fetchJSON(url, fallback = null) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return fallback;
    return await resp.json();
  } catch { return fallback; }
}

function createPortShape(rank, px, py, r, rounding) {
  let shape;
  if (rank >= 2) {
    const size = r * 2;
    shape = document.createElementNS(SVG_NS, 'rect');
    shape.setAttribute('x', px - size / 2);
    shape.setAttribute('y', py - size / 2);
    shape.setAttribute('width', size);
    shape.setAttribute('height', size);
    shape.setAttribute('rx', rounding ?? 2);
    shape.setAttribute('ry', rounding ?? 2);
  } else if (rank === 0) {
    shape = document.createElementNS(SVG_NS, 'polygon');
    shape.setAttribute('points', `${px},${py - r} ${px + r},${py} ${px},${py + r} ${px - r},${py}`);
  } else {
    shape = document.createElementNS(SVG_NS, 'circle');
    shape.setAttribute('cx', px);
    shape.setAttribute('cy', py);
    shape.setAttribute('r', r);
  }
  return shape;
}

function ensureBoundaryOverlay() {
  const wrap = document.getElementById('canvas-wrap');
  const svgEl = document.getElementById('canvas');
  let overlay = document.getElementById('boundary-ports-overlay');
  if (!overlay) {
    overlay = document.createElementNS(SVG_NS, 'svg');
    overlay.id = 'boundary-ports-overlay';
    overlay.style.cssText = 'position:absolute;left:0;right:0;bottom:0;z-index:15;pointer-events:none;';
    wrap.appendChild(overlay);
  }
  const svgRect = svgEl.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  overlay.style.top = (svgRect.top - wrapRect.top) + 'px';
  overlay.style.width = svgRect.width + 'px';
  overlay.style.height = svgRect.height + 'px';
  overlay.innerHTML = '';
  return { overlay, h: svgRect.height, w: svgRect.width };
}

// ── Scope / Level Definitions ────────────────────────────────────────────────
// Each scope level defines what components are available in the palette
// and what the "container" types are (double-click to zoom in).

const SCOPE_LEVELS = {
  project: {
    label: "Project",
    components: ["engine"],
    containers: { engine: "pipeline" },  // double-click engine → enter its pipeline
  },
  pipeline: {
    label: "Pipeline",
    components: ["char_tokenizer", "bpe_tokenizer", "sequential_window", "sliding_window", "random_window", "random_init", "zero_init", "learned_init", "cooperative", "loss", "optimizer"],
    containers: { cooperative: "ensemble" },  // double-click cooperative → enter ensemble scope
  },
  ensemble: {
    label: "Ensemble",
    components: ["transformer", "counter", "bigram", "global_router", "context_router", "gated_router", "stream_projector", "optimizer"],
    containers: { transformer: "transformer_internal", counter: "expert_internal", global_router: "global_router_internal" },
  },
  transformer_internal: {
    label: "Transformer",
    components: ["embedding", "attention_layer", "ffn_layer", "layer_norm", "output_head", "stream_proj_internal", "optimizer"],
    containers: { output_head: "output_head_internal", stream_proj_internal: "stream_proj_math", embedding: "embedding_internal", ffn_layer: "ffn_internal", layer_norm: "layer_norm_internal" },
  },
  expert_internal: {
    label: "Expert",
    components: ["embedding", "ffn_layer", "layer_norm", "output_head", "stream_proj_internal", "optimizer"],
    containers: { output_head: "output_head_internal", stream_proj_internal: "stream_proj_math", embedding: "embedding_internal", ffn_layer: "ffn_internal", layer_norm: "layer_norm_internal" },
  },
  output_head_internal: {
    label: "Output Head",
    components: ["matmul", "add_bias", "weight_param", "bias_param"],
    containers: {},
  },
  stream_proj_math: {
    label: "Stream Projection",
    components: ["matmul", "add_bias", "weight_param", "bias_param"],
    containers: {},
  },
  embedding_internal: {
    label: "Embedding",
    components: ["lookup", "embedding_table"],
    containers: {},
  },
  ffn_internal: {
    label: "Feed-Forward",
    components: ["matmul", "add_bias", "relu", "weight_param", "bias_param"],
    containers: {},
  },
  global_router_internal: {
    label: "Global Router",
    components: ["mean_pool", "matmul", "add_bias", "weight_param", "bias_param", "softmax", "eps_blend", "broadcast"],
    containers: {},
  },
  layer_norm_internal: {
    label: "Layer Norm",
    components: ["reduce_mean", "subtract", "reduce_var", "normalize", "scale_param", "elem_mul", "bias_param", "add_bias"],
    containers: {},
  },
};

// ── State ────────────────────────────────────────────────────────────────────

const MAX_CARDS = 8;
const cards = [];       // Array of card objects (each owns a graph + engine state)
let activeCardIdx = 0;

// Training session — shown as a special card at the top of the rail
// { cards: [...participating cards], active: bool }
let trainSession = null;
const SESSION_COLORS = ['#4a90d9', '#e6994a', '#7bc67e', '#d94a7b', '#c44ad9', '#4ad9b8', '#d9c84a', '#d94a4a'];

// Project-level graph — shared across all engines
const projectGraph = { nodes: [], edges: [] };
let projectNextId = 1;

function createEmptyCard(name) {
  const nodeId = projectNextId++;
  const node = {
    id: nodeId,
    type: 'engine',
    x: (projectGraph.nodes.length % 4) * 250,
    y: Math.floor(projectGraph.nodes.length / 4) * 200,
    params: {},
    children: { nodes: [], edges: [] },
  };
  projectGraph.nodes.push(node);
  return {
    id: crypto.randomUUID(),
    name: name || '',
    starred: true,
    nodeId: nodeId,  // links to engine node in projectGraph
    engine: {
      built: false,
      training: false,
      lossHistory: [],
      modelHash: null,
      totalSteps: 0,
    },
    graphMode: false,
  };
}

// Check if a node type is a container in ANY scope (not just the current one)
function getChildScope(nodeType) {
  // Check current scope first
  const current = SCOPE_LEVELS[state.currentScope]?.containers[nodeType];
  if (current) return current;
  // Fall back to any scope
  for (const key of Object.keys(SCOPE_LEVELS)) {
    const s = SCOPE_LEVELS[key].containers[nodeType];
    if (s) return s;
  }
  return null;
}

function findNodeRecursive(graph, nodeId) {
  for (const n of graph.nodes) {
    if (n.id === nodeId) return n;
    if (n.children) {
      const found = findNodeRecursive(n.children, nodeId);
      if (found) return found;
    }
  }
  return null;
}

function getActiveCard() { return cards[activeCardIdx]; }
function isActiveLocked() { return !!getActiveCard()?.engine?.training; }
function getCardForNode(nodeId) { return cards.find(c => c.nodeId === nodeId); }

const state = {
  registry: null,       // component definitions from components.json
  // Current view references
  nodes: [],            // current scope's nodes
  edges: [],            // current scope's edges
  selected: null,       // node id
  // Scope navigation
  scopeStack: [],       // array of {nodeId, nodeLabel, scopeType}
  currentScope: 'project',
  // Interaction
  dragging: null,
  connecting: null,
  // View transform (kept in sync with D3)
  viewX: 0, viewY: 0, zoom: 1,
};

// rootGraph is always the project graph
Object.defineProperty(state, 'rootGraph', {
  get() { return projectGraph; },
});
Object.defineProperty(state, 'nextId', {
  get() { return projectNextId; },
  set(v) { projectNextId = v; },
});

// ── DOM refs ─────────────────────────────────────────────────────────────────

const canvas     = document.getElementById('canvas');
const nodesLayer = document.getElementById('nodes-layer');
const edgesLayer = document.getElementById('edges-layer');
const draftEdge  = document.getElementById('draft-edge');
const paletteList = document.getElementById('palette-list');
// props-content removed — properties now rendered in topbar
const statusEl   = document.getElementById('status');
const scopeInfoEl = document.getElementById('scope-info');

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  const resp = await fetch('components.json');
  state.registry = await resp.json();

  // Try to restore from localStorage, otherwise create demo
  if (!restoreProject()) {
    const card = createEmptyCard('Demo');
    cards.push(card);
    activeCardIdx = 0;
    state.nodes = projectGraph.nodes;
    state.edges = projectGraph.edges;
    state.currentScope = 'project';
    loadDemo();
  }

  buildPalette();
  setupD3Zoom();
  setupCanvasEvents();
  setupCards();
  setupLeftPanel();
  setupChat();
  renderBreadcrumbs();
  updateStatus();
  renderTree();
  renderCardList();
  _autoSaveEnabled = true;
}

// ── Palette (scope-aware) ────────────────────────────────────────────────────

function buildPalette() {
  const level = SCOPE_LEVELS[state.currentScope];
  const allowedTypes = level ? level.components : state.registry.components.map(c => c.type);

  const byCategory = {};
  const allComps = [...state.registry.components, ENGINE_COMPONENT];
  for (const comp of allComps) {
    if (!allowedTypes.includes(comp.type)) continue;
    (byCategory[comp.category] = byCategory[comp.category] || []).push(comp);
  }
  paletteList.innerHTML = '';
  for (const [cat, comps] of Object.entries(byCategory)) {
    const section = document.createElement('div');
    section.className = 'palette-section';
    section.innerHTML = `<h3>${cat}</h3>`;
    for (const comp of comps) {
      const item = document.createElement('div');
      item.className = 'palette-item';
      item.dataset.type = comp.type;
      const containerHint = level?.containers[comp.type] ? ' <span style="opacity:0.4;font-size:10px">[zoom in]</span>' : '';
      item.innerHTML = `<div class="palette-dot" style="background:${comp.color}"></div><span>${comp.label}${containerHint}</span>`;
      item.draggable = true;
      item.addEventListener('dragstart', onPaletteDragStart);
      section.appendChild(item);
    }
    paletteList.appendChild(section);
  }
}

// ── Model Tree ──────────────────────────────────────────────────────────────

const treeExpanded = new Set(); // track manually expanded nodes

function renderTree() {
  const container = document.getElementById('tree-content');
  if (!container) return;
  container.innerHTML = '';

  // Auto-expand nodes in the current scope path
  for (const s of state.scopeStack) treeExpanded.add(s.nodeId);

  renderTreeLevel(container, state.rootGraph.nodes, 'project', 0, []);
}

function renderTreeLevel(parentEl, nodes, scopeType, depth, scopePath) {
  const level = SCOPE_LEVELS[scopeType];

  for (const node of nodes) {
    const comp = getCompDef(node.type);
    if (!comp) continue;

    const childScope = getChildScope(node.type);
    const isContainer = !!childScope;
    const isExpanded = treeExpanded.has(node.id);
    const isInPath = state.scopeStack.some(s => s.nodeId === node.id);
    const isCurrentScope = state.nodes.includes(node);
    const isSelected = state.selected === node.id && isCurrentScope;

    const row = document.createElement('div');
    row.className = 'tree-node' + (isSelected ? ' active' : '');
    row.style.paddingLeft = (12 + depth * 16) + 'px';

    // Toggle arrow
    const toggle = document.createElement('span');
    toggle.className = 'tree-toggle';
    if (isContainer) {
      toggle.textContent = isExpanded ? '\u25BE' : '\u25B8';
      toggle.style.cursor = 'pointer';
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        if (treeExpanded.has(node.id)) treeExpanded.delete(node.id);
        else treeExpanded.add(node.id);
        renderTree();
      });
    }
    row.appendChild(toggle);

    // Color dot
    const dot = document.createElement('span');
    dot.className = 'tree-dot';
    dot.style.background = comp.color;
    row.appendChild(dot);

    // Label
    const label = document.createElement('span');
    label.className = 'tree-label';
    if (node.type === 'engine') {
      const card = getCardForNode(node.id);
      label.textContent = card?.name || 'Engine';
    } else {
      label.textContent = node.params?.name || comp.label;
    }
    if (isInPath) {
      label.style.color = comp.color;
      label.style.fontWeight = '600';
    }
    row.appendChild(label);

    // Click label to navigate to scope containing this node & select it
    row.addEventListener('click', (e) => {
      e.stopPropagation();
      navigateToNodeInTree(node.id, scopePath);
    });

    // Double-click to enter container
    if (isContainer) {
      row.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        // Navigate to parent scope first, then zoom in
        navigateToNodeInTree(node.id, scopePath);
        zoomIntoNode(node.id);
      });
    }

    parentEl.appendChild(row);

    // Render children if expanded
    if (isContainer && isExpanded) {
      // Ensure children exist
      if (!node.children) {
        node.children = { nodes: [], edges: [] };
        const defaults = getDefaultChildren(node.type, node.params || {});
        if (defaults) {
          node.children = defaults;
          for (const n of node.children.nodes) {
            if (n.id >= state.nextId) state.nextId = n.id + 1;
          }
          for (const e of node.children.edges) {
            if (e.id >= state.nextId) state.nextId = e.id + 1;
          }
        }
      }
      if (node.children.nodes.length > 0) {
        const childPath = [...scopePath, { nodeId: node.id, scopeType: childScope }];
        renderTreeLevel(parentEl, node.children.nodes, childScope, depth + 1, childPath);
      }
    }
  }
}

function navigateToNodeInTree(nodeId, scopePath) {
  // Rebuild scope stack from the path
  if (scopePath.length === 0) {
    state.scopeStack = [];
    state.nodes = state.rootGraph.nodes;
    state.edges = state.rootGraph.edges;
    state.currentScope = 'project';
  } else {
    state.scopeStack = scopePath.map(sp => {
      // Walk the graph to find this node
      let graph = state.rootGraph;
      for (const prev of scopePath) {
        if (prev.nodeId === sp.nodeId) break;
        const n = graph.nodes.find(nd => nd.id === prev.nodeId);
        if (n && n.children) graph = n.children;
      }
      const n = graph.nodes.find(nd => nd.id === sp.nodeId);
      const comp = getCompDef(n?.type);
      let nodeLabel;
      if (n?.type === 'engine') {
        const card = getCardForNode(n.id);
        nodeLabel = card?.name || 'Engine';
      } else {
        nodeLabel = n?.params?.name || ((comp?.label || '?') + ' #' + sp.nodeId);
      }
      return {
        nodeId: sp.nodeId,
        nodeLabel,
        scopeType: sp.scopeType,
      };
    });

    let graph = state.rootGraph;
    let scopeType = 'project';
    for (const scope of state.scopeStack) {
      const node = graph.nodes.find(n => n.id === scope.nodeId);
      if (node && node.children) {
        graph = node.children;
        scopeType = scope.scopeType;
      }
    }
    state.nodes = graph.nodes;
    state.edges = graph.edges;
    state.currentScope = scopeType;
  }

  state.selected = nodeId;
  buildPalette();
  renderBreadcrumbs();
  updateViewportBorder();
  renderAll();
  fitToView();
  updateStatus();
}

// ── Breadcrumbs ──────────────────────────────────────────────────────────────

function renderBreadcrumbs() {
  const el = document.getElementById('breadcrumbs');

  let html = '';

  // Back button when inside a scope
  if (state.scopeStack.length > 0) {
    html += `<span class="crumb-back" id="back-btn" title="Go back (Esc)">\u21B0</span>`;
  }

  html += `<span class="crumb clickable" data-depth="-1">Project</span>`;
  // Walk the scope stack to find each node's component color
  let graph = state.rootGraph;
  state.scopeStack.forEach((scope, i) => {
    const node = graph.nodes.find(n => n.id === scope.nodeId);
    const comp = node ? getCompDef(node.type) : null;
    const color = comp?.color || 'var(--text-dim)';
    html += `<span class="crumb-sep">\u25B8</span>`;
    html += `<span class="crumb clickable" data-depth="${i}" style="color:${color}">${scope.nodeLabel}</span>`;
    if (node && node.children) graph = node.children;
  });

  el.innerHTML = html;

  // Bind back button
  const backBtn = document.getElementById('back-btn');
  if (backBtn) {
    backBtn.addEventListener('click', () => {
      navigateToDepth(state.scopeStack.length - 2);
    });
  }

  // Bind click handlers
  el.querySelectorAll('.crumb.clickable').forEach(crumb => {
    crumb.addEventListener('click', () => {
      const depth = parseInt(crumb.dataset.depth);
      navigateToDepth(depth);
    });
  });
}

function getParentNode() {
  if (state.scopeStack.length === 0) return null;
  let graph = state.rootGraph;
  let parentNode = null;
  for (const scope of state.scopeStack) {
    parentNode = graph.nodes.find(n => n.id === scope.nodeId);
    if (parentNode && parentNode.children) graph = parentNode.children;
  }
  return parentNode;
}

// Engine component — shown as a container node at project scope
const ENGINE_COMPONENT = {
  type: 'engine',
  label: 'Engine',
  category: 'structure',
  color: '#4a90d9',
  ports: {
    in: [
      { id: 'raw_text', label: 'data', dataType: 'raw_text', shape: [] }
    ],
    out: []
  },
  params: {},
};

// Pipeline-level boundary ports — data enters from the left wall
const PIPELINE_BOUNDARY = {
  ports: {
    in: [
      { id: 'raw_text', label: 'data', dataType: 'raw_text', shape: [] }
    ],
    out: []
  },
  color: '#d9c84a'
};

function updateViewportBorder() {
  const wrap = document.getElementById('canvas-wrap');
  if (state.scopeStack.length === 0) {
    // Project scope — no border, no boundary ports
    wrap.style.border = 'none';
    const svgEl = document.getElementById('canvas');
    svgEl.style.outline = 'none';
    svgEl.style.borderRadius = '';
    removeBoundaryPorts();
    return;
  }
  const parentNode = getParentNode();
  if (!parentNode) return;
  const comp = getCompDef(parentNode.type);
  if (!comp) return;
  wrap.style.border = 'none';
  const svgEl = document.getElementById('canvas');
  svgEl.style.outline = `4px solid ${comp.color}`;
  svgEl.style.outlineOffset = '-4px';
  svgEl.style.borderRadius = '8px';
  if (parentNode.type === 'engine') {
    // Inside engine (pipeline level) — show the data boundary port on left wall
    renderPipelineBoundaryPorts();
  } else {
    renderBoundaryPorts(parentNode, comp);
  }
}

function removeBoundaryPorts() {
  const overlay = document.getElementById('boundary-ports-overlay');
  if (overlay) {
    if (overlay._resizeHandler) window.removeEventListener('resize', overlay._resizeHandler);
    overlay.remove();
  }
}

function renderPipelineBoundaryPorts() {
  const { overlay, h } = ensureBoundaryOverlay();
  const portsIn = PIPELINE_BOUNDARY.ports.in;
  portsIn.forEach((p, i) => {
    renderBoundaryPort(overlay, p, false, 2, distributePort(i, portsIn.length, h), PIPELINE_BOUNDARY);
  });
  if (overlay._resizeHandler) window.removeEventListener('resize', overlay._resizeHandler);
  overlay._resizeHandler = () => { renderPipelineBoundaryPorts(); updateBoundaryEdges(); };
  window.addEventListener('resize', overlay._resizeHandler);
}

function renderBoundaryPorts(parentNode, comp) {
  const { overlay, h, w } = ensureBoundaryOverlay();
  const borderW = 4;
  const portsIn = comp.ports.in || [];
  const portsOut = comp.ports.out || [];
  portsIn.forEach((p, i) => {
    renderBoundaryPort(overlay, p, false, borderW / 2, distributePort(i, portsIn.length, h), comp);
  });
  portsOut.forEach((p, i) => {
    renderBoundaryPort(overlay, p, true, w - borderW / 2, distributePort(i, portsOut.length, h), comp);
  });
  if (overlay._resizeHandler) window.removeEventListener('resize', overlay._resizeHandler);
  overlay._resizeHandler = () => { renderBoundaryPorts(parentNode, comp); updateBoundaryEdges(); };
  window.addEventListener('resize', overlay._resizeHandler);
}

function distributePort(index, total, viewHeight) {
  // 1 port → 50%, 2 ports → 33%/66%, 3 ports → 25%/50%/75%
  return viewHeight * (index + 1) / (total + 1);
}

function renderBoundaryPort(overlay, portDef, isOutput, px, py, comp) {
  const dtColor = state.registry.dataTypes[portDef.dataType]?.color || '#888';
  const rank = (portDef.shape || []).length;

  const g = document.createElementNS(SVG_NS, 'g');
  g.style.pointerEvents = 'all';

  // Scale to match node ports (7r circle / 12px square at graph coords)
  const z = state.zoom;
  const r = Math.max(7 * z, 4);       // match node port radius, min 4px
  const sw = Math.max(2 * z, 1.5);    // stroke width
  const fs = Math.max(10 * z, 8);     // font size
  const labelGap = Math.max(12 * z, 10);

  const shape = createPortShape(rank, px, py, r, Math.max(2 * z, 1));
  if (!isOutput) {
    shape.setAttribute('fill', dtColor);
    shape.setAttribute('stroke', dtColor);
    shape.setAttribute('stroke-width', sw);
  } else {
    shape.setAttribute('fill', 'transparent');
    shape.setAttribute('stroke', dtColor);
    shape.setAttribute('stroke-width', sw * 1.2);
  }
  g.appendChild(shape);

  // Label
  const label = document.createElementNS(SVG_NS, 'text');
  label.setAttribute('x', isOutput ? px - labelGap : px + labelGap);
  label.setAttribute('y', py + fs * 0.35);
  label.setAttribute('text-anchor', isOutput ? 'end' : 'start');
  label.setAttribute('font-size', fs);
  label.setAttribute('fill', 'rgba(255,255,255,0.85)');
  label.setAttribute('font-family', 'inherit');
  label.textContent = portDef.label;
  g.appendChild(label);

  // Store data for connection logic
  g.dataset.boundaryPort = portDef.id;
  g.dataset.isOutput = isOutput;
  g.dataset.dataType = portDef.dataType;
  g.classList.add('port', 'boundary-port');

  // Make boundary ports connectable
  g.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    onBoundaryPortMouseDown(e, portDef, isOutput);
  });
  g.addEventListener('mouseup', (e) => {
    e.stopPropagation();
    onBoundaryPortMouseUp(e, portDef, isOutput);
  });

  overlay.appendChild(g);
}

// Boundary port interaction: parent inputs act as outputs inside the scope (they feed data in),
// parent outputs act as inputs inside the scope (they receive data out).
function onBoundaryPortMouseDown(e, portDef, isParentOutput) {
  e.stopPropagation();
  e.preventDefault();
  // Parent input → data source inside scope (acts like an output port)
  // Parent output → data sink inside scope (acts like an input port)
  const pos = getBoundaryPortPosition(portDef.id, isParentOutput);
  state.connecting = {
    nodeId: -1,
    portId: portDef.id,
    portDef: portDef,
    isOutput: !isParentOutput, // flip: parent input feeds data in = output inside scope
    startX: pos.x,
    startY: pos.y,
    fromBoundary: true,
  };
  draftEdge.setAttribute('stroke', state.registry.dataTypes[portDef.dataType]?.color || '#888');
}

function onBoundaryPortMouseUp(e, portDef, isParentOutput) {
  if (!state.connecting) return;
  e.stopPropagation();
  // Check direction compatibility:
  // parent input = output inside scope, parent output = input inside scope
  const boundaryActsAsOutput = !isParentOutput;
  if (boundaryActsAsOutput === state.connecting.isOutput) return; // same direction, can't connect

  const fromDT = state.connecting.isOutput ? state.connecting.portDef.dataType : portDef.dataType;
  const toDT = state.connecting.isOutput ? portDef.dataType : state.connecting.portDef.dataType;

  if (canConnect(fromDT, toDT)) {
    if (state.connecting.isOutput) {
      addEdge(state.connecting.nodeId, state.connecting.portId, -1, portDef.id);
    } else {
      addEdge(-1, portDef.id, state.connecting.nodeId, state.connecting.portId);
    }
  }
  state.connecting = null;
  draftEdge.setAttribute('d', '');
  renderAll();
}

function getBoundaryPortPosition(portId, _isOutputHint) {
  const svgEl = document.getElementById('canvas');
  const h = svgEl.clientHeight;
  const w = svgEl.clientWidth;
  const parentNode = getParentNode();

  // At pipeline level, use pipeline boundary ports
  let portsIn, portsOut;
  if (!parentNode) {
    portsIn = PIPELINE_BOUNDARY.ports.in || [];
    portsOut = PIPELINE_BOUNDARY.ports.out || [];
  } else {
    const comp = getCompDef(parentNode.type);
    if (!comp) return { x: 0, y: 0 };
    portsIn = comp.ports.in || [];
    portsOut = comp.ports.out || [];
  }
  let idx = portsIn.findIndex(p => p.id === portId);
  let isOnRight = false;
  if (idx < 0) {
    idx = portsOut.findIndex(p => p.id === portId);
    isOnRight = true;
  }
  if (idx < 0) return { x: 0, y: 0 };

  const ports = isOnRight ? portsOut : portsIn;
  const screenX = isOnRight ? w - 2 : 2;
  const screenY = distributePort(idx, ports.length, h);

  // Convert screen coords to graph coords (inverse of D3 transform)
  return {
    x: (screenX - state.viewX) / state.zoom,
    y: (screenY - state.viewY) / state.zoom,
  };
}

function navigateToDepth(depth) {
  // depth = -1 means root (pipeline)
  if (depth < -1) return;

  // Remember the child we're leaving so we can select it
  const childNodeId = (state.scopeStack.length > depth + 1)
    ? state.scopeStack[depth + 1].nodeId
    : null;

  // Pop scope stack to target depth
  while (state.scopeStack.length > depth + 1) {
    state.scopeStack.pop();
  }

  // Resolve the current graph from the root
  let graph = state.rootGraph;
  let scopeType = 'project';
  for (const scope of state.scopeStack) {
    const node = graph.nodes.find(n => n.id === scope.nodeId);
    if (node && node.children) {
      graph = node.children;
      scopeType = scope.scopeType;
    }
  }

  state.nodes = graph.nodes;
  state.edges = graph.edges;
  state.currentScope = scopeType;
  state.selected = childNodeId;

  buildPalette();
  renderBreadcrumbs();
  renderProps();
  updateViewportBorder();
  renderAll();
  fitToView();
  updateStatus();
  renderTree();
}

function zoomIntoNode(nodeId) {
  const node = state.nodes.find(n => n.id === nodeId);
  if (!node) return;

  // Check current scope first, then fall back to any scope that has this type as a container
  let childScope = SCOPE_LEVELS[state.currentScope]?.containers[node.type];
  if (!childScope) {
    for (const scopeKey of Object.keys(SCOPE_LEVELS)) {
      const s = SCOPE_LEVELS[scopeKey];
      if (s.containers[node.type]) { childScope = s.containers[node.type]; break; }
    }
  }
  if (!childScope) return;

  // Initialize children graph if needed
  if (!node.children || node.children.nodes.length === 0) {
    node.children = { nodes: [], edges: [] };
    // Pre-populate with default internal layout
    const defaults = getDefaultChildren(node.type, node.params);
    if (defaults) {
      node.children = defaults;
      // Update nextId to avoid collisions
      for (const n of node.children.nodes) {
        if (n.id >= state.nextId) state.nextId = n.id + 1;
      }
      for (const e of node.children.edges) {
        if (e.id >= state.nextId) state.nextId = e.id + 1;
      }
    }
  }

  let scopeLabel;
  if (node.type === 'engine') {
    const card = getCardForNode(node.id);
    scopeLabel = card?.name || 'Engine';
  } else {
    scopeLabel = node.params?.name || (getCompDef(node.type)?.label + ' #' + node.id);
  }
  state.scopeStack.push({
    nodeId: node.id,
    nodeLabel: scopeLabel,
    scopeType: childScope,
  });

  state.nodes = node.children.nodes;
  state.edges = node.children.edges;
  state.currentScope = childScope;
  state.selected = null;

  buildPalette();
  renderBreadcrumbs();
  renderProps();
  updateViewportBorder();
  renderAll();
  fitToView();
  updateStatus();
  renderTree();
}

// ── Default internal layouts ─────────────────────────────────────────────────

function getDefaultChildren(type, params) {
  if (type === 'cooperative') {
    return getCooperativeDefaults(params);
  }
  if (type === 'transformer') {
    return getTransformerDefaults(params);
  }
  if (type === 'output_head') {
    return getOutputHeadDefaults(params);
  }
  if (type === 'stream_proj_internal') {
    return getStreamProjDefaults(params);
  }
  if (type === 'embedding') {
    return getEmbeddingDefaults(params);
  }
  if (type === 'global_router') {
    return getGlobalRouterDefaults(params);
  }
  if (type === 'ffn_layer') {
    return getFFNDefaults(params);
  }
  if (type === 'layer_norm') {
    return getLayerNormDefaults(params);
  }
  return null;
}

function getCooperativeDefaults(params) {
  const sd = params.stream_dim || 64;
  const id = () => state.nextId++;
  const n1 = id(), n2 = id(), n3 = id();

  return {
    nodes: [
      { id: n1, type: "transformer", x: 80, y: 40, params: { d_model: sd, n_layers: 3, stream_dim: sd }, children: null },
      { id: n2, type: "transformer", x: 80, y: 220, params: { d_model: sd, n_layers: 3, stream_dim: sd }, children: null },
      { id: n3, type: "global_router", x: 400, y: 130, params: { stream_dim: sd, epsilon: 0.2 } },
    ],
    edges: [
      // Boundary tokens → both experts
      { id: id(), from: { nodeId: -1, portId: 'token_ids' }, to: { nodeId: n1, portId: 'token_ids' } },
      { id: id(), from: { nodeId: -1, portId: 'token_ids' }, to: { nodeId: n2, portId: 'token_ids' } },
      // Boundary stream → Expert A
      { id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: n1, portId: 'stream_in' } },
      // Expert A stream → Expert B (cooperative chaining)
      { id: id(), from: { nodeId: n1, portId: 'stream_out' }, to: { nodeId: n2, portId: 'stream_in' } },
      // Stream also feeds router (from boundary — sees initial state)
      { id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: n3, portId: 'stream_in' } },
      // Expert logits → router (multi-input)
      { id: id(), from: { nodeId: n1, portId: 'logits_out' }, to: { nodeId: n3, portId: 'logits_in' } },
      { id: id(), from: { nodeId: n2, portId: 'logits_out' }, to: { nodeId: n3, portId: 'logits_in' } },
      // Router → boundary output
      { id: id(), from: { nodeId: n3, portId: 'logits_out' }, to: { nodeId: -1, portId: 'logits_out' } },
    ],
  };
}

function getTransformerDefaults(params) {
  const dm = params.d_model || 64;
  const nl = params.n_layers || 3;
  const sd = params.stream_dim || 64;
  const id = () => state.nextId++;

  const nodes = [];
  const edges = [];
  const xStep = 220;
  const yStep = 180;

  // Track chain for wiring
  let lastId = null, lastPort = null;
  let row = 0;

  // Embedding: tokens → [d_model] vectors
  const embId = id();
  nodes.push({ id: embId, type: "embedding", x: 100, y: row * yStep + 60, params: { d_model: dm } });
  edges.push({ id: id(), from: { nodeId: -1, portId: 'token_ids' }, to: { nodeId: embId, portId: 'in' } });
  lastId = embId; lastPort = 'out';

  // Stream projection in (if stream_dim != d_model) — adds to embedding output
  if (sd !== dm) {
    const projIn = id();
    nodes.push({ id: projIn, type: "stream_proj_internal", x: 100 + xStep, y: row * yStep + 60, params: { d_in: sd, d_out: dm } });
    edges.push({ id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: projIn, portId: 'in' } });
    // TODO: visually show additive combination of embedding + stream projection
  } else {
    // Stream feeds directly (same dim, additive with embedding in the engine)
  }
  row++;

  // Each layer block is a row: LN → Attn → LN → FFN (left to right)
  for (let i = 0; i < nl; i++) {
    const y = row * yStep + 60;
    const lnA = id(), att = id(), lnF = id(), ffn = id();
    nodes.push({ id: lnA, type: "layer_norm",     x: 100,              y, params: { dim: dm } });
    nodes.push({ id: att, type: "attention_layer", x: 100 + xStep,     y, params: { d_model: dm, n_heads: Math.max(1, dm / 16) } });
    nodes.push({ id: lnF, type: "layer_norm",      x: 100 + xStep * 2, y, params: { dim: dm } });
    nodes.push({ id: ffn, type: "ffn_layer",       x: 100 + xStep * 3, y, params: { d_model: dm, d_ff: dm * 4 } });

    // Wire within the row: LN→Attn→LN→FFN
    edges.push({ id: id(), from: { nodeId: lnA, portId: 'out' }, to: { nodeId: att, portId: 'in' } });
    edges.push({ id: id(), from: { nodeId: att, portId: 'out' }, to: { nodeId: lnF, portId: 'in' } });
    edges.push({ id: id(), from: { nodeId: lnF, portId: 'out' }, to: { nodeId: ffn, portId: 'in' } });

    // Chain from previous row or boundary
    if (lastId) {
      edges.push({ id: id(), from: { nodeId: lastId, portId: lastPort }, to: { nodeId: lnA, portId: 'in' } });
    } else {
      edges.push({ id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: lnA, portId: 'in' } });
    }

    lastId = ffn; lastPort = 'out';
    row++;
  }

  // The last FFN output splits two ways:
  // 1. → Output Head → logits_out (boundary)
  // 2. → (optional proj) → stream_out (boundary)
  const streamLastId = lastId;
  const streamLastPort = lastPort;

  // Output head (logits path)
  const y = row * yStep + 60;
  const outId = id();
  nodes.push({ id: outId, type: "output_head", x: 100, y, params: { d_model: dm } });
  edges.push({ id: id(), from: { nodeId: streamLastId, portId: streamLastPort }, to: { nodeId: outId, portId: 'in' } });
  edges.push({ id: id(), from: { nodeId: outId, portId: 'out' }, to: { nodeId: -1, portId: 'logits_out' } });

  // Stream path (bypass output head)
  if (sd !== dm) {
    const projOut = id();
    nodes.push({ id: projOut, type: "stream_proj_internal", x: 100 + xStep, y, params: { d_in: dm, d_out: sd } });
    edges.push({ id: id(), from: { nodeId: streamLastId, portId: streamLastPort }, to: { nodeId: projOut, portId: 'in' } });
    edges.push({ id: id(), from: { nodeId: projOut, portId: 'out' }, to: { nodeId: -1, portId: 'stream_out' } });
  } else {
    edges.push({ id: id(), from: { nodeId: streamLastId, portId: streamLastPort }, to: { nodeId: -1, portId: 'stream_out' } });
  }

  return { nodes, edges };
}

function getOutputHeadDefaults(params) {
  const dm = params.d_model || 64;
  const id = () => state.nextId++;

  const wId = id(), bId = id(), mmId = id(), addId = id();

  return {
    nodes: [
      { id: wId,  type: "weight_param", x: 60,  y: 40,  params: { rows: dm, cols: dm } },
      { id: mmId, type: "matmul",       x: 300, y: 40,  params: { d_in: dm, d_out: dm } },
      { id: bId,  type: "bias_param",   x: 60,  y: 200, params: { dim: dm } },
      { id: addId, type: "add_bias",    x: 300, y: 200, params: { dim: dm } },
    ],
    edges: [
      // boundary hidden input → matmul x
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: mmId, portId: 'x' } },
      // weight → matmul W
      { id: id(), from: { nodeId: wId, portId: 'out' }, to: { nodeId: mmId, portId: 'W' } },
      // matmul out → add_bias x
      { id: id(), from: { nodeId: mmId, portId: 'out' }, to: { nodeId: addId, portId: 'x' } },
      // bias → add_bias b
      { id: id(), from: { nodeId: bId, portId: 'out' }, to: { nodeId: addId, portId: 'b' } },
      // add_bias out → boundary logits output
      { id: id(), from: { nodeId: addId, portId: 'out' }, to: { nodeId: -1, portId: 'out' } },
    ],
  };
}

function getStreamProjDefaults(params) {
  const dIn = params.d_in || 64;
  const dOut = params.d_out || 64;
  const id = () => state.nextId++;

  const wId = id(), bId = id(), mmId = id(), addId = id();

  return {
    nodes: [
      { id: wId,  type: "weight_param", x: 60,  y: 40,  params: { rows: dIn, cols: dOut } },
      { id: mmId, type: "matmul",       x: 300, y: 40,  params: { d_in: dIn, d_out: dOut } },
      { id: bId,  type: "bias_param",   x: 60,  y: 200, params: { dim: dOut } },
      { id: addId, type: "add_bias",    x: 300, y: 200, params: { dim: dOut } },
    ],
    edges: [
      // boundary input → matmul x
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: mmId, portId: 'x' } },
      // weight → matmul W
      { id: id(), from: { nodeId: wId, portId: 'out' }, to: { nodeId: mmId, portId: 'W' } },
      // matmul out → add_bias x
      { id: id(), from: { nodeId: mmId, portId: 'out' }, to: { nodeId: addId, portId: 'x' } },
      // bias → add_bias b
      { id: id(), from: { nodeId: bId, portId: 'out' }, to: { nodeId: addId, portId: 'b' } },
      // add_bias out → boundary output
      { id: id(), from: { nodeId: addId, portId: 'out' }, to: { nodeId: -1, portId: 'out' } },
    ],
  };
}

function getEmbeddingDefaults(params) {
  const dm = params.d_model || 64;
  const id = () => state.nextId++;

  const tableId = id(), lookupId = id();

  return {
    nodes: [
      { id: tableId,  type: "embedding_table", x: 60,  y: 40,  params: { vocab_size: 65, d_model: dm } },
      { id: lookupId, type: "lookup",           x: 300, y: 40,  params: { d_model: dm } },
    ],
    edges: [
      // boundary token_ids → lookup ids
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: lookupId, portId: 'ids' } },
      // embedding table → lookup table
      { id: id(), from: { nodeId: tableId, portId: 'out' }, to: { nodeId: lookupId, portId: 'table' } },
      // lookup out → boundary output
      { id: id(), from: { nodeId: lookupId, portId: 'out' }, to: { nodeId: -1, portId: 'out' } },
    ],
  };
}

function getGlobalRouterDefaults(params) {
  const sd = params.stream_dim || 64;
  const nr = params.n_experts || 2;
  const eps = params.epsilon || 0.2;
  const id = () => state.nextId++;

  // mean_pool(stream) → W·x + b → softmax → ε-blend → broadcast
  const poolId = id(), wId = id(), mmId = id(), bId = id(), addId = id();
  const smId = id(), blendId = id(), bcastId = id();

  return {
    nodes: [
      { id: poolId,  type: "mean_pool",    x: 300, y: 40,  params: { dim: sd } },
      { id: wId,     type: "weight_param", x: 60,  y: 200, params: { rows: sd, cols: nr } },
      { id: mmId,    type: "matmul",       x: 300, y: 200, params: { d_in: sd, d_out: nr } },
      { id: bId,     type: "bias_param",   x: 60,  y: 360, params: { dim: nr } },
      { id: addId,   type: "add_bias",     x: 300, y: 360, params: { dim: nr } },
      { id: smId,    type: "softmax",      x: 300, y: 520, params: { dim: nr } },
      { id: blendId, type: "eps_blend",    x: 300, y: 680, params: { n_experts: nr, epsilon: eps } },
      { id: bcastId, type: "broadcast",    x: 300, y: 840, params: { dim: nr } },
    ],
    edges: [
      // boundary stream → mean pool
      { id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: poolId, portId: 'in' } },
      // mean pool → matmul x
      { id: id(), from: { nodeId: poolId, portId: 'out' }, to: { nodeId: mmId, portId: 'x' } },
      // W → matmul
      { id: id(), from: { nodeId: wId, portId: 'out' }, to: { nodeId: mmId, portId: 'W' } },
      // matmul → add_bias
      { id: id(), from: { nodeId: mmId, portId: 'out' }, to: { nodeId: addId, portId: 'x' } },
      // b → add_bias
      { id: id(), from: { nodeId: bId, portId: 'out' }, to: { nodeId: addId, portId: 'b' } },
      // add_bias → softmax
      { id: id(), from: { nodeId: addId, portId: 'out' }, to: { nodeId: smId, portId: 'in' } },
      // softmax → ε-blend
      { id: id(), from: { nodeId: smId, portId: 'out' }, to: { nodeId: blendId, portId: 'in' } },
      // ε-blend → broadcast
      { id: id(), from: { nodeId: blendId, portId: 'out' }, to: { nodeId: bcastId, portId: 'in' } },
      // broadcast → boundary logits output
      { id: id(), from: { nodeId: bcastId, portId: 'out' }, to: { nodeId: -1, portId: 'logits_out' } },
    ],
  };
}

function getLayerNormDefaults(params) {
  const dim = params.dim || 64;
  const id = () => state.nextId++;

  // γ * (x - μ) / sqrt(σ² + ε) + β
  const meanId = id(), subId = id(), varId = id(), normId = id();
  const gammaId = id(), mulId = id(), betaId = id(), addId = id();

  return {
    nodes: [
      { id: meanId,  type: "reduce_mean", x: 60,  y: 40,  params: { dim } },
      { id: subId,   type: "subtract",    x: 300, y: 40,  params: { dim } },
      { id: varId,   type: "reduce_var",  x: 60,  y: 200, params: { dim } },
      { id: normId,  type: "normalize",   x: 300, y: 200, params: { dim, eps: 1e-5 } },
      { id: gammaId, type: "scale_param", x: 60,  y: 360, params: { dim } },
      { id: mulId,   type: "elem_mul",    x: 300, y: 360, params: { dim } },
      { id: betaId,  type: "bias_param",  x: 60,  y: 520, params: { dim } },
      { id: addId,   type: "add_bias",    x: 300, y: 520, params: { dim } },
    ],
    edges: [
      // boundary input → mean
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: meanId, portId: 'in' } },
      // boundary input → subtract a (original x)
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: subId, portId: 'a' } },
      // mean → subtract b
      { id: id(), from: { nodeId: meanId, portId: 'out' }, to: { nodeId: subId, portId: 'b' } },
      // subtract (centered) → variance
      { id: id(), from: { nodeId: subId, portId: 'out' }, to: { nodeId: varId, portId: 'in' } },
      // subtract (centered) → normalize x
      { id: id(), from: { nodeId: subId, portId: 'out' }, to: { nodeId: normId, portId: 'x' } },
      // variance → normalize var
      { id: id(), from: { nodeId: varId, portId: 'out' }, to: { nodeId: normId, portId: 'var' } },
      // normalize → elem_mul x
      { id: id(), from: { nodeId: normId, portId: 'out' }, to: { nodeId: mulId, portId: 'x' } },
      // gamma → elem_mul scale
      { id: id(), from: { nodeId: gammaId, portId: 'out' }, to: { nodeId: mulId, portId: 'scale' } },
      // elem_mul → add_bias x
      { id: id(), from: { nodeId: mulId, portId: 'out' }, to: { nodeId: addId, portId: 'x' } },
      // beta → add_bias b
      { id: id(), from: { nodeId: betaId, portId: 'out' }, to: { nodeId: addId, portId: 'b' } },
      // add_bias → boundary output
      { id: id(), from: { nodeId: addId, portId: 'out' }, to: { nodeId: -1, portId: 'out' } },
    ],
  };
}

function getFFNDefaults(params) {
  const dm = params.d_model || 64;
  const dff = params.d_ff || 256;
  const id = () => state.nextId++;

  // Layer 1: x @ W1 + b1 → ReLU
  const w1 = id(), b1 = id(), mm1 = id(), add1 = id(), relu = id();
  // Layer 2: ReLU(…) @ W2 + b2
  const w2 = id(), b2 = id(), mm2 = id(), add2 = id();

  return {
    nodes: [
      { id: w1,   type: "weight_param", x: 60,  y: 40,  params: { rows: dm, cols: dff } },
      { id: mm1,  type: "matmul",       x: 300, y: 40,  params: { d_in: dm, d_out: dff } },
      { id: b1,   type: "bias_param",   x: 60,  y: 200, params: { dim: dff } },
      { id: add1, type: "add_bias",     x: 300, y: 200, params: { dim: dff } },
      { id: relu, type: "relu",         x: 300, y: 360, params: { dim: dff } },
      { id: w2,   type: "weight_param", x: 60,  y: 520, params: { rows: dff, cols: dm } },
      { id: mm2,  type: "matmul",       x: 300, y: 520, params: { d_in: dff, d_out: dm } },
      { id: b2,   type: "bias_param",   x: 60,  y: 680, params: { dim: dm } },
      { id: add2, type: "add_bias",     x: 300, y: 680, params: { dim: dm } },
    ],
    edges: [
      // boundary input → matmul1 x
      { id: id(), from: { nodeId: -1, portId: 'in' }, to: { nodeId: mm1, portId: 'x' } },
      // W1 → matmul1
      { id: id(), from: { nodeId: w1, portId: 'out' }, to: { nodeId: mm1, portId: 'W' } },
      // matmul1 → add_bias1
      { id: id(), from: { nodeId: mm1, portId: 'out' }, to: { nodeId: add1, portId: 'x' } },
      // b1 → add_bias1
      { id: id(), from: { nodeId: b1, portId: 'out' }, to: { nodeId: add1, portId: 'b' } },
      // add_bias1 → relu
      { id: id(), from: { nodeId: add1, portId: 'out' }, to: { nodeId: relu, portId: 'in' } },
      // relu → matmul2 x
      { id: id(), from: { nodeId: relu, portId: 'out' }, to: { nodeId: mm2, portId: 'x' } },
      // W2 → matmul2
      { id: id(), from: { nodeId: w2, portId: 'out' }, to: { nodeId: mm2, portId: 'W' } },
      // matmul2 → add_bias2
      { id: id(), from: { nodeId: mm2, portId: 'out' }, to: { nodeId: add2, portId: 'x' } },
      // b2 → add_bias2
      { id: id(), from: { nodeId: b2, portId: 'out' }, to: { nodeId: add2, portId: 'b' } },
      // add_bias2 → boundary output
      { id: id(), from: { nodeId: add2, portId: 'out' }, to: { nodeId: -1, portId: 'out' } },
    ],
  };
}

// ── Palette drag → canvas drop ───────────────────────────────────────────────

let dragGhost = null;

function onPaletteDragStart(e) {
  const type = e.currentTarget.dataset.type;
  e.dataTransfer.setData('text/plain', type);
  e.dataTransfer.effectAllowed = 'copy';

  const comp = getCompDef(type);
  dragGhost = document.createElement('div');
  dragGhost.className = 'drag-ghost';
  dragGhost.style.background = comp.color;
  dragGhost.textContent = comp.label;
  document.body.appendChild(dragGhost);
  e.dataTransfer.setDragImage(dragGhost, 80, 20);
  setTimeout(() => { if (dragGhost) { dragGhost.remove(); dragGhost = null; } }, 0);
}

const canvasWrap = document.getElementById('canvas-wrap');
canvasWrap.addEventListener('dragover', e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
canvasWrap.addEventListener('drop', e => {
  e.preventDefault();
  const type = e.dataTransfer.getData('text/plain');
  if (!type) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left - state.viewX) / state.zoom;
  const y = (e.clientY - rect.top - state.viewY) / state.zoom;
  addNode(type, x, y);
});

// ── Node management ──────────────────────────────────────────────────────────

function getCompDef(type) {
  if (type === 'engine') return ENGINE_COMPONENT;
  return state.registry.components.find(c => c.type === type);
}

// Extract the pipeline-level graph from a card's engine node
function getPipelineGraph(card) {
  const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
  return engineNode?.children || { nodes: [], edges: [] };
}

function addNode(type, x, y) {
  if (isActiveLocked()) { setStatus('Cannot edit while training', { flash: true }); return; }
  if (type === 'engine') {
    if (cards.length >= MAX_CARDS) { setStatus('Max engines reached', { flash: true }); return; }
    const card = createEmptyCard('Engine ' + (cards.length + 1));
    // Position the engine node where it was dropped
    const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
    if (engineNode) { engineNode.x = x; engineNode.y = y; }
    cards.push(card);
    renderAll();
    renderCardList();
    updateStatus();
    return;
  }
  const comp = getCompDef(type);
  if (!comp) return;
  const node = {
    id: state.nextId++,
    type: type,
    label: comp.label,
    x: x,
    y: y,
    params: {},
    children: null,
  };
  for (const [k, v] of Object.entries(comp.params || {})) {
    node.params[k] = v.default;
  }
  state.nodes.push(node);
  renderAll();
  selectNode(node.id);
  updateStatus();
  return node;
}

function deleteNode(id) {
  if (isActiveLocked()) { setStatus('Cannot edit while training', { flash: true }); return; }
  const node = state.nodes.find(n => n.id === id);
  if (node?.type === 'engine') { setStatus('Cannot delete engines from canvas'); return; }
  state.edges = state.edges.filter(e => {
    if (e.from.nodeId === id || e.to.nodeId === id) {
      removeEdgeEl(e.id);
      return false;
    }
    return true;
  });
  state.nodes.splice(state.nodes.findIndex(n => n.id === id), 1);
  const el = document.getElementById(`node-${id}`);
  if (el) el.remove();
  if (state.selected === id) {
    state.selected = null;
    renderProps();
  }
  updateStatus();
}

// ── Render all ───────────────────────────────────────────────────────────────
// Simple: clear layers, render current scope nodes and edges, update props.
// D3 zoom handles the SVG transform — no applyViewTransform() needed.

function renderAll() {
  nodesLayer.innerHTML = '';
  edgesLayer.innerHTML = '';
  portPlacementCache.clear();

  for (const node of state.nodes) renderNode(node);
  for (const e of state.edges) renderEdge(e);

  renderProps();
  renderTree();
  debouncedAutoSave();
}

// ── Port placement optimization ──────────────────────────────────────────────
// For each port, find connected nodes and place the port on the side of the
// rectangle that minimizes wire length. Unconnected ports use defaults.

function computePortPlacements(node, comp, w, h, headerH, portSpacing) {
  const ports_in = comp.ports.in || [];
  const ports_out = comp.ports.out || [];
  const placements = [];

  // Gather connected target positions for each port
  function getConnectedCenter(nodeId, portId, isOutput) {
    // Find edges involving this port
    const targets = [];
    for (const e of state.edges) {
      if (isOutput && e.from.nodeId === nodeId && e.from.portId === portId) {
        const tn = e.to.nodeId === -1 ? null : state.nodes.find(n => n.id === e.to.nodeId);
        if (tn) targets.push({ x: tn.x + 90, y: tn.y + h / 2 });
        else if (e.to.nodeId === -1) {
          // Boundary port — estimate position from viewport
          const bp = getBoundaryPortPosition(e.to.portId, false);
          if (bp) targets.push(bp);
        }
      }
      if (!isOutput && e.to.nodeId === nodeId && e.to.portId === portId) {
        const fn = e.from.nodeId === -1 ? null : state.nodes.find(n => n.id === e.from.nodeId);
        if (fn) targets.push({ x: fn.x + 90, y: fn.y + h / 2 });
        else if (e.from.nodeId === -1) {
          const bp = getBoundaryPortPosition(e.from.portId, true);
          if (bp) targets.push(bp);
        }
      }
    }
    return targets;
  }

  // Determine best side for a port given target positions
  // Returns { side: 'left'|'right'|'top'|'bottom', targets }
  function bestSide(targets, isOutput) {
    if (targets.length === 0) {
      return isOutput ? 'right' : 'left'; // default
    }
    // Average target position relative to node center
    const cx = node.x + w / 2;
    const cy = node.y + h / 2;
    let avgDx = 0, avgDy = 0;
    for (const t of targets) {
      avgDx += t.x - cx;
      avgDy += t.y - cy;
    }
    avgDx /= targets.length;
    avgDy /= targets.length;

    // Pick the side that best faces the average target
    // Use aspect-ratio-weighted comparison so wider nodes prefer top/bottom less
    const absX = Math.abs(avgDx);
    const absY = Math.abs(avgDy) * (w / h); // scale to account for aspect ratio

    if (absX >= absY) {
      return avgDx >= 0 ? 'right' : 'left';
    } else {
      return avgDy >= 0 ? 'bottom' : 'top';
    }
  }

  // Assign each port to a side
  const sidePorts = { left: [], right: [], top: [], bottom: [] };

  for (const p of ports_in) {
    const targets = getConnectedCenter(node.id, p.id, false);
    const side = bestSide(targets, false);
    sidePorts[side].push({ portDef: p, isOutput: false, targets });
  }
  for (const p of ports_out) {
    const targets = getConnectedCenter(node.id, p.id, true);
    const side = bestSide(targets, true);
    sidePorts[side].push({ portDef: p, isOutput: true, targets });
  }

  // Distribute ports along each side
  function distribute(ports, side) {
    const n = ports.length;
    if (n === 0) return;

    for (let i = 0; i < n; i++) {
      let px, py;
      const t = (i + 1) / (n + 1); // even distribution ratio

      switch (side) {
        case 'left':
          px = 0;
          py = t * h;
          break;
        case 'right':
          px = w;
          py = t * h;
          break;
        case 'top':
          px = t * w;
          py = 0;
          break;
        case 'bottom':
          px = t * w;
          py = h;
          break;
      }

      placements.push({
        portDef: ports[i].portDef,
        isOutput: ports[i].isOutput,
        x: px,
        y: py,
        side: side,
      });
    }
  }

  distribute(sidePorts.left, 'left');
  distribute(sidePorts.right, 'right');
  distribute(sidePorts.top, 'top');
  distribute(sidePorts.bottom, 'bottom');

  return placements;
}

// Cache port placements per node for getPortPosition lookups
const portPlacementCache = new Map();

// ── Node rendering ───────────────────────────────────────────────────────────
// Renders at fixed local coordinates: 180px width, 12px font.
// D3 zoom transform on nodesLayer handles screen-space scaling.

function renderNode(node) {
  const comp = getCompDef(node.type);
  if (!comp) return;
  const g = document.createElementNS(SVG_NS, 'g');
  g.id = `node-${node.id}`;
  g.classList.add('node');
  if (state.selected === node.id) g.classList.add('selected');
  g.dataset.nodeId = node.id;

  const ports_in = comp.ports.in || [];
  const ports_out = comp.ports.out || [];
  const portRows = Math.max(ports_in.length, ports_out.length, 1);
  const headerH = 36;
  const portSpacing = 28;
  const h = headerH + portRows * portSpacing + 8;
  const w = 180;

  // Is this a container? Add visual hint
  const level = SCOPE_LEVELS[state.currentScope];
  const isContainer = !!(getChildScope(node.type));

  // Body
  const rect = document.createElementNS(SVG_NS, 'rect');
  rect.classList.add('node-body');
  rect.setAttribute('width', w);
  rect.setAttribute('height', h);
  rect.setAttribute('fill', '#1a1a2e');
  rect.setAttribute('stroke', comp.color);
  if (isContainer) {
    rect.setAttribute('stroke-dasharray', '6 3');
  }
  g.appendChild(rect);

  // Header label — containers show name + type, others just type
  const customName = node.type === 'engine'
    ? (getCardForNode(node.id)?.name || '')
    : (node.params?.name || '');
  if (isContainer && customName) {
    const nameEl = document.createElementNS(SVG_NS, 'text');
    nameEl.classList.add('node-label');
    nameEl.setAttribute('x', w / 2);
    nameEl.setAttribute('y', 17);
    nameEl.setAttribute('text-anchor', 'middle');
    nameEl.setAttribute('font-size', '12');
    nameEl.textContent = customName;
    g.appendChild(nameEl);

    const typeEl = document.createElementNS(SVG_NS, 'text');
    typeEl.classList.add('node-type-label');
    typeEl.setAttribute('x', w / 2);
    typeEl.setAttribute('y', 30);
    typeEl.setAttribute('text-anchor', 'middle');
    typeEl.setAttribute('font-size', '9');
    typeEl.setAttribute('fill', 'rgba(255,255,255,0.4)');
    typeEl.textContent = comp.label;
    g.appendChild(typeEl);
  } else {
    const label = document.createElementNS(SVG_NS, 'text');
    label.classList.add('node-label');
    label.setAttribute('x', w / 2);
    label.setAttribute('y', 22);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-size', '12');
    label.textContent = customName || comp.label;
    g.appendChild(label);
  }

  // Container hint: "double-click to zoom in" icon
  if (isContainer) {
    const hint = document.createElementNS(SVG_NS, 'text');
    hint.setAttribute('x', w - 8);
    hint.setAttribute('y', 14);
    hint.setAttribute('text-anchor', 'end');
    hint.setAttribute('font-size', '14');
    hint.setAttribute('fill', 'rgba(255,255,255,0.3)');
    hint.setAttribute('pointer-events', 'none');
    hint.textContent = '\u2197'; // ↗
    g.appendChild(hint);
  }

  // Compute optimal port placements based on connected nodes
  const portPlacements = computePortPlacements(node, comp, w, h, headerH, portSpacing);

  // Cache for getPortPosition lookups
  const pmap = new Map();
  for (const p of portPlacements) {
    pmap.set(`${p.isOutput ? 'out' : 'in'}:${p.portDef.id}`, { x: p.x, y: p.y, side: p.side });
  }
  portPlacementCache.set(node.id, pmap);

  // Render all ports at computed positions
  for (const pp of portPlacements) {
    const pg = createPort(node.id, pp.portDef, pp.isOutput, pp.x, pp.y, comp, pp.side);
    g.appendChild(pg);
  }

  g.setAttribute('transform', `translate(${node.x}, ${node.y})`);

  // Node drag (single click)
  g.addEventListener('mousedown', e => onNodeMouseDown(e, node.id));

  // Consume dblclick so canvas handler doesn't fire
  g.addEventListener('dblclick', e => {
    e.stopPropagation();
    e.preventDefault();
  });

  nodesLayer.appendChild(g);
}

// ── Port rendering ───────────────────────────────────────────────────────────

function createPort(nodeId, portDef, isOutput, px, py, comp, side) {
  side = side || (isOutput ? 'right' : 'left');
  const dtColor = state.registry.dataTypes[portDef.dataType]?.color || '#888';
  const g = document.createElementNS(SVG_NS, 'g');
  g.classList.add('port');
  if (isOutput) g.classList.add('port-out');
  else g.classList.add('port-in');
  g.dataset.nodeId = nodeId;
  g.dataset.portId = portDef.id;
  g.dataset.isOutput = isOutput;
  g.dataset.dataType = portDef.dataType;

  const rank = (portDef.shape || []).length;
  const shape = createPortShape(rank, px, py, 7);
  if (isOutput) {
    shape.setAttribute('fill', dtColor);
    shape.setAttribute('stroke', dtColor);
  } else {
    shape.setAttribute('fill', 'transparent');
    shape.setAttribute('stroke', dtColor);
    shape.setAttribute('stroke-width', '2.5');
  }
  g.appendChild(shape);

  // Direction arrow(s) — multi-input ports get three stacked triangles
  // Arrow character and position depend on which side the port is on
  const arrowCount = portDef.multi ? 3 : 1;
  const isHorizontal = side === 'left' || side === 'right';

  if (isHorizontal) {
    const arrowSpacing = 5;
    const arrowStartY = py - (arrowCount - 1) * arrowSpacing / 2;
    for (let ai = 0; ai < arrowCount; ai++) {
      const arrow = document.createElementNS(SVG_NS, 'text');
      arrow.setAttribute('font-size', arrowCount > 1 ? '8' : '10');
      arrow.setAttribute('fill', dtColor);
      arrow.setAttribute('pointer-events', 'none');
      arrow.setAttribute('dominant-baseline', 'central');
      const ay = arrowStartY + ai * arrowSpacing;
      if (side === 'right') {
        arrow.setAttribute('x', px + 11);
        arrow.setAttribute('y', ay);
        arrow.setAttribute('text-anchor', 'start');
        arrow.textContent = '\u25B8'; // ▸
      } else {
        arrow.setAttribute('x', px - 11);
        arrow.setAttribute('y', ay);
        arrow.setAttribute('text-anchor', 'end');
        arrow.textContent = '\u25B8'; // ▸
      }
      g.appendChild(arrow);
    }
  } else {
    const arrowSpacing = 7;
    const arrowStartX = px - (arrowCount - 1) * arrowSpacing / 2;
    for (let ai = 0; ai < arrowCount; ai++) {
      const arrow = document.createElementNS(SVG_NS, 'text');
      arrow.setAttribute('font-size', arrowCount > 1 ? '8' : '10');
      arrow.setAttribute('fill', dtColor);
      arrow.setAttribute('pointer-events', 'none');
      arrow.setAttribute('text-anchor', 'middle');
      const ax = arrowStartX + ai * arrowSpacing;
      arrow.setAttribute('x', ax);
      if (side === 'bottom') {
        arrow.setAttribute('y', py + 13);
        arrow.textContent = '\u25BE'; // ▾
      } else {
        arrow.setAttribute('y', py - 8);
        arrow.textContent = '\u25B4'; // ▴
      }
      g.appendChild(arrow);
    }
  }

  g.addEventListener('mousedown', e => {
    e.stopPropagation();
    onPortMouseDown(e, nodeId, portDef, isOutput, px, py);
  });

  g.addEventListener('mouseenter', (e) => {
    if (state.connecting) highlightCompatibility(g);
    showPortTooltip(e, nodeId, portDef, comp);
  });
  g.addEventListener('mouseleave', () => {
    g.classList.remove('compatible', 'incompatible');
    hidePortTooltip();
  });

  return g;
}

// ── Port tooltip ─────────────────────────────────────────────────────────────

let portTooltipEl = null;

function findDatasetSeqLen() {
  // Walk all nodes to find a windower's seq_len
  const all = state.rootGraph ? state.rootGraph.nodes : state.nodes;
  const windowTypes = ['sequential_window', 'sliding_window', 'random_window', 'dataset'];
  function search(nodes) {
    for (const n of nodes) {
      if (windowTypes.includes(n.type) && n.params?.seq_len) return n.params.seq_len;
      if (n.children?.nodes) {
        const found = search(n.children.nodes);
        if (found) return found;
      }
    }
    return null;
  }
  return search(all);
}

function resolveShape(shapeDef, nodeParams) {
  if (!shapeDef || shapeDef.length === 0) return [];
  return shapeDef.map(d => {
    if (d === null) {
      const sl = findDatasetSeqLen();
      return sl ? String(sl) : 'seq_len';
    }
    if (typeof d === 'number') return String(d);
    // Try to resolve from node params
    if (nodeParams && nodeParams[d] !== undefined) return String(nodeParams[d]);
    // Try to resolve from sibling nodes at current scope level
    const sibVal = findSiblingParam(d);
    if (sibVal !== undefined) return String(sibVal);
    // Prettify param name
    return d.replace(/_/g, ' ');
  });
}

// Search sibling nodes in the current scope for a named param value.
// This lets initializers (zero_init, etc.) inherit stream_dim from the cooperative node.
function findSiblingParam(paramName) {
  for (const n of state.nodes) {
    if (n.params && n.params[paramName] !== undefined) return n.params[paramName];
  }
  return undefined;
}

function showPortTooltip(e, nodeId, portDef, comp) {
  hidePortTooltip();
  const node = state.nodes.find(n => n.id === nodeId);
  const params = node ? node.params : {};
  const resolved = resolveShape(portDef.shape, params);
  const dtInfo = state.registry.dataTypes[portDef.dataType] || {};
  const dtLabel = dtInfo.label || portDef.dataType;
  const dtColor = dtInfo.color || '#888';
  const isInput = !e.target.closest('.port').classList.contains('port-out');
  const direction = isInput ? 'Input' : 'Output';

  // Build shape description
  let shapeDesc, sizeStr;
  if (resolved.length === 0) {
    shapeDesc = 'Scalar';
    sizeStr = '1';
  } else if (resolved.length === 1) {
    shapeDesc = 'Vector (rank 1)';
    sizeStr = `[${resolved[0]}]`;
  } else {
    shapeDesc = `Matrix (rank ${resolved.length})`;
    sizeStr = `[${resolved.join(' × ')}]`;
  }

  const rows = [
    `<div class="ptt-header"><span class="ptt-name">${portDef.label}</span><span class="ptt-size">${sizeStr}</span></div>`,
    `<div class="ptt-row"><span class="ptt-label">Type</span><span class="ptt-value"><span class="ptt-dot" style="background:${dtColor}"></span>${dtLabel}</span></div>`,
    `<div class="ptt-row"><span class="ptt-label">Shape</span><span class="ptt-value">${shapeDesc}</span></div>`,
    `<div class="ptt-row"><span class="ptt-label">Direction</span><span class="ptt-value">${direction}${portDef.multi ? ' (multi)' : ''}</span></div>`,
  ];

  // Show resolved dimension details
  if (portDef.shape && portDef.shape.length > 0) {
    const dimDetails = portDef.shape.map((d, i) => {
      const val = resolved[i];
      if (d === null) return `dim ${i}: <em>seq_len</em> (variable)`;
      if (typeof d === 'number') return `dim ${i}: ${d} (fixed)`;
      const resolvedVal = params[d] !== undefined ? ` = ${params[d]}` : '';
      return `dim ${i}: <em>${d}</em>${resolvedVal}`;
    }).join('<br>');
    rows.push(`<div class="ptt-dims">${dimDetails}</div>`);
  }

  const tip = document.createElement('div');
  tip.className = 'port-tooltip';
  tip.innerHTML = rows.join('');
  document.body.appendChild(tip);
  portTooltipEl = tip;

  const rect = e.target.closest('.port').getBoundingClientRect();
  tip.style.left = (rect.left + rect.width / 2) + 'px';
  tip.style.top = (rect.top - 8) + 'px';
}

function hidePortTooltip() {
  if (portTooltipEl) { portTooltipEl.remove(); portTooltipEl = null; }
}

function updateNodePosition(nodeId) {
  renderAll();
}

// ── Selection ────────────────────────────────────────────────────────────────

function selectNode(id) {
  if (state.selected !== null) {
    document.getElementById(`node-${state.selected}`)?.classList.remove('selected');
  }
  state.selected = id;
  if (id !== null) {
    document.getElementById(`node-${id}`)?.classList.add('selected');
  }
  renderProps();
  renderTree();
}

// ── Properties (inline in top bar) ───────────────────────────────────────────

// Build an editable field HTML string for a param
function paramFieldHtml(key, schema, value) {
  let html = `<label>${key}</label>`;
  if (schema.options) {
    const ptype = schema.type === 'string' ? ' data-ptype="string"' : '';
    html += `<select data-param="${key}"${ptype}>`;
    for (const opt of schema.options) {
      const sel = (schema.type === 'string' ? value === opt : value == opt) ? 'selected' : '';
      html += `<option value="${opt}" ${sel}>${opt}</option>`;
    }
    html += `</select>`;
  } else if (schema.type === 'float') {
    html += `<input type="number" data-param="${key}" data-ptype="float" value="${value}" step="${schema.step || 0.1}" min="${schema.min ?? ''}" max="${schema.max ?? ''}">`;
  } else if (schema.type === 'string') {
    html += `<input type="text" data-param="${key}" data-ptype="string" value="${value || ''}" style="width:120px">`;
  } else {
    html += `<input type="number" data-param="${key}" value="${value}" min="${schema.min ?? ''}" max="${schema.max ?? ''}">`;
  }
  return html;
}

// Bind change handlers on [data-param] elements within a container, writing to targetNode
function bindParamHandlers(container, targetNode) {
  container.querySelectorAll('[data-param]').forEach(el => {
    if (isActiveLocked()) el.disabled = true;
    el.addEventListener('change', () => {
      if (isActiveLocked()) { setStatus('Cannot edit while training', { flash: true }); return; }
      const param = el.dataset.param;
      const ptype = el.dataset.ptype;
      if (ptype === 'string') {
        targetNode.params[param] = el.value;
      } else if (ptype === 'float') {
        targetNode.params[param] = parseFloat(el.value);
      } else {
        targetNode.params[param] = parseInt(el.value);
      }
      // Mark children as stale — they'll be regenerated on next build
      targetNode._paramsDirty = true;
    });
  });
}

function renderProps() {
  const container = document.getElementById('topbar-props');
  const crumbs = document.getElementById('breadcrumbs');
  const delBtn = document.getElementById('btn-delete-selected');
  if (delBtn) delBtn.style.display = state.selected !== null ? '' : 'none';

  if (state.selected === null) {
    crumbs.querySelector('.crumb-selected')?.remove();
    crumbs.querySelector('.crumb-sep-selected')?.remove();
    renderScopeProps(container);
    return;
  }

  const node = state.nodes.find(n => n.id === state.selected);
  if (!node) { container.innerHTML = ''; return; }
  const comp = getCompDef(node.type);
  if (!comp) { container.innerHTML = ''; return; }

  const level = SCOPE_LEVELS[state.currentScope];
  const isContainer = !!(getChildScope(node.type));

  // Add selected node to breadcrumbs
  crumbs.querySelector('.crumb-selected')?.remove();
  crumbs.querySelector('.crumb-sep-selected')?.remove();
  const sep = document.createElement('span');
  sep.className = 'crumb-sep crumb-sep-selected';
  sep.textContent = '\u25B8';
  crumbs.appendChild(sep);
  const crumb = document.createElement('span');
  crumb.className = 'crumb crumb-selected';
  crumb.style.color = comp.color || 'var(--text)';
  if (isContainer) {
    crumb.style.textDecoration = 'underline';
    crumb.style.cursor = 'pointer';
    crumb.id = 'btn-zoom-in';
  }
  crumb.textContent = comp.label;
  crumbs.appendChild(crumb);

  // Build prop crumbs (DOM order = rightmost first due to row-reverse)
  // Inherited values go rightmost (emitted first), own params go left (emitted last)
  const inherited = getInheritedValues();
  const ownKeys = new Set(Object.keys(comp.params || {}));
  const inheritedFiltered = inherited.filter(item => !ownKeys.has(item.key));
  let html = '';

  // Inherited (rightmost) — emitted first in DOM, pipeline order
  inheritedFiltered.forEach((item, i) => {
    html += `<span class="prop-crumb inherited" title="from ${item.source}"><label>${item.key}</label><span>${item.value}</span></span>`;
    if (i < inheritedFiltered.length - 1) html += `<span class="prop-sep">\u25C2</span>`;
  });

  // Separator between inherited and own
  const paramEntries = Object.entries(comp.params || {});
  if (inheritedFiltered.length > 0 && paramEntries.length > 0) {
    html += `<span class="prop-sep">\u25C2</span>`;
  }

  // Own params (leftmost) — emitted last in DOM
  paramEntries.forEach(([key, schema], i) => {
    html += `<span class="prop-crumb editable">${paramFieldHtml(key, schema, node.params[key])}</span>`;
    if (i < paramEntries.length - 1) html += `<span class="prop-sep">\u25C2</span>`;
  });

  container.innerHTML = html;
  bindParamHandlers(container, node);
  document.getElementById('btn-zoom-in')?.addEventListener('click', () => zoomIntoNode(node.id));
}

// Collect inherited values visible at the current scope.
// Returns an array in pipeline order (earliest first = rightmost in row-reverse display).
function getInheritedValues() {
  const items = []; // {key, value, source}
  const seen = new Set();
  const pipelineGraph = getPipelineGraph(getActiveCard());
  const pipelineNodes = pipelineGraph.nodes;

  // Pipeline-level: seq_len from windower
  const windowTypes = ['sequential_window', 'sliding_window', 'random_window', 'dataset'];
  for (const n of pipelineNodes) {
    if (windowTypes.includes(n.type) && n.params?.seq_len) {
      items.push({ key: 'seq_len', value: n.params.seq_len, source: getCompDef(n.type)?.label || n.type });
      seen.add('seq_len');
    }
  }

  // Walk ancestor scopes — each parent's params in scope order
  let graph = state.rootGraph;
  for (const scope of state.scopeStack) {
    const node = graph.nodes.find(n => n.id === scope.nodeId);
    if (!node) break;
    const comp = getCompDef(node.type);
    if (comp) {
      for (const [key, schema] of Object.entries(comp.params || {})) {
        if (node.params[key] !== undefined && !seen.has(key)) {
          items.push({ key, value: node.params[key], source: comp.label });
          seen.add(key);
        }
      }
    }
    if (node.children) graph = node.children;
  }
  return items;
}

function renderScopeProps(container) {
  if (state.scopeStack.length === 0) {
    container.innerHTML = '';
    return;
  }
  const parentNode = getParentNode();
  if (!parentNode) { container.innerHTML = ''; return; }
  const comp = getCompDef(parentNode.type);
  if (!comp) { container.innerHTML = ''; return; }

  const inherited = getInheritedValues();
  const ownKeys = new Set(Object.keys(comp.params || {}));
  const inheritedFiltered = inherited.filter(item => !ownKeys.has(item.key));
  let html = '';

  // Inherited (rightmost) — pipeline order
  inheritedFiltered.forEach((item, i) => {
    html += `<span class="prop-crumb inherited" title="from ${item.source}"><label>${item.key}</label><span>${item.value}</span></span>`;
    if (i < inheritedFiltered.length - 1) html += `<span class="prop-sep">\u25C2</span>`;
  });

  // Separator between inherited and own
  const paramEntries = Object.entries(comp.params || {});
  if (inheritedFiltered.length > 0 && paramEntries.length > 0) {
    html += `<span class="prop-sep">\u25C2</span>`;
  }

  // Own editable params (leftmost)
  paramEntries.forEach(([key, schema], i) => {
    html += `<span class="prop-crumb editable">${paramFieldHtml(key, schema, parentNode.params[key])}</span>`;
    if (i < paramEntries.length - 1) html += `<span class="prop-sep">\u25C2</span>`;
  });

  container.innerHTML = html;
  bindParamHandlers(container, parentNode);
}

// ── Edge rendering ───────────────────────────────────────────────────────────

function getPortPosition(nodeId, portId, isOutput) {
  if (nodeId === -1) {
    return getBoundaryPortPosition(portId, isOutput);
  }
  const node = state.nodes.find(n => n.id === nodeId);
  if (!node) return { x: 0, y: 0 };

  // Use cached placement if available
  const nodeCache = portPlacementCache.get(nodeId);
  if (nodeCache) {
    const key = `${isOutput ? 'out' : 'in'}:${portId}`;
    const cached = nodeCache.get(key);
    if (cached) {
      return { x: node.x + cached.x, y: node.y + cached.y, side: cached.side };
    }
  }

  // Fallback to default placement
  const comp = getCompDef(node.type);
  if (!comp) return { x: node.x, y: node.y };
  const ports = isOutput ? (comp.ports.out || []) : (comp.ports.in || []);
  const idx = ports.findIndex(p => p.id === portId);
  if (idx < 0) return { x: node.x, y: node.y };
  const headerH = 36;
  const portSpacing = 28;
  const w = 180;
  const py = headerH + idx * portSpacing + portSpacing / 2;
  return {
    x: node.x + (isOutput ? w : 0),
    y: node.y + py,
    side: isOutput ? 'right' : 'left',
  };
}

function edgePath(x1, y1, x2, y2, fromSide, toSide) {
  const dist = Math.hypot(x2 - x1, y2 - y1);
  const tension = Math.max(30, dist * 0.4);

  // Control point offsets based on port side
  function controlOffset(side) {
    switch (side) {
      case 'right':  return { dx:  tension, dy: 0 };
      case 'left':   return { dx: -tension, dy: 0 };
      case 'bottom': return { dx: 0, dy:  tension };
      case 'top':    return { dx: 0, dy: -tension };
      default:       return { dx:  tension, dy: 0 };
    }
  }

  const c1 = controlOffset(fromSide || 'right');
  const c2 = controlOffset(toSide || 'left');

  return `M ${x1} ${y1} C ${x1 + c1.dx} ${y1 + c1.dy}, ${x2 + c2.dx} ${y2 + c2.dy}, ${x2} ${y2}`;
}

function getEdgeDataType(edge) {
  if (edge.from.nodeId === -1) {
    // Boundary source — look up port type from parent component or pipeline boundary
    const parentNode = getParentNode();
    const portDefs = parentNode
      ? (getCompDef(parentNode.type)?.ports.in || [])
      : (PIPELINE_BOUNDARY.ports.in || []);
    const port = portDefs.find(p => p.id === edge.from.portId);
    if (port) return port.dataType;
  }
  const fromNode = state.nodes.find(n => n.id === edge.from.nodeId);
  const fromComp = fromNode ? getCompDef(fromNode.type) : null;
  const fromPort = fromComp ? (fromComp.ports.out || []).find(p => p.id === edge.from.portId) : null;
  return fromPort?.dataType;
}

function renderEdge(edge) {
  let el = document.getElementById(`edge-${edge.id}`);
  const from = getPortPosition(edge.from.nodeId, edge.from.portId, true);
  const to = getPortPosition(edge.to.nodeId, edge.to.portId, false);
  const d = edgePath(from.x, from.y, to.x, to.y, from.side, to.side);

  if (!el) {
    el = document.createElementNS(SVG_NS, 'path');
    el.id = `edge-${edge.id}`;
    el.classList.add('edge');
    if (edge.from.nodeId === -1 || edge.to.nodeId === -1) {
      el.classList.add('boundary-edge');
    }
    const dt = getEdgeDataType(edge);
    const dtColor = state.registry.dataTypes[dt]?.color || '#888';
    el.setAttribute('stroke', dtColor);
    el.addEventListener('click', () => deleteEdge(edge.id));
    edgesLayer.appendChild(el);
  }
  el.setAttribute('d', d);
}

function updateBoundaryEdges() {
  for (const edge of state.edges) {
    if (edge.from.nodeId === -1 || edge.to.nodeId === -1) {
      const el = document.getElementById(`edge-${edge.id}`);
      if (el) {
        const from = getPortPosition(edge.from.nodeId, edge.from.portId, true);
        const to = getPortPosition(edge.to.nodeId, edge.to.portId, false);
        el.setAttribute('d', edgePath(from.x, from.y, to.x, to.y, from.side, to.side));
      }
    }
  }
}

function removeEdgeEl(id) {
  document.getElementById(`edge-${id}`)?.remove();
}

function deleteEdge(id) {
  if (isActiveLocked()) { setStatus('Cannot edit while training', { flash: true }); return; }
  state.edges = state.edges.filter(e => e.id !== id);
  removeEdgeEl(id);
  updateStatus();
}

function addEdge(fromNodeId, fromPortId, toNodeId, toPortId) {
  if (isActiveLocked()) { setStatus('Cannot edit while training', { flash: true }); return; }
  if (fromNodeId === toNodeId) return;
  const exists = state.edges.some(e =>
    e.from.nodeId === fromNodeId && e.from.portId === fromPortId &&
    e.to.nodeId === toNodeId && e.to.portId === toPortId
  );
  if (exists) return;
  // Remove existing connection to non-multi inputs
  const toNode = state.nodes.find(n => n.id === toNodeId);
  const toComp = toNode ? getCompDef(toNode.type) : null;
  const toPortDef = toComp ? (toComp.ports.in || []).find(p => p.id === toPortId) : null;
  if (!toPortDef?.multi) {
    state.edges = state.edges.filter(e => {
      if (e.to.nodeId === toNodeId && e.to.portId === toPortId) {
        removeEdgeEl(e.id);
        return false;
      }
      return true;
    });
  }
  const edge = {
    id: state.nextId++,
    from: { nodeId: fromNodeId, portId: fromPortId },
    to: { nodeId: toNodeId, portId: toPortId },
  };
  state.edges.push(edge);
  renderEdge(edge);
  updateStatus();
}

// ── Port compatibility ───────────────────────────────────────────────────────

function canConnect(fromDataType, toDataType) {
  if (fromDataType === toDataType) return true;
  // Tensors are broadly compatible (stream ↔ stream, etc)
  const tensorTypes = ['stream', 'tensor'];
  if (tensorTypes.includes(fromDataType) && tensorTypes.includes(toDataType)) return true;
  return false;
}

function highlightCompatibility(portG) {
  const conn = state.connecting;
  if (!conn) return;
  const isOutput = portG.dataset.isOutput === 'true';
  if (isOutput === conn.isOutput) {
    portG.classList.add('incompatible');
    return;
  }
  if (parseInt(portG.dataset.nodeId) === conn.nodeId) {
    portG.classList.add('incompatible');
    return;
  }
  const fromDT = conn.isOutput ? conn.portDef.dataType : portG.dataset.dataType;
  const toDT = conn.isOutput ? portG.dataset.dataType : conn.portDef.dataType;
  if (canConnect(fromDT, toDT)) {
    portG.classList.add('compatible');
  } else {
    portG.classList.add('incompatible');
  }
}

// ── D3 Zoom ─────────────────────────────────────────────────────────────────
// D3 handles zoom/pan via a single SVG transform — no DOM rebuild on scroll.

let d3ZoomBehavior;

function setupD3Zoom() {
  const svg = d3.select('#canvas');

  d3ZoomBehavior = d3.zoom()
    .scaleExtent([0.02, 200])
    .on('zoom', (event) => {
      const t = event.transform;
      state.viewX = t.x;
      state.viewY = t.y;
      state.zoom = t.k;

      // Apply transform directly — no renderAll(), no DOM rebuild
      const tf = `translate(${t.x},${t.y}) scale(${t.k})`;
      nodesLayer.setAttribute('transform', tf);
      edgesLayer.setAttribute('transform', tf);
      draftEdge.setAttribute('transform', tf);
      // Re-render boundary ports (scale with zoom) and re-path their edges
      updateViewportBorder();
      updateBoundaryEdges();
    });

  svg.call(d3ZoomBehavior)
    .on('dblclick.zoom', null);  // we use dblclick for containers

  console.log('D3 zoom initialized, version:', d3.version);
}

// ── Canvas interaction ───────────────────────────────────────────────────────

function setupCanvasEvents() {
  canvas.addEventListener('mousedown', onCanvasMouseDown);
  canvas.addEventListener('mousemove', onCanvasMouseMove);
  canvas.addEventListener('mouseup', onCanvasMouseUp);
  canvas.addEventListener('dblclick', onCanvasDblClick);
  window.addEventListener('keydown', onKeyDown);
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  setupPalettePopup();
}

function onCanvasDblClick(e) {
  if (isCanvasBg(e.target)) {
    selectNode(null);  // deselect any node, show parent container's props
  }
}

function setupPalettePopup() {
  const popup = document.getElementById('palette-popup');
  const tab = document.getElementById('palette-tab');
  const wrap = document.getElementById('canvas-wrap');

  tab.addEventListener('click', () => popup.classList.add('open'));

  // Mouse near bottom edge opens the palette
  wrap.addEventListener('mousemove', (e) => {
    const rect = wrap.getBoundingClientRect();
    const distFromBottom = rect.bottom - e.clientY;
    if (distFromBottom < 8 && !popup.classList.contains('open')) {
      popup.classList.add('open');
    }
  });

  // Mouse leaving the popup closes it (unless pinned open by a button)
  popup.addEventListener('mouseleave', (e) => {
    if (palettePinned) return;
    if (e.clientY < popup.getBoundingClientRect().top + 10) {
      popup.classList.remove('open');
    }
  });

  // Click on canvas area clears pin and closes popup
  wrap.addEventListener('click', (e) => {
    if (palettePinned && !popup.contains(e.target)) {
      palettePinned = false;
      popup.classList.remove('open');
    }
  });

  // Tab switching
  document.querySelectorAll('.palette-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.palette-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.palette-tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`palette-tab-${btn.dataset.tab}`).classList.add('active');
    });
  });
}

function screenToWorld(sx, sy) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (sx - rect.left - state.viewX) / state.zoom,
    y: (sy - rect.top - state.viewY) / state.zoom,
  };
}

let _lastNodeClick = { id: null, time: 0 };

function onNodeMouseDown(e, nodeId) {
  if (e.target.closest('.port')) return;
  e.stopPropagation();

  const now = Date.now();
  if (_lastNodeClick.id === nodeId && now - _lastNodeClick.time < 400) {
    // Double-click detected
    _lastNodeClick = { id: null, time: 0 };
    const node = state.nodes.find(n => n.id === nodeId);
    if (node) {
      const level = SCOPE_LEVELS[state.currentScope];
      if (getChildScope(node.type)) {
        zoomIntoNode(nodeId);
        return;
      }
    }
  }
  _lastNodeClick = { id: nodeId, time: now };

  selectNode(nodeId);
  const node = state.nodes.find(n => n.id === nodeId);
  const world = screenToWorld(e.clientX, e.clientY);
  state.dragging = {
    nodeId,
    offsetX: world.x - node.x,
    offsetY: world.y - node.y,
  };
}

function onPortMouseDown(e, nodeId, portDef, isOutput, localX, localY) {
  e.stopPropagation();
  const node = state.nodes.find(n => n.id === nodeId);
  state.connecting = {
    nodeId,
    portId: portDef.id,
    portDef,
    isOutput,
    startX: node.x + localX,
    startY: node.y + localY,
  };
  draftEdge.setAttribute('stroke', state.registry.dataTypes[portDef.dataType]?.color || '#888');
}

function isCanvasBg(target) {
  return target === canvas || target.classList.contains('grid-bg') || target.tagName === 'svg' || target.closest('rect.grid-bg');
}

function onCanvasMouseDown(e) {
  // Only deselect on single click — don't interfere with dblclick on nodes
  if (isCanvasBg(e.target) && state.selected !== null) {
    selectNode(null);
  }
}

function onCanvasMouseMove(e) {
  if (state.dragging) {
    const world = screenToWorld(e.clientX, e.clientY);
    const node = state.nodes.find(n => n.id === state.dragging.nodeId);
    node.x = Math.round((world.x - state.dragging.offsetX) / 10) * 10;
    node.y = Math.round((world.y - state.dragging.offsetY) / 10) * 10;
    updateNodePosition(node.id);
  }

  if (state.connecting) {
    const world = screenToWorld(e.clientX, e.clientY);
    const { startX, startY, isOutput } = state.connecting;
    const d = isOutput
      ? edgePath(startX, startY, world.x, world.y)
      : edgePath(world.x, world.y, startX, startY);
    draftEdge.setAttribute('d', d);
  }
}

function onCanvasMouseUp(e) {
  if (state.connecting) {
    const portEl = e.target.closest('.port');
    if (portEl) {
      const targetNodeId = parseInt(portEl.dataset.nodeId);
      const targetPortId = portEl.dataset.portId;
      const targetIsOutput = portEl.dataset.isOutput === 'true';

      if (targetIsOutput !== state.connecting.isOutput && targetNodeId !== state.connecting.nodeId) {
        const fromDT = state.connecting.isOutput ? state.connecting.portDef.dataType : portEl.dataset.dataType;
        const toDT = state.connecting.isOutput ? portEl.dataset.dataType : state.connecting.portDef.dataType;

        if (canConnect(fromDT, toDT)) {
          if (state.connecting.isOutput) {
            addEdge(state.connecting.nodeId, state.connecting.portId, targetNodeId, targetPortId);
          } else {
            addEdge(targetNodeId, targetPortId, state.connecting.nodeId, state.connecting.portId);
          }
        }
      }
    }
    state.connecting = null;
    draftEdge.setAttribute('d', '');
    document.querySelectorAll('.port.compatible, .port.incompatible').forEach(el => {
      el.classList.remove('compatible', 'incompatible');
    });
  }

  state.dragging = null;
}

function onKeyDown(e) {
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (state.selected !== null) {
      deleteNode(state.selected);
    }
  }
  if (e.key === 'Escape' && state.scopeStack.length > 0) {
    navigateToDepth(state.scopeStack.length - 2);
  }
}

function ensureChildren(node) {
  if (!node.children) {
    node.children = { nodes: [], edges: [] };
    const defaults = getDefaultChildren(node.type, node.params);
    if (defaults) {
      node.children = defaults;
      for (const n of node.children.nodes) {
        if (n.id >= state.nextId) state.nextId = n.id + 1;
      }
      for (const e of node.children.edges) {
        if (e.id >= state.nextId) state.nextId = e.id + 1;
      }
    }
  }
}

// Recursively ensure all children graphs are populated (for graph-mode build)
function ensureAllChildren(graph) {
  if (!graph || !graph.nodes) return;
  for (const node of graph.nodes) {
    // If params were changed, regenerate children from new params
    if (node._paramsDirty && node.children) {
      node.children = null;
      delete node._paramsDirty;
    }
    ensureChildren(node);
    if (node.children) {
      ensureAllChildren(node.children);
    }
  }
}

// ── Toolbar (delegated to active card) ───────────────────────────────────────

function doDemo() {
  state.scopeStack = [];
  state.currentScope = 'project';
  state.nodes = projectGraph.nodes;
  state.edges = projectGraph.edges;
  loadDemo();
  renderBreadcrumbs();
  renderCardList();
}

function exportGraph() {
  const card = getActiveCard();
  const name = card.name || 'untitled';
  const pipelineGraph = getPipelineGraph(card);
  const data = {
    version: 4,
    name: name,
    graph: serializeGraph(pipelineGraph),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
  a.click();
  URL.revokeObjectURL(url);
  setStatus('Design exported.');
}

function serializeGraph(graph) {
  return {
    nodes: graph.nodes.map(n => {
      const out = { id: n.id, type: n.type, x: n.x, y: n.y, params: n.params };
      if (n.children && n.children.nodes.length > 0) {
        out.children = serializeGraph(n.children);
      }
      return out;
    }),
    edges: graph.edges.map(e => ({ id: e.id, from: e.from, to: e.to })),
  };
}

function importGraph(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      const card = getActiveCard();
      const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
      if ((data.version >= 2) && data.graph) {
        engineNode.children = deserializeGraph(data.graph);
      } else {
        const pg = engineNode.children;
        for (const n of data.nodes || []) {
          pg.nodes.push({ ...n, label: getCompDef(n.type)?.label || n.type, children: null });
          if (n.id >= projectNextId) projectNextId = n.id + 1;
        }
        for (const ed of data.edges || []) {
          pg.edges.push({ id: projectNextId++, from: ed.from, to: ed.to });
        }
      }
      if (data.name) card.name = data.name;
      card.engine.built = false;
      loadCardIntoCanvas(activeCardIdx);
      renderCardList();
      setStatus('Design imported into ' + (card.name || 'engine'));
    } catch (err) {
      setStatus('Import failed: ' + err.message);
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

function deserializeGraph(data) {
  const graph = { nodes: [], edges: [] };
  for (const n of data.nodes || []) {
    const node = { id: n.id, type: n.type, x: n.x, y: n.y, params: n.params || {}, label: getCompDef(n.type)?.label || n.type, children: null };
    if (n.children) {
      node.children = deserializeGraph(n.children);
    }
    graph.nodes.push(node);
    if (n.id >= state.nextId) state.nextId = n.id + 1;
  }
  for (const e of data.edges || []) {
    graph.edges.push({ id: e.id || state.nextId++, from: e.from, to: e.to });
    if (e.id >= state.nextId) state.nextId = e.id + 1;
  }
  return graph;
}

function fitToView() {
  if (state.nodes.length === 0) {
    state.viewX = 0;
    state.viewY = 0;
    state.zoom = 1;
    if (d3ZoomBehavior) {
      d3.select('#canvas').call(d3ZoomBehavior.transform, d3.zoomIdentity);
    }
    return;
  }
  const rect = canvas.getBoundingClientRect();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of state.nodes) {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x + 180);
    maxY = Math.max(maxY, n.y + 100);
  }
  const padding = 60;
  const w = maxX - minX + padding * 2;
  const h = maxY - minY + padding * 2;
  state.zoom = Math.min(rect.width / w, rect.height / h, 2);
  state.viewX = (rect.width - w * state.zoom) / 2 - minX * state.zoom + padding * state.zoom;
  state.viewY = (rect.height - h * state.zoom) / 2 - minY * state.zoom + padding * state.zoom;

  // Sync D3's internal transform state (this also applies the transform visually)
  if (d3ZoomBehavior) {
    const t = d3.zoomIdentity.translate(state.viewX, state.viewY).scale(state.zoom);
    d3.select('#canvas').call(d3ZoomBehavior.transform, t);
  }
}

function clearCanvas(silent) {
  if (state.currentScope === 'project') { setStatus('Cannot clear project scope'); return; }
  if (!silent && state.nodes.length > 0 && !confirm('Clear current scope?')) return;
  nodesLayer.innerHTML = '';
  edgesLayer.innerHTML = '';
  state.nodes.length = 0;
  state.edges.length = 0;
  state.selected = null;
  renderProps();
  updateStatus();
}

// ── Status ───────────────────────────────────────────────────────────────────

function updateStatus() {
  const scopeLabel = SCOPE_LEVELS[state.currentScope]?.label || state.currentScope;
  scopeInfoEl.textContent = `${scopeLabel}: ${state.nodes.length} components, ${state.edges.length} connections`;
}

function setStatus(msg, { flash } = {}) {
  statusEl.textContent = msg;
  if (flash) {
    statusEl.classList.remove('flash-warn');
    void statusEl.offsetWidth;          // force reflow to restart animation
    statusEl.classList.add('flash-warn');
  }
  setTimeout(updateStatus, 3000);
}


// ── Utilities ────────────────────────────────────────────────────────────────

function darken(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${Math.round(r * amount)}, ${Math.round(g * amount)}, ${Math.round(b * amount)})`;
}

// ── Demo graph ───────────────────────────────────────────────────────────────

function loadDemo() {
  const card = getActiveCard();
  const id = () => projectNextId++;

  const tokId = id(), winId = id(), initId = id(), coopId = id(), lossId = id(), optId = id();

  // Populate the engine node's pipeline
  const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
  engineNode.children = {
    nodes: [
      { id: tokId, type: "char_tokenizer", x: -200, y: 100, params: {}, children: null },
      { id: winId, type: "sequential_window", x: 0, y: 100, params: { seq_len: 128 }, children: null },
      { id: initId, type: "zero_init", x: 0, y: -60, params: {}, children: null },
      { id: coopId, type: "cooperative", x: 280, y: 60, params: { stream_dim: 64 }, children: null },
      { id: lossId, type: "loss", x: 560, y: 100, params: {}, children: null },
      { id: optId, type: "optimizer", x: 560, y: -60, params: { algorithm: "adam", learning_rate: 0.0003, beta1: 0.9, beta2: 0.999 }, children: null },
    ],
    edges: [
      { id: id(), from: { nodeId: -1, portId: "raw_text" }, to: { nodeId: tokId, portId: "raw_text" } },
      { id: id(), from: { nodeId: tokId, portId: "token_ids" }, to: { nodeId: winId, portId: "token_ids" } },
      { id: id(), from: { nodeId: winId, portId: "input_ids" }, to: { nodeId: coopId, portId: "token_ids" } },
      { id: id(), from: { nodeId: winId, portId: "target_ids" }, to: { nodeId: lossId, portId: "targets" } },
      { id: id(), from: { nodeId: initId, portId: "stream_out" }, to: { nodeId: coopId, portId: "stream_in" } },
      { id: id(), from: { nodeId: coopId, portId: "logits_out" }, to: { nodeId: lossId, portId: "logits_in" } },
    ],
  };

  // Pre-populate children so they're visible when zooming in
  const coopNode = engineNode.children.nodes.find(n => n.id === coopId);
  if (coopNode) ensureChildren(coopNode);

  card.engine.built = false;
  card.engine.lossHistory = [];

  // Navigate to project level, then auto-zoom into the engine
  state.scopeStack = [];
  state.currentScope = 'project';
  state.nodes = projectGraph.nodes;
  state.edges = projectGraph.edges;
  zoomIntoNode(card.nodeId);
}

// ── Canonical Model Hash ──────────────────────────────────────────────────────
// Hashes the architectural structure of the graph (types, connections, params
// that affect architecture) but excludes visual layout (x, y positions).
// Same architecture → same hash, regardless of canvas arrangement.

function sortedObj(obj) {
  const out = {};
  for (const k of Object.keys(obj).sort()) {
    // Exclude non-architectural params (name is cosmetic)
    if (k === 'name') continue;
    out[k] = obj[k];
  }
  return out;
}

async function sha256hex(str) {
  const data = new TextEncoder().encode(str);
  const buf = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
}

// Recursive hash: each node's hash = SHA256(type + params + sorted children hashes)
// Leaf nodes hash just type + params. Container nodes incorporate their subtree.
// Edge topology uses node hashes (not IDs) so structurally identical subgraphs
// produce identical hashes regardless of node ID assignment.
// Returns full 256-bit hex hash. Stores both full and display (8-char) on each node.
async function computeNodeHash(node) {
  let childrenHash = '';
  if (node.children && node.children.nodes.length > 0) {
    // Hash each child node recursively
    const childHashes = await Promise.all(
      node.children.nodes.map(n => computeNodeHash(n))
    );

    // Build a map from node ID → node hash for edge encoding
    const idToHash = {};
    for (const n of node.children.nodes) {
      idToHash[n.id] = n._hashFull;
    }
    // Boundary ports (nodeId: -1) get a fixed sentinel
    idToHash[-1] = 'boundary';

    // Encode edges using node hashes instead of IDs
    // This makes topology position-independent
    const edgeKeys = node.children.edges.map(e =>
      `${idToHash[e.from.nodeId] || e.from.nodeId}:${e.from.portId}->` +
      `${idToHash[e.to.nodeId] || e.to.nodeId}:${e.to.portId}`
    );
    edgeKeys.sort();

    // Sort child hashes for order-independence
    childHashes.sort();
    childrenHash = childHashes.join(',') + '|' + edgeKeys.join(',');
  }
  const canonical = JSON.stringify({ type: node.type, params: sortedObj(node.params || {}) }) + childrenHash;
  const fullHash = await sha256hex(canonical);
  node._hashFull = fullHash;           // full 256-bit (64 hex chars)
  node._hash = fullHash.slice(0, 8);   // display-friendly 32-bit
  return fullHash;
}

// Compute the model hash for a pipeline graph (engine node's children)
// Returns full 256-bit hash; caller truncates for display
async function computeModelHash(graph) {
  if (!graph.nodes.length) return '0'.repeat(64);
  const virtualRoot = { type: '__pipeline__', params: {}, children: graph };
  const fullHash = await computeNodeHash(virtualRoot);
  return fullHash;
}

// ── Engine (now per-card) ─────────────────────────────────────────────────────

// Convenience accessor for active card's engine
function getEngine() { return getActiveCard().engine; }

// Find the effective optimizer config by walking the graph hierarchy
// (nearest optimizer node wins — deeper scopes override)
function resolveOptimizer(graph) {
  // Check this scope for an optimizer node
  const optNode = graph.nodes.find(n => n.type === 'optimizer');
  if (optNode) return optNode.params;
  return null;
}

function resolveOptimizerForGraph(rootGraph) {
  // Walk top-down: pipeline → ensemble → expert internals
  // For now just find the first optimizer in the pipeline (top scope)
  // Full inheritance resolution happens on the server
  const opt = resolveOptimizer(rootGraph);
  return opt || { algorithm: 'adam', learning_rate: 0.0003, beta1: 0.9, beta2: 0.999 };
}

async function doSaveWeights(name, card) {
  card = card || getActiveCard();
  const eng = card.engine;
  if (!eng.modelHash) return;
  name = name || card.name || 'default';
  try {
    const data = await apiPost('/api/save', { hash: eng.modelHash, name, card_id: card.id });
    if (data.saved) {
      setStatus(`Weights saved: ${eng.modelHash.slice(0,8)}/${name}`);
    } else {
      setStatus('Save failed: ' + (data.error || 'unknown'));
    }
  } catch (err) {
    setStatus('Save error: ' + err.message);
  }
}

async function fetchRunMetrics(hash, name) {
  return fetchJSON(`/api/saves/${hash}/${name}`);
}

async function doLoadWeights(name, card) {
  card = card || getActiveCard();
  const eng = card.engine;
  if (!eng.modelHash) return;
  name = name || 'default';
  try {
    const data = await apiPost('/api/load', { hash: eng.modelHash, name, card_id: card.id });
    if (data.loaded) {
      setStatus(`Weights loaded: ${eng.modelHash.slice(0,8)}/${name}`);
      return true;
    }
    return false;
  } catch { return false; }
}

async function autoLoadLatest(card) {
  card = card || getActiveCard();
  const eng = card.engine;
  if (!eng.modelHash) return;
  const data = await fetchJSON(`/api/saves/${eng.modelHash}`);
  if (data?.saves?.length > 0) {
    const loaded = await doLoadWeights(data.saves[0].name, card);
    if (loaded) setStatus(`Auto-loaded weights: ${data.saves[0].name}`);
  }
}

function showRunsPicker() {
  const eng = getEngine();
  if (!eng.modelHash) { setStatus('Build a model first'); return; }
  const popup = document.getElementById('runs-popup');
  const btn = document.querySelector('.project-card.expanded .btn-runs');
  if (!btn) return;
  const rect = btn.getBoundingClientRect();
  popup.style.left = Math.max(0, rect.left - 260) + 'px';
  popup.style.top = rect.bottom + 4 + 'px';
  popup.style.display = 'block';
  loadRunsList();
}

async function loadRunsList() {
  const eng = getEngine();
  const list = document.getElementById('runs-list');
  list.innerHTML = '<div class="runs-empty">Loading...</div>';
  try {
    const data = await fetchJSON(`/api/saves/${eng.modelHash}`, {});
    if (!data.saves || data.saves.length === 0) {
      list.innerHTML = '<div class="runs-empty">No saved runs for this model</div>';
      return;
    }
    list.innerHTML = '';
    for (const save of data.saves) {
      const item = document.createElement('div');
      item.className = 'run-item';
      const lossStr = save.final_loss != null ? `loss: ${save.final_loss.toFixed(4)}` : '';
      const dateStr = save.modified ? new Date(save.modified).toLocaleString() : '';
      item.innerHTML = `
        <div style="flex:1">
          <div class="run-name">${save.name}</div>
          <div class="run-meta">${dateStr}</div>
        </div>
        ${lossStr ? `<div class="run-loss">${lossStr}</div>` : ''}
      `;
      item.addEventListener('click', async () => {
        document.getElementById('runs-popup').style.display = 'none';
        const rnEl = document.getElementById('train-run-name');
        if (rnEl) rnEl.value = save.name;
        const loaded = await doLoadWeights(save.name);
        if (loaded) setStatus(`Loaded weights: ${save.name}`);
        else setStatus(`Failed to load: ${save.name}`);
      });
      list.appendChild(item);
    }
  } catch {
    list.innerHTML = '<div class="runs-empty">Failed to fetch saves</div>';
  }
}

async function doResetWeights() {
  const card = getActiveCard();
  const eng = card.engine;
  if (!eng.built) return;
  if (eng.training) { setStatus('Cannot reset while training', { flash: true }); return; }
  if (!confirm('Reset all weights to random initialization? Training progress will be lost.')) return;
  try {
    const resetPipeline = getPipelineGraph(card || getActiveCard());
    const payload = { version: 2, graph: serializeGraph(resetPipeline), card_id: card.id };
    const data = await apiPost('/api/build', payload);
    if (data.built) {
      eng.lossHistory = [];
      setStatus('Weights reset — model reinitialized from scratch');
      renderCardList();
    } else {
      setStatus('Reset failed: ' + (data.details || data.error || 'unknown'));
    }
  } catch (err) {
    setStatus('Reset error: ' + err.message);
  }
}

async function doBuild(card) {
  if (!card || card instanceof Event) card = getActiveCard();
  const eng = card.engine;
  if (eng.training) { setStatus('Cannot build while training', { flash: true }); return; }

  try {
    const pipelineGraph = getPipelineGraph(card);
    const graphMode = card.graphMode || false;
    // In graph mode, ensure all children are populated before serializing
    if (graphMode) ensureAllChildren(pipelineGraph);
    const clientHash = await computeModelHash(pipelineGraph);
    const dataFile = document.getElementById('train-data-file')?.value?.trim() || 'data/input.txt';
    const payload = { version: 2, graph: serializeGraph(pipelineGraph), card_id: card.id, hash: clientHash, data_file: dataFile, graph_mode: graphMode };
    const data = await apiPost('/api/build', payload);

    if (data.built) {
      eng.built = true;
      eng.lossHistory = [];
      // Server returns its own hash once implemented; use client hash as primary for now
      eng.modelHash = data.model_hash || clientHash;
      if (data.model_hash && data.hash_match === false) {
        console.warn(`Hash mismatch: client=${clientHash} server=${data.model_hash}`);
        setStatus(`Warning: client/server hash mismatch`);
      }
      eng.summary = data.summary;
      eng.graphWarnings = data.graph_warnings || [];
      eng.lastError = null;
      const warnCount = eng.graphWarnings.length;
      const warnSuffix = warnCount > 0 ? ` (${warnCount} warning${warnCount > 1 ? 's' : ''})` : '';
      setStatus(`Model built: ${fmtParams(data.summary.total_params)} params, ${data.summary.n_experts} experts${warnSuffix}`);
      renderCardList();
      await autoLoadLatest(card);
    } else {
      const errors = data.details || data.errors || [data.error || 'Unknown error'];
      alert('Build failed:\n' + errors.join('\n'));
    }
  } catch (err) {
    alert('Build error: ' + err.message);
  }
}

async function doTrainAll() {
  const starredAll = cards.filter(c => c.starred);
  if (starredAll.length === 0) {
    alert('No starred engines to train.');
    return;
  }

  // Auto-build any unbuilt starred cards
  const unbuilt = starredAll.filter(c => !c.engine.built);
  if (unbuilt.length > 0) {
    setStatus(`Building ${unbuilt.length} unbuilt engine(s)...`);
    await Promise.all(unbuilt.map(c => doBuild(c)));
  }

  const starred = starredAll.filter(c => c.engine.built);
  if (starred.length === 0) {
    alert('Build failed for all starred engines.');
    return;
  }

  const steps = parseInt(document.getElementById('train-steps')?.value) || 1000;
  const runName = document.getElementById('train-run-name')?.value?.trim() || '';

  // Show stop button
  document.getElementById('btn-train-all').style.display = 'none';
  document.getElementById('btn-stop-all').style.display = '';

  // Create training session
  trainSession = { cards: starred, active: true };
  renderCardList();
  renderTrainSession();

  // Train all starred engines in parallel (server holds N models keyed by card_id)
  await Promise.all(starred.map(card => trainSingleCard(card, steps, runName)));

  trainSession.active = false;
  document.getElementById('btn-train-all').style.display = '';
  document.getElementById('btn-stop-all').style.display = 'none';
  renderCardList();
  renderTrainSession();
}

async function trainSingleCard(card, steps, runName) {
  const eng = card.engine;
  const name = runName || card.name || '';
  try {
    const data = await apiPost('/api/train/start', { steps, name, card_id: card.id });
    if (data.started) {
      eng.training = true;
      eng.lossHistory = [];
      renderCardList();
      setStatus(`Training ${card.name || 'untitled'}...`);
      await watchTraining(card);
    } else {
      setStatus(`${card.name || 'untitled'}: ${data.error || 'Failed to start'}`);
    }
  } catch (err) {
    setStatus(`Train error (${card.name || 'untitled'}): ${err.message}`);
  }
}

function watchTraining(card) {
  const eng = card.engine;
  let resolved = false;

  function finish(avgLoss) {
    if (resolved) return;
    resolved = true;
    eng.training = false;
    // Accumulate total steps
    const lastStep = eng.lossHistory.length > 0 ? eng.lossHistory[eng.lossHistory.length - 1].step : 0;
    eng.totalSteps = (eng.totalSteps || 0) + lastStep;
    if (avgLoss != null) {
      setStatus(`${card.name || 'untitled'}: done. Loss: ${avgLoss.toFixed(4)} (${eng.totalSteps} total steps)`);
    }
    renderCardList();
    // Always auto-save — weights are keyed by model hash
    const rn = document.getElementById('train-run-name')?.value?.trim() || card.name || 'default';
    doSaveWeights(rn, card);
  }

  function handleStepData(data) {
    // Update loss history
    if (eng.lossHistory.length === 0 || eng.lossHistory[eng.lossHistory.length - 1].step !== data.step) {
      eng.lossHistory.push({ step: data.step, loss: data.avg_loss });
      drawAllSparklines();
      // Update session chart in train panel
      const sessionCanvas = document.querySelector('#train-panel-session .session-chart');
      if (sessionCanvas) drawSessionChart(sessionCanvas);
    }

    // Update session slot for this engine
    const slotEl = document.querySelector(`.session-slot[data-card-id="${card.id}"]`);
    if (slotEl) {
      const elapsed = data.elapsed_sec || 0;
      const stepsPerSec = elapsed > 0 ? (data.step / elapsed).toFixed(1) : '\u2014';
      const timeStr = elapsed >= 60
        ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
        : `${elapsed.toFixed(1)}s`;
      const lossEl = slotEl.querySelector('.session-slot-loss');
      const stepEl = slotEl.querySelector('.session-slot-step');
      if (lossEl) lossEl.textContent = data.avg_loss.toFixed(4);
      if (stepEl) stepEl.textContent = `Step ${data.step}/${data.steps} \u00b7 ${timeStr} \u00b7 ${stepsPerSec} steps/s`;
    }
  }

  return new Promise((resolve) => {
    // WebSocket for real-time step updates
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${location.host}/ws/train?card_id=${card.id}`);
    eng._trainWs = ws;

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === 'done') {
          ws.close();
          eng._trainWs = null;
          finish(data.avg_loss);
          resolve();
          return;
        }
        handleStepData(data);
      } catch (e) {
        console.error('WS message error:', e);
      }
    };

    ws.onerror = () => {};

    ws.onclose = () => {
      eng._trainWs = null;
    };

    // Fallback: poll for completion every 3s in case WS misses events
    const fallback = setInterval(async () => {
      if (resolved) { clearInterval(fallback); return; }
      try {
        const resp = await fetch(`/api/train/status?card_id=${card.id}`);
        const data = await resp.json();
        // Update UI from poll too
        if (data.step > 0) handleStepData(data);
        if (data.last_error) eng.lastError = data.last_error;
        if (data.graph_warnings) eng.graphWarnings = data.graph_warnings;
        if (!data.training) {
          clearInterval(fallback);
          if (ws.readyState === WebSocket.OPEN) ws.close();
          eng._trainWs = null;
          if (data.last_error) {
            setStatus(`Training error: ${data.last_error}`, { flash: true });
          }
          finish(data.avg_loss);
          resolve();
        }
      } catch {}
    }, 3000);
  });
}

async function doStopAll() {
  // Stop each training card on the server
  const stopPromises = cards
    .filter(c => c.engine.training)
    .map(c => apiPost('/api/train/stop', { card_id: c.id }).catch(() => {}));
  await Promise.all(stopPromises);
  // Mark all as stopped — the server will send "done" events via WebSocket
  // which will close the connections and resolve the watchTraining promises
  cards.forEach(c => {
    if (c.engine.training) {
      c.engine.training = 'stopped';
    }
  });
  setStatus('Training stopped.');
}

let palettePinned = false;

function togglePaletteTab(tab, forceOpen) {
  const popup = document.getElementById('palette-popup');
  if (forceOpen !== false) {
    popup.classList.add('open');
    palettePinned = true;
  }
  document.querySelectorAll('.palette-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.palette-tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector(`.palette-tab-btn[data-tab="${tab}"]`)?.classList.add('active');
  document.getElementById(`palette-tab-${tab}`)?.classList.add('active');
}

function toggleTestPanel(forceOpen) { togglePaletteTab('test', forceOpen); }
function toggleTrainPanel(forceOpen) { togglePaletteTab('train', forceOpen); }

function renderTrainSession() {
  const container = document.getElementById('train-panel-session');
  if (!container) return;
  if (!trainSession) {
    container.innerHTML = '<div style="padding:16px;color:var(--text-dim);font-size:12px;text-align:center">No active session</div>';
    return;
  }
  const activeLabel = trainSession.active ? 'TRAINING' : 'DONE';
  const slots = trainSession.cards.map((c, i) => {
    const color = SESSION_COLORS[i % SESSION_COLORS.length];
    const eng = c.engine;
    const last = eng.lossHistory.length > 0 ? eng.lossHistory[eng.lossHistory.length - 1] : null;
    const lossStr = last ? last.loss.toFixed(4) : '\u2014';
    const stepStr = last ? `Step ${last.step}` : 'Waiting...';
    const badge = eng.training ? '<span class="card-training" style="font-size:10px">LIVE</span>' : '';
    return `
      <div class="session-slot" data-card-id="${c.id}">
        <span class="session-slot-swatch" style="background:${color}"></span>
        <span class="session-slot-name">${c.name || 'untitled'}</span>
        ${badge}
        <span class="session-slot-loss">${lossStr}</span>
        <span class="session-slot-step">${stepStr}</span>
      </div>`;
  }).join('');
  container.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:6px 12px;border-bottom:1px solid var(--border)">
      <span style="font-weight:600;font-size:13px">Session</span>
      <span class="card-training" style="font-size:11px">${activeLabel}</span>
      <button class="session-close" title="Dismiss" style="margin-left:auto">&times;</button>
    </div>
    <div class="session-slots">${slots}</div>
    <div style="padding:6px 12px 10px">
      <canvas class="session-chart" width="600" height="90"></canvas>
    </div>
  `;
  container.querySelector('.session-close').addEventListener('click', () => {
    trainSession = null;
    renderTrainSession();
    renderCardList();
  });
  setTimeout(() => drawSessionChart(container.querySelector('.session-chart')), 0);
}

async function doGenerateAll() {
  const targets = cards.filter(c => c.starred && c.engine.built);
  if (targets.length === 0) {
    setStatus('No starred+built engines to test');
    return;
  }

  const seed = document.getElementById('gen-seed')?.value || 'First Citizen:\n';
  const btn = document.getElementById('btn-generate');
  const container = document.getElementById('test-panel-outputs');
  container.innerHTML = '';

  for (const card of targets) {
    const col = document.createElement('div');
    col.className = 'test-output-col';
    const eng = card.engine;
    const lossInfo = eng.lossHistory.length > 0
      ? `Loss: ${eng.lossHistory[eng.lossHistory.length - 1].loss.toFixed(4)}` : '';
    col.innerHTML = `
      <div class="test-output-header">
        <span>${card.name || 'untitled'}</span>
        <span class="test-loss">${lossInfo}</span>
      </div>
      <pre class="test-output-text generating" data-card-id="${card.id}">Generating...</pre>
    `;
    container.appendChild(col);
  }

  btn.disabled = true;
  btn.textContent = 'Generating...';

  await Promise.allSettled(targets.map(async (card) => {
    const pre = container.querySelector(`pre[data-card-id="${card.id}"]`);
    try {
      const data = await apiPost('/api/generate', { seed, max_tokens: 200, temperature: 0.8, card_id: card.id });
      pre.textContent = data.text || data.error || 'No output';
      pre.classList.remove('generating');
    } catch (err) {
      pre.textContent = 'Error: ' + err.message;
      pre.classList.remove('generating');
    }
  }));

  btn.disabled = false;
  btn.textContent = 'Generate All \u2605';
}



function fmtParams(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

// ── Cards System ─────────────────────────────────────────────────────────────

function setupCards() {
  document.getElementById('btn-fit').addEventListener('click', fitToView);
  document.getElementById('btn-delete-selected').addEventListener('click', () => {
    if (state.selected) deleteNode(state.selected);
  });
  document.getElementById('btn-clear').addEventListener('click', () => clearCanvas());
  document.getElementById('btn-train-all').addEventListener('click', doTrainAll);
  document.getElementById('btn-stop-all').addEventListener('click', doStopAll);
  document.getElementById('btn-generate').addEventListener('click', doGenerateAll);
  document.getElementById('runs-close').addEventListener('click', () => {
    document.getElementById('runs-popup').style.display = 'none';
  });
  document.getElementById('import-file').addEventListener('change', importGraph);
  document.getElementById('btn-pick-data').addEventListener('click', showDataFilePicker);
  document.getElementById('btn-save-project').addEventListener('click', doSaveProject);
  document.getElementById('btn-load-project').addEventListener('click', showProjectsPicker);
  document.getElementById('btn-add-card').addEventListener('click', addCard);
  document.getElementById('btn-copy-card').addEventListener('click', copyCard);
  document.getElementById('projects-close').addEventListener('click', () => {
    document.getElementById('projects-popup').style.display = 'none';
  });
}

async function showDataFilePicker() {
  try {
    const data = await fetchJSON('/api/data-files', {});
    const files = data.files || [];
    if (files.length === 0) {
      setStatus('No .txt files found in data/', { flash: true });
      return;
    }

    // Remove existing picker if any
    document.getElementById('data-file-dropdown')?.remove();

    const btn = document.getElementById('btn-pick-data');
    const dropdown = document.createElement('div');
    dropdown.id = 'data-file-dropdown';
    dropdown.className = 'data-file-dropdown';
    files.forEach(f => {
      const item = document.createElement('div');
      item.className = 'data-file-option';
      item.textContent = f;
      item.addEventListener('click', () => {
        document.getElementById('train-data-file').value = f;
        dropdown.remove();
      });
      dropdown.appendChild(item);
    });

    btn.parentElement.style.position = 'relative';
    btn.parentElement.appendChild(dropdown);

    // Close on outside click
    const close = (e) => {
      if (!dropdown.contains(e.target) && e.target !== btn) {
        dropdown.remove();
        document.removeEventListener('click', close);
      }
    };
    setTimeout(() => document.addEventListener('click', close), 0);
  } catch (err) {
    setStatus('Error listing files: ' + err.message, { flash: true });
  }
}

let lastProjectName = '';

async function doSaveProject() {
  // Require all cards to be built (so their graph.json exists on disk)
  const unbuilt = cards.filter(c => !c.engine.modelHash);
  if (unbuilt.length > 0) {
    alert(`${unbuilt.length} engine(s) haven't been built yet. Build all engines before saving a project.`);
    return;
  }
  const name = prompt('Project name:', lastProjectName);
  if (!name || !name.trim()) return;
  lastProjectName = name.trim();
  try {
    const engines = cards.map(c => ({
      hash: c.engine.modelHash,
      card_name: c.name,
      starred: c.starred,
    }));
    const data = await apiPost('/api/project/save', { name: name.trim(), engines });
    if (data.saved) {
      setStatus('Project saved: ' + name.trim());
    } else {
      setStatus(data.error || 'Save failed', { flash: true });
    }
  } catch (err) {
    setStatus('Save error: ' + err.message, { flash: true });
  }
}

async function showProjectsPicker() {
  try {
    const data = await fetchJSON('/api/projects', { projects: [] });
    const projects = data.projects || [];
    const list = document.getElementById('projects-list');

    if (projects.length === 0) {
      list.innerHTML = '<div class="runs-empty">No saved projects</div>';
    } else {
      list.innerHTML = projects.map(p => `
        <div class="project-item" data-name="${p.name}">
          <div>
            <div class="project-name">${p.name}</div>
            <div class="project-meta">${p.engine_count} engine(s) &middot; ${p.saved_at ? new Date(p.saved_at).toLocaleDateString() : ''}</div>
          </div>
          <button class="project-delete" data-name="${p.name}" title="Delete project">&times;</button>
        </div>
      `).join('');
    }

    const popup = document.getElementById('projects-popup');
    popup.style.display = 'block';

    // Bind click to load
    list.querySelectorAll('.project-item').forEach(item => {
      item.addEventListener('click', async (e) => {
        if (e.target.closest('.project-delete')) return;
        const name = item.dataset.name;
        if (!confirm(`Load project "${name}"? Current workspace will be replaced.`)) return;
        popup.style.display = 'none';
        await doLoadProject(name);
      });
    });

    // Bind delete
    list.querySelectorAll('.project-delete').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const name = btn.dataset.name;
        if (!confirm(`Delete project "${name}"?`)) return;
        await apiPost('/api/project', { name }, 'DELETE');
        showProjectsPicker(); // refresh list
      });
    });
  } catch (err) {
    setStatus('Error listing projects: ' + err.message, { flash: true });
  }
}

async function doLoadProject(name) {
  try {
    const data = await apiPost('/api/project/load', { name });
    if (data.error) { setStatus(data.error, { flash: true }); return; }

    // Close any active training WebSockets
    cards.forEach(c => {
      if (c.engine._trainWs) { c.engine._trainWs.close(); c.engine._trainWs = null; }
    });

    // Reset project state
    cards.length = 0;
    projectGraph.nodes.length = 0;
    projectGraph.edges.length = 0;
    projectNextId = 1;

    // Reconstruct cards
    for (const eng of data.engines) {
      const card = createEmptyCard(eng.card_name);
      card.starred = eng.starred;
      card.engine.modelHash = eng.hash;

      if (eng.graph) {
        const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
        engineNode.children = deserializeGraph(eng.graph);
      }

      cards.push(card);
    }

    // If empty project, add a default card
    if (cards.length === 0) {
      cards.push(createEmptyCard('Engine 1'));
    }

    activeCardIdx = 0;
    lastProjectName = name;
    loadCardIntoCanvas(0);
    renderCardList();
    setStatus('Project loaded: ' + name);
  } catch (err) {
    setStatus('Load error: ' + err.message, { flash: true });
  }
}

// ── Auto-save/restore project to localStorage ────────────────────────────────

let _autoSaveTimer = null;
let _autoSaveEnabled = false;
function debouncedAutoSave() {
  if (!_autoSaveEnabled) return;
  if (_autoSaveTimer) clearTimeout(_autoSaveTimer);
  _autoSaveTimer = setTimeout(autoSaveProject, 2000);
}

function autoSaveProject() {
  if (!_autoSaveEnabled) return;
  try {
    const data = {
      cards: cards.map(c => ({
        name: c.name,
        starred: c.starred,
        graphMode: c.graphMode || false,
        modelHash: c.engine.modelHash,
        summary: c.engine.summary,
        totalSteps: c.engine.totalSteps || 0,
        nodeId: c.nodeId,
      })),
      activeCardIdx,
      projectGraph: serializeGraph(projectGraph),
      projectNextId,
    };
    localStorage.setItem('projectState', JSON.stringify(data));
  } catch (e) {
    console.warn('Auto-save failed:', e);
  }
}

function restoreProject() {
  try {
    const raw = localStorage.getItem('projectState');
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (!data.cards || !data.projectGraph) return false;

    // Restore project graph
    projectGraph.nodes.length = 0;
    projectGraph.edges.length = 0;
    const restored = deserializeGraph(data.projectGraph);
    restored.nodes.forEach(n => projectGraph.nodes.push(n));
    restored.edges.forEach(e => projectGraph.edges.push(e));
    projectNextId = data.projectNextId || state.nextId;

    // Restore cards
    cards.length = 0;
    data.cards.forEach(c => {
      cards.push({
        id: crypto.randomUUID(),
        name: c.name || '',
        starred: c.starred || false,
        graphMode: c.graphMode || false,
        nodeId: c.nodeId,
        engine: {
          built: false,
          training: false,
          lossHistory: [],
          modelHash: c.modelHash || null,
          summary: c.summary || null,
          totalSteps: c.totalSteps || 0,
        },
      });
    });

    if (cards.length === 0) return false;
    activeCardIdx = Math.min(data.activeCardIdx || 0, cards.length - 1);

    // Navigate into the active card
    state.nodes = projectGraph.nodes;
    state.edges = projectGraph.edges;
    state.currentScope = 'project';
    loadCardIntoCanvas(activeCardIdx);
    return true;
  } catch (e) {
    console.warn('Restore failed:', e);
    return false;
  }
}

function addCard() {
  if (cards.length >= MAX_CARDS) return;
  const card = createEmptyCard('Engine ' + (cards.length + 1));
  cards.push(card);
  switchToCard(cards.length - 1);
}

function copyCard() {
  if (cards.length >= MAX_CARDS) { setStatus('Max engines reached', { flash: true }); return; }
  const source = getActiveCard();
  if (!source) return;

  // Create new card with copied graph
  const sourceEngine = projectGraph.nodes.find(n => n.id === source.nodeId);
  const newCard = createEmptyCard(source.name + ' (copy)');
  newCard.graphMode = source.graphMode;

  // Deep-copy the graph by serializing and deserializing
  if (sourceEngine?.children) {
    const newEngine = projectGraph.nodes.find(n => n.id === newCard.nodeId);
    const serialized = serializeGraph(sourceEngine.children);
    newEngine.children = deserializeGraph(serialized);
  }

  cards.push(newCard);
  switchToCard(cards.length - 1);
  setStatus(`Copied "${source.name}"`);
}

async function destroySavedWeights(card) {
  const eng = card.engine;
  if (!eng.modelHash) { setStatus('No model hash — nothing to delete'); return; }
  // Show saves for this hash so user knows what they're deleting
  try {
    const data = await fetchJSON(`/api/saves/${eng.modelHash}`, {});
    const saves = data.saves || [];
    if (saves.length === 0) {
      setStatus('No saved weights on disk for this model');
      return;
    }
    const names = saves.map(s => s.name).join(', ');
    if (!confirm(`Permanently delete all saved weights for model ${eng.modelHash.slice(0,8)}?\n\nSaves: ${names}\n\nThis cannot be undone.`)) return;
    // Delete each save
    for (const save of saves) {
      await apiPost('/api/save', { hash: eng.modelHash, name: save.name }, 'DELETE');
    }
    setStatus(`Deleted ${saves.length} save(s) for model ${eng.modelHash.slice(0,8)}`);
  } catch (err) {
    setStatus('Delete error: ' + err.message);
  }
}

function deleteCard(idx) {
  if (cards.length <= 1) return;
  const card = cards[idx];
  // Close WebSocket if training
  if (card.engine._trainWs) {
    card.engine._trainWs.close();
    card.engine._trainWs = null;
  }
  // Free server slot
  apiPost('/api/slot', { card_id: card.id }, 'DELETE').catch(() => {});
  // Remove engine node from project graph
  const nodeIdx = projectGraph.nodes.findIndex(n => n.id === card.nodeId);
  if (nodeIdx >= 0) projectGraph.nodes.splice(nodeIdx, 1);
  projectGraph.edges = projectGraph.edges.filter(e => e.from.nodeId !== card.nodeId && e.to.nodeId !== card.nodeId);
  cards.splice(idx, 1);
  if (activeCardIdx >= cards.length) activeCardIdx = cards.length - 1;
  if (idx === activeCardIdx || activeCardIdx >= cards.length) {
    activeCardIdx = Math.min(activeCardIdx, cards.length - 1);
    loadCardIntoCanvas(activeCardIdx);
  } else if (idx < activeCardIdx) {
    activeCardIdx--;
  }
  renderCardList();
}

function switchToCard(newIdx) {
  if (newIdx === activeCardIdx) return;
  // Save current DOM state to old card
  const nameInput = document.querySelector('.project-card.expanded .card-name-input');
  if (nameInput) getActiveCard().name = nameInput.value.trim();

  activeCardIdx = newIdx;
  loadCardIntoCanvas(newIdx);
  renderCardList();
}

function loadCardIntoCanvas(idx) {
  const card = cards[idx];
  // Navigate into this engine's pipeline
  const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
  if (!engineNode) return;
  if (!engineNode.children) engineNode.children = { nodes: [], edges: [] };

  state.scopeStack = [{
    nodeId: card.nodeId,
    nodeLabel: card.name || 'Engine',
    scopeType: 'pipeline',
  }];
  state.nodes = engineNode.children.nodes;
  state.edges = engineNode.children.edges;
  state.currentScope = 'pipeline';
  state.selected = null;

  buildPalette();
  renderBreadcrumbs();
  updateViewportBorder();
  renderAll();
  fitToView();
  updateStatus();
  renderTree();
}

function renderCardList() {
  const list = document.getElementById('cards-list');
  list.innerHTML = '';

  cards.forEach((card, idx) => {
    const isActive = idx === activeCardIdx;
    const div = document.createElement('div');
    const ghosted = card.engine.training ? ' ghosted' : '';
    div.className = `project-card ${isActive ? 'expanded' : 'collapsed'}${ghosted}`;
    div.dataset.idx = idx;

    if (isActive) {
      div.innerHTML = renderExpandedCard(card);
      list.appendChild(div);
      bindExpandedCardEvents(div, card, idx);
    } else {
      div.innerHTML = renderCollapsedCard(card);
      div.querySelector('.card-star')?.addEventListener('click', (e) => {
        e.stopPropagation();
        card.starred = !card.starred;
        renderCardList();
      });
      div.addEventListener('click', (e) => {
        if (e.target.closest('.card-star')) return;
        switchToCard(idx);
      });
      list.appendChild(div);
    }
  });

  drawAllSparklines();
  autoSaveProject();
}

function renderCollapsedCard(card) {
  const eng = card.engine;
  const name = card.name || 'untitled';
  const lossStr = eng.lossHistory.length > 0
    ? eng.lossHistory[eng.lossHistory.length - 1].loss.toFixed(4) : '';
  let trainBadge = '';
  if (eng.training) {
    const last = eng.lossHistory.length > 0 ? eng.lossHistory[eng.lossHistory.length - 1] : null;
    const stepInfo = last ? ` ${last.step}` : '';
    trainBadge = `<span class="card-training">TRAINING${stepInfo}</span>`;
  }
  const starCls = card.starred ? 'starred' : '';
  return `
    <div class="card-header">
      <span class="card-star ${starCls}" title="Toggle training">${starSvg}</span>
      <span class="card-name">${name}</span>
      ${trainBadge}
      <canvas class="card-sparkline" width="80" height="24" data-card-id="${card.id}"></canvas>
      ${lossStr ? `<span class="card-loss">${lossStr}</span>` : ''}
    </div>
  `;
}

function renderExpandedCard(card) {
  const eng = card.engine;
  const isTraining = eng.training;

  let modelInfoHtml = '';
  if (eng.summary) {
    const s = eng.summary;
    modelInfoHtml = `
      <div style="padding:8px 12px;font-size:11px;border-bottom:1px solid var(--border)">
        <div class="mi-row"><img class="mi-icon" src="svg/params.svg" title="Params"><span class="mi-value">${fmtParams(s.total_params)}</span></div>
        <div class="mi-row"><img class="mi-icon" src="svg/expert.svg" title="Experts"><span class="mi-value">${s.n_experts}</span></div>
        <div class="mi-row"><img class="mi-icon" src="svg/stream.svg" title="Stream"><span class="mi-value">d=${s.stream_dim}</span></div>
        <div class="mi-row"><img class="mi-icon" src="svg/router.svg" title="Router"><span class="mi-value">${s.router}</span></div>
        <div class="mi-row"><img class="mi-icon" src="svg/vocab.svg" title="Vocab"><span class="mi-value">${s.vocab_size}</span></div>
        ${eng.modelHash ? `<div class="mi-row"><img class="mi-icon" src="svg/hash.svg" title="Hash"><span class="mi-value" title="${eng.modelHash}">${eng.modelHash.slice(0, 8)}</span></div>` : ''}
        ${eng.totalSteps ? `<div class="mi-row"><span class="mi-value" style="font-size:10px;color:var(--text-dim)">${eng.totalSteps} steps trained</span></div>` : ''}
      </div>
    `;
  }

  const dis = isTraining ? 'disabled' : '';
  return `
    <div class="card-header">
      <span class="card-star ${card.starred ? 'starred' : ''}" title="Toggle training">${starSvg}</span>
      <input type="text" class="card-name-input" value="${card.name}" placeholder="untitled">
      ${cards.length > 1 && !isTraining ? '<button class="card-close" title="Remove from rail">&times;</button>' : ''}
      ${eng.modelHash && !isTraining ? '<button class="card-trash" title="Delete saved weights from disk">\u{1F5D1}</button>' : ''}
    </div>
    <div class="card-engine">
      <div class="prop-group" style="display:flex;gap:4px">
        <button class="toolbar-btn engine-btn btn-build" style="flex:1;font-size:12px;padding:4px 8px" ${dis}>${eng.built ? 'Rebuild' : 'Build'}</button>
        <button class="toolbar-btn btn-export" style="flex:1;font-size:12px;padding:4px 8px">Export</button>
        <button class="toolbar-btn btn-import" style="flex:1;font-size:12px;padding:4px 8px">Import</button>
      </div>
      <div style="padding:2px 12px">
        <label style="font-size:11px;color:var(--text-dim);cursor:pointer;display:flex;align-items:center;gap:4px">
          <input type="checkbox" class="chk-graph-mode" ${card.graphMode ? 'checked' : ''}> Graph execution
        </label>
      </div>
      ${modelInfoHtml}
      ${eng.built ? `
        <div class="prop-group" style="display:flex;gap:4px">
          <button class="toolbar-btn btn-runs" title="Prior Runs" style="flex:1"><img class="btn-icon" src="svg/prior.svg"></button>
          <button class="toolbar-btn engine-btn btn-open-train" title="Train" style="flex:1"><img class="btn-icon" src="svg/train.svg"></button>
          <button class="toolbar-btn engine-btn btn-manual-test" title="Test Console" style="flex:1"><img class="btn-icon" src="svg/test.svg"></button>
          <button class="toolbar-btn btn-reset-weights" title="Reset Weights" style="flex:1" ${dis}><img class="btn-icon" src="svg/reset.svg"></button>
        </div>
      ` : ''}
    </div>
  `;
}

function bindExpandedCardEvents(div, card, idx) {
  // Name
  const nameInput = div.querySelector('.card-name-input');
  nameInput?.addEventListener('input', () => { card.name = nameInput.value.trim(); });

  // Close (remove from rail only)
  div.querySelector('.card-close')?.addEventListener('click', (e) => {
    e.stopPropagation();
    deleteCard(idx);
  });

  // Trash (delete saved weights from disk)
  div.querySelector('.card-trash')?.addEventListener('click', (e) => {
    e.stopPropagation();
    destroySavedWeights(card);
  });

  // Design buttons
  div.querySelector('.btn-export')?.addEventListener('click', exportGraph);
  div.querySelector('.btn-import')?.addEventListener('click', () => document.getElementById('import-file').click());


  // Star toggle
  div.querySelector('.card-star')?.addEventListener('click', (e) => {
    e.stopPropagation();
    card.starred = !card.starred;
    renderCardList();
  });

  // Graph mode toggle
  div.querySelector('.chk-graph-mode')?.addEventListener('change', (e) => {
    card.graphMode = e.target.checked;
  });

  // Engine buttons
  div.querySelector('.btn-build')?.addEventListener('click', doBuild);
  div.querySelector('.btn-open-train')?.addEventListener('click', () => toggleTrainPanel(true));
  div.querySelector('.btn-manual-test')?.addEventListener('click', () => toggleTestPanel(true));
  div.querySelector('.btn-runs')?.addEventListener('click', showRunsPicker);
  div.querySelector('.btn-reset-weights')?.addEventListener('click', doResetWeights);

}

function drawSparkline(canvasEl, lossHistory) {
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  const w = canvasEl.width, h = canvasEl.height;
  ctx.clearRect(0, 0, w, h);
  if (lossHistory.length < 2) return;
  const losses = lossHistory.map(d => d.loss);
  const min = Math.min(...losses), max = Math.max(...losses);
  const range = max - min || 1;
  ctx.strokeStyle = '#4a90d9';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  losses.forEach((l, i) => {
    const x = (i / (losses.length - 1)) * w;
    const y = h - ((l - min) / range) * (h - 2) - 1;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawAllSparklines() {
  cards.forEach(card => {
    const el = document.querySelector(`canvas.card-sparkline[data-card-id="${card.id}"]`);
    if (el) drawSparkline(el, card.engine.lossHistory);
  });
}

function drawSessionChart(canvasEl) {
  if (!canvasEl || !trainSession) return;
  const ctx = canvasEl.getContext('2d');
  const w = canvasEl.width, h = canvasEl.height;
  ctx.clearRect(0, 0, w, h);

  // Gather all losses to find global min/max
  const allLosses = trainSession.cards.flatMap(c => c.engine.lossHistory.map(d => d.loss));
  if (allLosses.length < 2) return;

  let minL = Math.min(...allLosses), maxL = Math.max(...allLosses);
  if (maxL - minL < 0.01) { minL -= 0.5; maxL += 0.5; }
  const pad = (maxL - minL) * 0.1;
  minL -= pad; maxL += pad;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = h * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw each engine's loss curve
  trainSession.cards.forEach((card, ci) => {
    const data = card.engine.lossHistory;
    if (data.length < 2) return;
    ctx.strokeStyle = SESSION_COLORS[ci % SESSION_COLORS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((d.loss - minL) / (maxL - minL)) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Y-axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(maxL.toFixed(2), 2, 10);
  ctx.fillText(minL.toFixed(2), 2, h - 2);
}


// ── Left Panel Tabs ─────────────────────────────────────────────────────────

function setupLeftPanel() {
  document.querySelectorAll('.left-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.left-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.left-tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`left-tab-${btn.dataset.tab}`)?.classList.add('active');
    });
  });
}

// ── Chat ─────────────────────────────────────────────────────────────────────

let chatSessionId = localStorage.getItem('chatSessionId') || crypto.randomUUID();
localStorage.setItem('chatSessionId', chatSessionId);
let chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');

function setupChat() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-chat-send');

  sendBtn.addEventListener('click', () => sendChatMessage());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  document.getElementById('btn-chat-clear').addEventListener('click', clearChat);

  // Restore chat history from localStorage
  for (const msg of chatHistory) {
    appendChatMessage(msg.role, msg.text, false);
  }
}

async function clearChat() {
  // Clear server-side session
  await apiPost('/api/chat', {}, 'DELETE').catch(() => {});
  // Clear frontend
  document.getElementById('chat-messages').innerHTML = '';
  chatHistory = [];
  localStorage.removeItem('chatHistory');
  // New session ID
  chatSessionId = crypto.randomUUID();
  localStorage.setItem('chatSessionId', chatSessionId);
}

function appendChatMessage(role, text, save = true) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  if (role === 'assistant') {
    div.innerHTML = text;
  } else {
    div.textContent = text;
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  if (save && role !== 'thinking') {
    chatHistory.push({ role, text });
    // Keep last 50 messages to avoid bloating localStorage
    if (chatHistory.length > 50) chatHistory = chatHistory.slice(-50);
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
  }
  return div;
}

function applyGraphUpdate(graphData, targetCardId) {
  // Find the target card — use card_id if provided, otherwise active card
  let card = null;
  let cardIdx = activeCardIdx;
  if (targetCardId) {
    const idx = cards.findIndex(c => c.id === targetCardId);
    if (idx >= 0) { card = cards[idx]; cardIdx = idx; }
  }
  if (!card) { card = getActiveCard(); cardIdx = activeCardIdx; }
  if (!card) return;

  const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
  if (!engineNode) return;

  try {
    engineNode.children = deserializeGraph(graphData);
    card.engine.built = false; // needs rebuild after graph change
    // Reload the canvas if this is the active card
    if (cardIdx === activeCardIdx) {
      loadCardIntoCanvas(activeCardIdx);
    }
    renderCardList();
    setStatus(`Graph updated by AI: ${card.name || 'untitled'}`);
  } catch (e) {
    console.error('Failed to apply graph update:', e);
    setStatus('Failed to apply AI graph update', { flash: true });
  }
}

async function syncTrainingState() {
  for (const card of cards) {
    if (card.engine.training) continue; // already tracking
    try {
      const resp = await fetch(`/api/train/status?card_id=${card.id}`);
      const data = await resp.json();
      if (data.training && !card.engine.training) {
        card.engine.training = true;
        card.engine.lossHistory = [];
        renderCardList();
        // Start watching this card's training
        watchTraining(card).then(() => renderCardList());
      }
    } catch {}
  }
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;

  input.value = '';
  appendChatMessage('user', message);

  const thinkingEl = appendChatMessage('assistant', '...', false);
  thinkingEl.classList.add('thinking');

  const card = getActiveCard();
  // Build viewscreen context (sent every time, but not stored in history)
  let viewscreen = null;
  if (card) {
    const engineNode = projectGraph.nodes.find(n => n.id === card.nodeId);
    // Active card details
    const activeCard = {
      card_name: card.name,
      starred: card.starred,
      built: card.engine.built,
      graph_mode: card.graphMode || false,
      model_hash: card.engine.modelHash,
      summary: card.engine.summary || null,
      graph: engineNode?.children ? serializeGraph(engineNode.children) : null,
      graph_warnings: card.engine.graphWarnings || [],
      last_error: card.engine.lastError || null,
    };
    // Summary of all cards
    const allCards = cards.map(c => ({
      card_name: c.name,
      active: c === card,
      starred: c.starred,
      built: c.engine.built,
      graph_mode: c.graphMode || false,
      model_hash: c.engine.modelHash,
      summary: c.engine.summary || null,
    }));
    viewscreen = { active_card: activeCard, all_cards: allCards };
  }
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000); // 2 min timeout
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: chatSessionId, card_id: card?.id, viewscreen }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    const data = await resp.json();

    thinkingEl.remove();

    if (data.error) {
      appendChatMessage('assistant', 'Error: ' + data.error);
    } else {
      // Show tools used
      if (data.tools_used && data.tools_used.length > 0) {
        const toolsEl = document.createElement('div');
        toolsEl.className = 'chat-tools-used';
        toolsEl.textContent = data.tools_used.join(' → ');
        document.getElementById('chat-messages').appendChild(toolsEl);
      }
      appendChatMessage('assistant', data.reply || '(no response)');
      // Apply graph update if the AI modified the model
      if (data.graph_update) {
        applyGraphUpdate(data.graph_update, data.graph_update_card_id);
      }
      // Check if AI started training — sync frontend state
      await syncTrainingState();
    }
  } catch (err) {
    thinkingEl.remove();
    const errMsg = err.name === 'AbortError' ? 'Request timed out. The AI may still be processing — try a simpler request.' : err.message;
    appendChatMessage('assistant', 'Error: ' + errMsg);
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────────

init();
