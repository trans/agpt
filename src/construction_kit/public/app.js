// microGPT Construction Kit — Graph Editor with Hierarchical Zoom
// Vanilla JS + SVG, D3 zoom for pan/zoom

'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const starSvg = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>';

// ── Scope / Level Definitions ────────────────────────────────────────────────
// Each scope level defines what components are available in the palette
// and what the "container" types are (double-click to zoom in).

const SCOPE_LEVELS = {
  pipeline: {
    label: "Pipeline",
    components: ["source", "char_tokenizer", "bpe_tokenizer", "sequential_window", "sliding_window", "random_window", "random_init", "zero_init", "learned_init", "cooperative", "loss"],
    containers: { cooperative: "ensemble" },  // double-click cooperative → enter ensemble scope
  },
  ensemble: {
    label: "Ensemble",
    components: ["transformer", "counter", "bigram", "global_router", "context_router", "gated_router", "stream_projector"],
    containers: { transformer: "transformer_internal", counter: "expert_internal" },
  },
  transformer_internal: {
    label: "Transformer",
    components: ["embedding", "attention_layer", "ffn_layer", "layer_norm", "output_head", "stream_proj_internal"],
    containers: {},
  },
  expert_internal: {
    label: "Expert",
    components: ["embedding", "ffn_layer", "layer_norm", "output_head", "stream_proj_internal"],
    containers: {},
  },
};

// ── State ────────────────────────────────────────────────────────────────────

const MAX_CARDS = 8;
const cards = [];       // Array of card objects (each owns a graph + engine state)
let activeCardIdx = 0;

function createEmptyCard(name) {
  return {
    id: crypto.randomUUID(),
    name: name || '',
    starred: true,
    rootGraph: { nodes: [], edges: [] },
    nextId: 1,
    scopeStack: [],
    currentScope: 'pipeline',
    optimizer: { type: 'adam', learning_rate: 0.0003, beta1: 0.9, beta2: 0.999 },
    engine: {
      built: false,
      training: false,
      lossHistory: [],
      pollTimer: null,
      modelHash: null,
      serverLoaded: false,  // true when this card's model is on the server
    },
  };
}

function getActiveCard() { return cards[activeCardIdx]; }

const state = {
  registry: null,       // component definitions from components.json
  // Current view references (derived from active card)
  nodes: [],            // current scope's nodes
  edges: [],            // current scope's edges
  selected: null,       // node id
  // Interaction
  dragging: null,
  connecting: null,
  // View transform (kept in sync with D3)
  viewX: 0, viewY: 0, zoom: 1,
};

// Compatibility getters — redirect rootGraph, nextId, scopeStack, currentScope to active card
Object.defineProperty(state, 'rootGraph', {
  get() { return getActiveCard().rootGraph; },
  set(v) { getActiveCard().rootGraph = v; },
});
Object.defineProperty(state, 'nextId', {
  get() { return getActiveCard().nextId; },
  set(v) { getActiveCard().nextId = v; },
});
Object.defineProperty(state, 'scopeStack', {
  get() { return getActiveCard().scopeStack; },
  set(v) { getActiveCard().scopeStack = v; },
});
Object.defineProperty(state, 'currentScope', {
  get() { return getActiveCard().currentScope; },
  set(v) { getActiveCard().currentScope = v; },
});

// ── DOM refs ─────────────────────────────────────────────────────────────────

const canvas     = document.getElementById('canvas');
const nodesLayer = document.getElementById('nodes-layer');
const edgesLayer = document.getElementById('edges-layer');
const draftEdge  = document.getElementById('draft-edge');
const paletteList = document.getElementById('palette-list');
// props-content removed — properties now rendered in topbar
const statusEl   = document.getElementById('status');

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  const resp = await fetch('components.json');
  state.registry = await resp.json();

  // Create initial card
  const card = createEmptyCard('Demo');
  cards.push(card);
  activeCardIdx = 0;
  state.nodes = state.rootGraph.nodes;
  state.edges = state.rootGraph.edges;

  buildPalette();
  setupD3Zoom();
  setupCanvasEvents();
  setupCards();
  loadDemo();
  renderBreadcrumbs();
  updateStatus();
  renderTree();
  renderCardList();
}

// ── Palette (scope-aware) ────────────────────────────────────────────────────

function buildPalette() {
  const level = SCOPE_LEVELS[state.currentScope];
  const allowedTypes = level ? level.components : state.registry.components.map(c => c.type);

  const byCategory = {};
  for (const comp of state.registry.components) {
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

  renderTreeLevel(container, state.rootGraph.nodes, 'pipeline', 0, []);
}

function renderTreeLevel(parentEl, nodes, scopeType, depth, scopePath) {
  const level = SCOPE_LEVELS[scopeType];

  for (const node of nodes) {
    const comp = getCompDef(node.type);
    if (!comp) continue;

    const childScope = level?.containers[node.type];
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
    label.textContent = comp.label;
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
    state.currentScope = 'pipeline';
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
      return {
        nodeId: sp.nodeId,
        nodeLabel: (comp?.label || '?') + ' #' + sp.nodeId,
        scopeType: sp.scopeType,
      };
    });

    let graph = state.rootGraph;
    let scopeType = 'pipeline';
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

  let html = `<span class="crumb clickable" data-depth="-1">Pipeline</span>`;
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

function updateViewportBorder() {
  const wrap = document.getElementById('canvas-wrap');
  if (state.scopeStack.length === 0) {
    wrap.style.border = 'none';
    removeBoundaryPorts();
    return;
  }
  const parentNode = getParentNode();
  if (!parentNode) return;
  const comp = getCompDef(parentNode.type);
  if (!comp) return;
  wrap.style.border = `4px solid ${comp.color}`;
  wrap.style.borderRadius = '8px';
  renderBoundaryPorts(parentNode, comp);
}

function removeBoundaryPorts() {
  const overlay = document.getElementById('boundary-ports-overlay');
  if (overlay) {
    if (overlay._resizeHandler) window.removeEventListener('resize', overlay._resizeHandler);
    overlay.remove();
  }
}

function renderBoundaryPorts(parentNode, comp) {
  const wrap = document.getElementById('canvas-wrap');
  let overlay = document.getElementById('boundary-ports-overlay');
  if (!overlay) {
    overlay = document.createElementNS(SVG_NS, 'svg');
    overlay.id = 'boundary-ports-overlay';
    overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;z-index:15;pointer-events:none;';
    wrap.appendChild(overlay);
  }
  overlay.innerHTML = '';

  const h = wrap.clientHeight;
  const borderW = 4;
  const portsIn = comp.ports.in || [];
  const portsOut = comp.ports.out || [];

  // Input ports on left edge
  portsIn.forEach((p, i) => {
    const py = distributePort(i, portsIn.length, h);
    renderBoundaryPort(overlay, p, false, borderW / 2, py, comp);
  });

  // Output ports on right edge
  const w = wrap.clientWidth;
  portsOut.forEach((p, i) => {
    const py = distributePort(i, portsOut.length, h);
    renderBoundaryPort(overlay, p, true, w - borderW / 2, py, comp);
  });

  // Re-render on resize
  if (overlay._resizeHandler) window.removeEventListener('resize', overlay._resizeHandler);
  overlay._resizeHandler = () => {
    renderBoundaryPorts(parentNode, comp);
    updateBoundaryEdges();
  };
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

  let shape;
  if (rank >= 2) {
    const size = r * 2;
    shape = document.createElementNS(SVG_NS, 'rect');
    shape.setAttribute('x', px - size / 2);
    shape.setAttribute('y', py - size / 2);
    shape.setAttribute('width', size);
    shape.setAttribute('height', size);
    shape.setAttribute('rx', Math.max(2 * z, 1));
    shape.setAttribute('ry', Math.max(2 * z, 1));
  } else {
    shape = document.createElementNS(SVG_NS, 'circle');
    shape.setAttribute('cx', px);
    shape.setAttribute('cy', py);
    shape.setAttribute('r', r);
  }
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
  const wrap = document.getElementById('canvas-wrap');
  const h = wrap.clientHeight;
  const w = wrap.clientWidth;
  const parentNode = getParentNode();
  if (!parentNode) return { x: 0, y: 0 };
  const comp = getCompDef(parentNode.type);
  if (!comp) return { x: 0, y: 0 };

  // Find which side this port is on by checking both port lists
  const portsIn = comp.ports.in || [];
  const portsOut = comp.ports.out || [];
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
  let scopeType = 'pipeline';
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

  const level = SCOPE_LEVELS[state.currentScope];
  const childScope = level?.containers[node.type];
  if (!childScope) return;

  // Initialize children graph if needed
  if (!node.children) {
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

  state.scopeStack.push({
    nodeId: node.id,
    nodeLabel: getCompDef(node.type)?.label + ' #' + node.id,
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

  // Stream projection in (if stream_dim != d_model)
  if (sd !== dm) {
    const projIn = id();
    nodes.push({ id: projIn, type: "stream_proj_internal", x: 100, y: row * yStep + 60, params: { d_in: sd, d_out: dm } });
    edges.push({ id: id(), from: { nodeId: -1, portId: 'stream_in' }, to: { nodeId: projIn, portId: 'in' } });
    lastId = projIn; lastPort = 'out';
    row++;
  }

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
  return state.registry.components.find(c => c.type === type);
}

function addNode(type, x, y) {
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
  const isContainer = !!(level?.containers[node.type]);

  // Body
  const rect = document.createElementNS(SVG_NS, 'rect');
  rect.classList.add('node-body');
  rect.setAttribute('width', w);
  rect.setAttribute('height', h);
  rect.setAttribute('fill', 'rgba(0,0,0,0.15)');
  rect.setAttribute('stroke', comp.color);
  if (isContainer) {
    rect.setAttribute('stroke-dasharray', '6 3');
  }
  g.appendChild(rect);

  // Header label
  const label = document.createElementNS(SVG_NS, 'text');
  label.classList.add('node-label');
  label.setAttribute('x', w / 2);
  label.setAttribute('y', 22);
  label.setAttribute('text-anchor', 'middle');
  label.setAttribute('font-size', '12');
  label.textContent = comp.label;
  g.appendChild(label);

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
  let shape;
  if (rank >= 2) {
    // Rank-2+: square
    const size = 12;
    shape = document.createElementNS(SVG_NS, 'rect');
    shape.setAttribute('x', px - size / 2);
    shape.setAttribute('y', py - size / 2);
    shape.setAttribute('width', size);
    shape.setAttribute('height', size);
    shape.setAttribute('rx', 2);
    shape.setAttribute('ry', 2);
  } else {
    // Rank-1: circle
    shape = document.createElementNS(SVG_NS, 'circle');
    shape.setAttribute('cx', px);
    shape.setAttribute('cy', py);
    shape.setAttribute('r', 7);
  }
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
    el.addEventListener('change', () => {
      const param = el.dataset.param;
      const ptype = el.dataset.ptype;
      if (ptype === 'string') {
        targetNode.params[param] = el.value;
      } else if (ptype === 'float') {
        targetNode.params[param] = parseFloat(el.value);
      } else {
        targetNode.params[param] = parseInt(el.value);
      }
    });
  });
}

function renderProps() {
  const container = document.getElementById('topbar-props');
  const crumbs = document.getElementById('breadcrumbs');

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
  const isContainer = !!(level?.containers[node.type]);

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
    html += `<span class="prop-crumb inherited" title="from ${item.source}">${item.key}: ${item.value}</span>`;
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

  // Delete button — prepend so it appears leftmost visually
  html = `<span class="topbar-delete" id="btn-delete-node" title="Delete" style="cursor:pointer">\u2715</span><span class="prop-sep">\u25C2</span>` + html;

  container.innerHTML = html;
  bindParamHandlers(container, node);
  document.getElementById('btn-delete-node')?.addEventListener('click', () => deleteNode(node.id));
  document.getElementById('btn-zoom-in')?.addEventListener('click', () => zoomIntoNode(node.id));
}

// Collect inherited values visible at the current scope.
// Returns an array in pipeline order (earliest first = rightmost in row-reverse display).
function getInheritedValues() {
  const items = []; // {key, value, source}
  const seen = new Set();
  const rootNodes = state.rootGraph.nodes;

  // Pipeline-level: source file (earliest)
  for (const n of rootNodes) {
    if (n.type === 'source' && n.params?.file) {
      items.push({ key: 'file', value: n.params.file, source: 'Text Source' });
      seen.add('file');
    }
  }

  // Pipeline-level: seq_len from windower
  const windowTypes = ['sequential_window', 'sliding_window', 'random_window', 'dataset'];
  for (const n of rootNodes) {
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
    html += `<span class="prop-crumb inherited" title="from ${item.source}">${item.key}: ${item.value}</span>`;
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
    // Boundary source — look up port type from parent component
    const parentNode = getParentNode();
    if (parentNode) {
      const comp = getCompDef(parentNode.type);
      const port = comp ? (comp.ports.in || []).find(p => p.id === edge.from.portId) : null;
      if (port) return port.dataType;
    }
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
  state.edges = state.edges.filter(e => e.id !== id);
  removeEdgeEl(id);
  updateStatus();
}

function addEdge(fromNodeId, fromPortId, toNodeId, toPortId) {
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
      if (state.scopeStack.length > 0) updateViewportBorder();
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

  // Mouse leaving the popup closes it
  popup.addEventListener('mouseleave', (e) => {
    if (e.clientY < popup.getBoundingClientRect().top + 10) {
      popup.classList.remove('open');
    }
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
      if (level?.containers[node.type]) {
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

// ── Toolbar (delegated to active card) ───────────────────────────────────────

function doDemo() {
  state.scopeStack = [];
  state.currentScope = 'pipeline';
  state.nodes = state.rootGraph.nodes;
  state.edges = state.rootGraph.edges;
  loadDemo();
  renderBreadcrumbs();
  renderCardList();
}

function exportGraph() {
  const card = getActiveCard();
  saveOptimizerFromDOM();
  const name = card.name || 'untitled';
  const data = {
    version: 3,
    name: name,
    optimizer: card.optimizer,
    graph: serializeGraph(state.rootGraph),
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
      if ((data.version === 2 || data.version === 3) && data.graph) {
        card.rootGraph = deserializeGraph(data.graph);
      } else {
        card.rootGraph = { nodes: [], edges: [] };
        for (const n of data.nodes || []) {
          card.rootGraph.nodes.push({ ...n, label: getCompDef(n.type)?.label || n.type, children: null });
          if (n.id >= card.nextId) card.nextId = n.id + 1;
        }
        for (const ed of data.edges || []) {
          card.rootGraph.edges.push({ id: card.nextId++, from: ed.from, to: ed.to });
        }
      }
      if (data.name) card.name = data.name;
      if (data.optimizer) Object.assign(card.optimizer, data.optimizer);

      card.scopeStack = [];
      card.currentScope = 'pipeline';
      state.nodes = card.rootGraph.nodes;
      state.edges = card.rootGraph.edges;
      card.engine.built = false;
      card.engine.serverLoaded = false;

      buildPalette();
      renderBreadcrumbs();
      renderAll();
      fitToView();
      renderCardList();
      setStatus('Design imported.');
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
  const hint = state.scopeStack.length > 0 ? ' | Esc to go back' : '';
  statusEl.textContent = `${scopeLabel}: ${state.nodes.length} components, ${state.edges.length} connections | Scroll to zoom, drag to pan${hint}`;
}

function setStatus(msg) {
  statusEl.textContent = msg;
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
  const id = () => card.nextId++;

  const srcId = id(), tokId = id(), winId = id(), initId = id(), coopId = id(), lossId = id();

  card.rootGraph = {
    nodes: [
      { id: srcId, type: "source", x: -400, y: 100, params: { file: "data/input.txt" }, children: null },
      { id: tokId, type: "char_tokenizer", x: -200, y: 100, params: {}, children: null },
      { id: winId, type: "sequential_window", x: 0, y: 100, params: { seq_len: 128 }, children: null },
      { id: initId, type: "zero_init", x: 0, y: -60, params: {}, children: null },
      { id: coopId, type: "cooperative", x: 280, y: 60, params: { stream_dim: 64 }, children: null },
      { id: lossId, type: "loss", x: 560, y: 100, params: {}, children: null },
    ],
    edges: [
      // Source → Tokenizer → Windower
      { id: id(), from: { nodeId: srcId, portId: "raw_text" }, to: { nodeId: tokId, portId: "raw_text" } },
      { id: id(), from: { nodeId: tokId, portId: "token_ids" }, to: { nodeId: winId, portId: "token_ids" } },
      // Windower tokens → ensemble + loss targets
      { id: id(), from: { nodeId: winId, portId: "input_ids" }, to: { nodeId: coopId, portId: "token_ids" } },
      { id: id(), from: { nodeId: winId, portId: "target_ids" }, to: { nodeId: lossId, portId: "targets" } },
      // Stream init → ensemble
      { id: id(), from: { nodeId: initId, portId: "stream_out" }, to: { nodeId: coopId, portId: "stream_in" } },
      // Ensemble → loss
      { id: id(), from: { nodeId: coopId, portId: "logits_out" }, to: { nodeId: lossId, portId: "logits_in" } },
    ],
  };

  // Pre-populate children so they're visible when zooming in
  const coopNode = card.rootGraph.nodes.find(n => n.id === coopId);
  if (coopNode) ensureChildren(coopNode);

  card.scopeStack = [];
  card.currentScope = 'pipeline';
  card.engine.built = false;
  card.engine.serverLoaded = false;
  card.engine.lossHistory = [];
  state.nodes = card.rootGraph.nodes;
  state.edges = card.rootGraph.edges;

  buildPalette();
  renderAll();
  fitToView();
  setStatus('Demo: Dataset → Cooperative Ensemble → Loss. Double-click containers to zoom in!');
}

// ── Canonical Model Hash ──────────────────────────────────────────────────────
// Hashes the architectural structure of the graph (types, connections, params
// that affect architecture) but excludes visual layout (x, y positions).
// Same architecture → same hash, regardless of canvas arrangement.

function canonicalGraphObj(graph) {
  // Sort nodes by id for determinism
  const nodes = [...graph.nodes].sort((a, b) => a.id - b.id).map(n => {
    const obj = { id: n.id, type: n.type, params: sortedObj(n.params || {}) };
    if (n.children && n.children.nodes.length > 0) {
      obj.children = canonicalGraphObj(n.children);
    }
    return obj;
  });
  // Sort edges deterministically
  const edges = [...graph.edges].sort((a, b) => {
    if (a.from.nodeId !== b.from.nodeId) return a.from.nodeId - b.from.nodeId;
    if (a.from.portId !== b.from.portId) return a.from.portId.localeCompare(b.from.portId);
    if (a.to.nodeId !== b.to.nodeId) return a.to.nodeId - b.to.nodeId;
    return a.to.portId.localeCompare(b.to.portId);
  }).map(e => ({ from: { nodeId: e.from.nodeId, portId: e.from.portId }, to: { nodeId: e.to.nodeId, portId: e.to.portId } }));
  return { nodes, edges };
}

function sortedObj(obj) {
  const out = {};
  for (const k of Object.keys(obj).sort()) out[k] = obj[k];
  return out;
}

async function computeModelHash(graph) {
  const canonical = JSON.stringify(canonicalGraphObj(graph));
  const data = new TextEncoder().encode(canonical);
  const hashBuf = await crypto.subtle.digest('SHA-256', data);
  const hashArr = Array.from(new Uint8Array(hashBuf));
  return hashArr.slice(0, 4).map(b => b.toString(16).padStart(2, '0')).join('');
}

// ── Engine (now per-card) ─────────────────────────────────────────────────────

// Convenience accessor for active card's engine
function getEngine() { return getActiveCard().engine; }

function getOptimizerConfig() {
  const card = getActiveCard();
  return { ...card.optimizer };
}

function saveOptimizerFromDOM() {
  const card = getActiveCard();
  const el = document.querySelector('.project-card.expanded');
  if (!el) return;
  const type = el.querySelector('.opt-type')?.value || 'adam';
  card.optimizer.type = type;
  card.optimizer.learning_rate = parseFloat(el.querySelector('.opt-lr')?.value || 0.0003);
  if (type === 'adam' || type === 'adamw') {
    card.optimizer.beta1 = parseFloat(el.querySelector('.opt-beta1')?.value || 0.9);
    card.optimizer.beta2 = parseFloat(el.querySelector('.opt-beta2')?.value || 0.999);
  }
  if (type === 'sgd') {
    card.optimizer.momentum = parseFloat(el.querySelector('.opt-momentum')?.value || 0);
  }
  if (type === 'adamw') {
    card.optimizer.weight_decay = parseFloat(el.querySelector('.opt-wd')?.value || 0.01);
  }
}

async function doSaveWeights(name) {
  const eng = getEngine();
  if (!eng.modelHash) return;
  name = name || getActiveCard().name || 'default';
  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hash: eng.modelHash, name }),
    });
    const data = await resp.json();
    if (data.saved) {
      setStatus(`Weights saved: ${eng.modelHash}/${name}`);
    } else {
      setStatus('Save failed: ' + (data.error || 'unknown'));
    }
  } catch (err) {
    setStatus('Save error: ' + err.message);
  }
}

async function fetchRunMetrics(hash, name) {
  try {
    const resp = await fetch(`/api/saves/${hash}/${name}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

async function doLoadWeights(name) {
  const eng = getEngine();
  if (!eng.modelHash) return;
  name = name || 'default';
  try {
    const resp = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hash: eng.modelHash, name }),
    });
    const data = await resp.json();
    if (data.loaded) {
      setStatus(`Weights loaded: ${eng.modelHash}/${name}`);
      return true;
    }
    return false;
  } catch { return false; }
}

async function autoLoadLatest() {
  const eng = getEngine();
  if (!eng.modelHash) return;
  try {
    const resp = await fetch(`/api/saves/${eng.modelHash}`);
    const data = await resp.json();
    if (data.saves && data.saves.length > 0) {
      const latest = data.saves[0];
      const loaded = await doLoadWeights(latest.name);
      if (loaded) setStatus(`Auto-loaded weights: ${latest.name}`);
    }
  } catch {}
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
    const resp = await fetch(`/api/saves/${eng.modelHash}`);
    const data = await resp.json();
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
  const eng = getEngine();
  if (!eng.built) return;
  try {
    const payload = { version: 2, graph: serializeGraph(state.rootGraph) };
    const resp = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.built) {
      eng.lossHistory = [];
      eng.serverLoaded = true;
      setStatus('Weights reset — model reinitialized from scratch');
      renderCardList();
    } else {
      setStatus('Reset failed: ' + (data.details || data.error || 'unknown'));
    }
  } catch (err) {
    setStatus('Reset error: ' + err.message);
  }
}

async function doBuild() {
  const eng = getEngine();
  const card = getActiveCard();
  saveOptimizerFromDOM();

  try {
    const payload = { version: 2, graph: serializeGraph(state.rootGraph) };
    const resp = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();

    if (data.built) {
      eng.built = true;
      eng.lossHistory = [];
      eng.modelHash = await computeModelHash(state.rootGraph);
      eng.serverLoaded = true;
      eng.summary = data.summary;
      setStatus(`Model built: ${fmtParams(data.summary.total_params)} params, ${data.summary.n_experts} experts`);
      renderCardList();
      await autoLoadLatest();
    } else {
      const errors = data.details || data.errors || [data.error || 'Unknown error'];
      alert('Build failed:\n' + errors.join('\n'));
    }
  } catch (err) {
    alert('Build error: ' + err.message);
  }
}

// Ensure active card's model is loaded on the server (rebuild if needed)
async function ensureServerLoaded() {
  const eng = getEngine();
  if (!eng.built) return false;
  if (eng.serverLoaded) return true;
  // Silently rebuild
  try {
    const payload = { version: 2, graph: serializeGraph(state.rootGraph) };
    const resp = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.built) {
      eng.serverLoaded = true;
      eng.summary = data.summary;
      await autoLoadLatest();
      return true;
    }
  } catch {}
  return false;
}

async function doTrainAll() {
  const starred = cards.filter(c => c.starred && c.engine.built);
  if (starred.length === 0) {
    alert('No starred engines with built models to train.');
    return;
  }

  const steps = parseInt(document.getElementById('train-steps')?.value) || 1000;
  const runName = document.getElementById('train-run-name')?.value?.trim() || '';

  // Show stop button
  document.getElementById('btn-train-all').style.display = 'none';
  document.getElementById('btn-stop-all').style.display = '';

  // Train each starred engine sequentially (server holds one model at a time)
  for (const card of starred) {
    if (card.engine.training === 'stopped') break; // user hit stop
    await trainSingleCard(card, steps, runName);
  }

  document.getElementById('btn-train-all').style.display = '';
  document.getElementById('btn-stop-all').style.display = 'none';
  renderCardList();
}

async function trainSingleCard(card, steps, runName) {
  const eng = card.engine;
  const prevIdx = activeCardIdx;

  // Ensure this card's model is on the server
  const cardIdx = cards.indexOf(card);
  if (cardIdx !== activeCardIdx) switchToCard(cardIdx);
  if (!await ensureServerLoaded()) return;
  saveOptimizerFromDOM();

  const name = runName || card.name || '';
  try {
    const resp = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ steps, name }),
    });
    const data = await resp.json();
    if (data.started) {
      eng.training = true;
      eng.lossHistory = [];
      renderCardList();
      setStatus(`Training ${card.name || 'untitled'}...`);
      // Poll until complete
      await pollUntilDone(card);
    } else {
      setStatus(`${card.name || 'untitled'}: ${data.error || 'Failed to start'}`);
    }
  } catch (err) {
    setStatus(`Train error (${card.name || 'untitled'}): ${err.message}`);
  }
}

async function pollUntilDone(card) {
  const eng = card.engine;
  return new Promise((resolve) => {
    const poll = async () => {
      try {
        const resp = await fetch('/api/train/status');
        const data = await resp.json();

        // Update loss history
        if (eng.lossHistory.length === 0 || eng.lossHistory[eng.lossHistory.length - 1].step !== data.step) {
          eng.lossHistory.push({ step: data.step, loss: data.avg_loss });
          drawLossChart();
          drawAllSparklines();
        }

        // Update train status in expanded card if visible
        const tsEl = document.querySelector('.project-card.expanded .train-status');
        if (tsEl && cards[activeCardIdx] === card) {
          const elapsed = data.elapsed_sec || 0;
          const stepsPerSec = elapsed > 0 ? (data.step / elapsed).toFixed(1) : '\u2014';
          const timeStr = elapsed >= 60
            ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
            : `${elapsed.toFixed(1)}s`;
          tsEl.innerHTML = `
            <div class="ts-loss">${data.avg_loss.toFixed(4)}</div>
            <div class="ts-step">Step ${data.step} / ${data.steps} \u00b7 ${timeStr} \u00b7 ${stepsPerSec} steps/s</div>
          `;
          tsEl.style.display = '';
        }

        if (!data.training) {
          eng.training = false;
          setStatus(`${card.name || 'untitled'}: done. Loss: ${data.avg_loss.toFixed(4)}`);
          renderCardList();
          // Auto-save
          if (document.getElementById('btn-save-toggle')?.classList.contains('active')) {
            const rn = document.getElementById('train-run-name')?.value?.trim() || card.name || 'default';
            await doSaveWeights(rn);
          }
          resolve();
        } else {
          setTimeout(poll, 4000);
        }
      } catch (e) {
        console.error('Poll error:', e);
        setTimeout(poll, 4000);
      }
    };
    poll();
  });
}

async function doStopAll() {
  try { await fetch('/api/train/stop', { method: 'POST' }); } catch {}
  // Mark all as stopped
  cards.forEach(c => {
    if (c.engine.training) {
      c.engine.training = 'stopped';
    }
  });
  setStatus('Training stopped.');
  // The pollUntilDone will see training=false from server and resolve
}

function drawLossChart() {
  const canvasEl = document.querySelector('.project-card.expanded .loss-chart');
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  const w = canvasEl.width;
  const h = canvasEl.height;
  const data = getEngine().lossHistory;
  if (data.length < 2) return;

  ctx.clearRect(0, 0, w, h);
  const losses = data.map(d => d.loss);
  let minL = Math.min(...losses);
  let maxL = Math.max(...losses);
  if (maxL - minL < 0.01) { minL -= 0.5; maxL += 0.5; }
  const pad = (maxL - minL) * 0.1;
  minL -= pad; maxL += pad;

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = h * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  ctx.strokeStyle = '#4a90d9';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((d.loss - minL) / (maxL - minL)) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(maxL.toFixed(2), 2, 10);
  ctx.fillText(minL.toFixed(2), 2, h - 2);
}

function openManualTest() {
  const eng = getEngine();
  if (!eng.built) return;
  const overlay = document.getElementById('gen-overlay');
  document.getElementById('gen-overlay-text').textContent = '';
  const lossEl = document.getElementById('gen-overlay-loss');
  if (eng.lossHistory.length > 0) {
    const last = eng.lossHistory[eng.lossHistory.length - 1];
    lossEl.textContent = `Loss: ${last.loss.toFixed(4)} @ step ${last.step}`;
  } else {
    lossEl.textContent = '';
  }
  overlay.style.display = '';
}

async function doGenerate() {
  const eng = getEngine();
  if (!eng.built) return;
  if (!await ensureServerLoaded()) return;

  const seed = document.getElementById('gen-seed')?.value || 'First Citizen:\n';
  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed, max_tokens: 200, temperature: 0.8 }),
    });
    const data = await resp.json();
    document.getElementById('gen-overlay-text').textContent = data.text || data.error || 'No output';
  } catch (err) {
    document.getElementById('gen-overlay-text').textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}



function fmtParams(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

// ── Cards System ─────────────────────────────────────────────────────────────

function setupCards() {
  document.getElementById('btn-add-card').addEventListener('click', addCard);
  document.getElementById('btn-fit').addEventListener('click', fitToView);
  document.getElementById('btn-clear').addEventListener('click', () => clearCanvas());
  document.getElementById('btn-train-all').addEventListener('click', doTrainAll);
  document.getElementById('btn-stop-all').addEventListener('click', doStopAll);
  document.getElementById('btn-save-toggle').addEventListener('click', (e) => {
    e.currentTarget.classList.toggle('active');
  });
  document.getElementById('btn-generate').addEventListener('click', doGenerate);
  const genOverlay = document.getElementById('gen-overlay');
  document.getElementById('gen-overlay-close').addEventListener('click', () => { genOverlay.style.display = 'none'; });
  genOverlay.addEventListener('click', (e) => { if (e.target === genOverlay) genOverlay.style.display = 'none'; });
  document.getElementById('runs-close').addEventListener('click', () => {
    document.getElementById('runs-popup').style.display = 'none';
  });
  document.getElementById('import-file').addEventListener('change', importGraph);
}

function addCard() {
  if (cards.length >= MAX_CARDS) return;
  const card = createEmptyCard('Engine ' + (cards.length + 1));
  cards.push(card);
  switchToCard(cards.length - 1);
}

async function destroySavedWeights(card) {
  const eng = card.engine;
  if (!eng.modelHash) { setStatus('No model hash — nothing to delete'); return; }
  // Show saves for this hash so user knows what they're deleting
  try {
    const resp = await fetch(`/api/saves/${eng.modelHash}`);
    const data = await resp.json();
    const saves = data.saves || [];
    if (saves.length === 0) {
      setStatus('No saved weights on disk for this model');
      return;
    }
    const names = saves.map(s => s.name).join(', ');
    if (!confirm(`Permanently delete all saved weights for model ${eng.modelHash}?\n\nSaves: ${names}\n\nThis cannot be undone.`)) return;
    // Delete each save
    for (const save of saves) {
      await fetch('/api/save', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hash: eng.modelHash, name: save.name }),
      });
    }
    setStatus(`Deleted ${saves.length} save(s) for model ${eng.modelHash}`);
  } catch (err) {
    setStatus('Delete error: ' + err.message);
  }
}

function deleteCard(idx) {
  if (cards.length <= 1) return;
  const card = cards[idx];
  // Stop polling if training
  if (card.engine.pollTimer) {
    clearInterval(card.engine.pollTimer);
    card.engine.pollTimer = null;
  }
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
  saveOptimizerFromDOM();
  const nameInput = document.querySelector('.project-card.expanded .card-name-input');
  if (nameInput) getActiveCard().name = nameInput.value.trim();

  // Mark old card's model as not on the server anymore (server holds one model)
  cards.forEach(c => c.engine.serverLoaded = false);

  activeCardIdx = newIdx;
  loadCardIntoCanvas(newIdx);
  renderCardList();
}

function loadCardIntoCanvas(idx) {
  const card = cards[idx];
  state.nodes = card.rootGraph.nodes;
  state.edges = card.rootGraph.edges;
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
  const addBtn = document.getElementById('btn-add-card');
  addBtn.disabled = cards.length >= MAX_CARDS;

  cards.forEach((card, idx) => {
    const isActive = idx === activeCardIdx;
    const div = document.createElement('div');
    div.className = `project-card ${isActive ? 'expanded' : 'collapsed'}`;
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

  // Draw sparklines after DOM is in place
  drawAllSparklines();
}

function renderCollapsedCard(card) {
  const eng = card.engine;
  const name = card.name || 'untitled';
  const lossStr = eng.lossHistory.length > 0
    ? eng.lossHistory[eng.lossHistory.length - 1].loss.toFixed(4) : '';
  const trainBadge = eng.training ? '<span class="card-training">TRAINING</span>' : '';
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
  const opt = card.optimizer;
  const isTraining = eng.training;

  let modelInfoHtml = '';
  if (eng.summary) {
    const s = eng.summary;
    modelInfoHtml = `
      <div style="padding:8px 12px;font-size:11px;border-bottom:1px solid var(--border)">
        <div class="mi-row"><span class="mi-label">Params</span><span class="mi-value">${fmtParams(s.total_params)}</span></div>
        <div class="mi-row"><span class="mi-label">Experts</span><span class="mi-value">${s.n_experts}</span></div>
        <div class="mi-row"><span class="mi-label">Stream</span><span class="mi-value">d=${s.stream_dim}</span></div>
        <div class="mi-row"><span class="mi-label">Router</span><span class="mi-value">${s.router}</span></div>
        <div class="mi-row"><span class="mi-label">Vocab</span><span class="mi-value">${s.vocab_size}</span></div>
        ${eng.modelHash ? `<div class="mi-row"><span class="mi-label">Hash</span><span class="mi-value">${eng.modelHash}</span></div>` : ''}
      </div>
    `;
  }

  return `
    <div class="card-header">
      <span class="card-star ${card.starred ? 'starred' : ''}" title="Toggle training">${starSvg}</span>
      <input type="text" class="card-name-input" value="${card.name}" placeholder="untitled">
      ${cards.length > 1 ? '<button class="card-close" title="Remove from rail">&times;</button>' : ''}
      ${eng.modelHash ? '<button class="card-trash" title="Delete saved weights from disk">\u{1F5D1}</button>' : ''}
    </div>
    <div class="card-engine">
      <div class="prop-group" style="display:flex;gap:4px;flex-wrap:wrap">
        <button class="toolbar-btn btn-demo" style="flex:1;font-size:12px;padding:4px 8px">Demo</button>
        <button class="toolbar-btn btn-import" style="flex:1;font-size:12px;padding:4px 8px">Import</button>
        <button class="toolbar-btn btn-export" style="flex:1;font-size:12px;padding:4px 8px">Export</button>

      </div>
      <div class="prop-group">
        <label>optimizer</label>
        <select class="opt-type">
          <option value="adam" ${opt.type === 'adam' ? 'selected' : ''}>Adam</option>
          <option value="sgd" ${opt.type === 'sgd' ? 'selected' : ''}>SGD</option>
          <option value="adamw" ${opt.type === 'adamw' ? 'selected' : ''}>AdamW</option>
        </select>
      </div>
      <div class="prop-group">
        <label>learning rate</label>
        <input type="number" class="opt-lr" value="${opt.learning_rate}" min="0.000001" max="0.1" step="0.0001">
        <div class="opt-adam-params" style="${opt.type === 'adam' ? '' : 'display:none'}">
          <label>beta1</label>
          <input type="number" class="opt-beta1" value="${opt.beta1 || 0.9}" min="0" max="0.999" step="0.01">
          <label>beta2</label>
          <input type="number" class="opt-beta2" value="${opt.beta2 || 0.999}" min="0" max="0.9999" step="0.001">
        </div>
        <div class="opt-sgd-params" style="${opt.type === 'sgd' ? '' : 'display:none'}">
          <label>momentum</label>
          <input type="number" class="opt-momentum" value="${opt.momentum || 0}" min="0" max="0.99" step="0.01">
        </div>
        <div class="opt-adamw-params" style="${opt.type === 'adamw' ? '' : 'display:none'}">
          <label>weight decay</label>
          <input type="number" class="opt-wd" value="${opt.weight_decay || 0.01}" min="0" max="1.0" step="0.001">
        </div>
      </div>
      <div class="prop-group">
        <button class="toolbar-btn engine-btn btn-build">${eng.built ? 'Rebuild Model' : 'Build Model'}</button>
      </div>
      ${modelInfoHtml}
      ${eng.built ? `
        <div style="padding:0 12px 6px;text-align:right">
          <button class="toolbar-btn btn-reset-weights" style="font-size:11px;padding:2px 8px">Reset Weights</button>
        </div>
        <div class="train-status" style="${isTraining ? '' : 'display:none'}; padding:8px 12px; font-size:12px; border-bottom:1px solid var(--border)"></div>
        <div class="loss-chart-wrap" style="${eng.lossHistory.length > 0 ? '' : 'display:none'}; padding:6px 12px; border-bottom:1px solid var(--border)">
          <canvas class="loss-chart" width="220" height="80"></canvas>
        </div>
        <div class="prop-group">
          <button class="toolbar-btn engine-btn btn-manual-test" style="width:100%">Manual Test</button>
          <button class="toolbar-btn btn-runs" style="width:100%;margin-top:4px;font-size:11px">Prior Runs</button>
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
  div.querySelector('.btn-demo')?.addEventListener('click', doDemo);
  div.querySelector('.btn-import')?.addEventListener('click', () => document.getElementById('import-file').click());
  div.querySelector('.btn-export')?.addEventListener('click', exportGraph);


  // Optimizer type switcher
  const optType = div.querySelector('.opt-type');
  optType?.addEventListener('change', () => {
    const v = optType.value;
    const adam = div.querySelector('.opt-adam-params');
    const sgd = div.querySelector('.opt-sgd-params');
    const adamw = div.querySelector('.opt-adamw-params');
    if (adam) adam.style.display = v === 'adam' ? '' : 'none';
    if (sgd) sgd.style.display = v === 'sgd' ? '' : 'none';
    if (adamw) adamw.style.display = v === 'adamw' ? '' : 'none';
  });

  // Star toggle
  div.querySelector('.card-star')?.addEventListener('click', (e) => {
    e.stopPropagation();
    card.starred = !card.starred;
    renderCardList();
  });

  // Engine buttons
  div.querySelector('.btn-build')?.addEventListener('click', doBuild);
  div.querySelector('.btn-manual-test')?.addEventListener('click', openManualTest);
  div.querySelector('.btn-runs')?.addEventListener('click', showRunsPicker);
  div.querySelector('.btn-reset-weights')?.addEventListener('click', doResetWeights);

  // Draw loss chart if data exists
  if (card.engine.lossHistory.length >= 2) {
    setTimeout(drawLossChart, 0);
  }
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

// ── Boot ─────────────────────────────────────────────────────────────────────

init();
