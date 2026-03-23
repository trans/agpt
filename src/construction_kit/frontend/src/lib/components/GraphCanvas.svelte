<script>
  import { onMount } from 'svelte';
  import { nodes, edges, groups, findNode, addEdge, removeNode, childGroupPaths, nodesUnderGroup, groupBoundaryPorts } from '../stores/graph.js';
  import { portRank, portShapeAttrs } from '../utils/portShapes.js';
  import { currentGroup, selectedNode, registry, viewTransform, getCompDef, getDataTypeColor } from '../stores/ui.js';
  import Node from './Node.svelte';
  import Edge from './Edge.svelte';
  import GroupBox from './GroupBox.svelte';
  import Breadcrumbs from './Breadcrumbs.svelte';

  let svgEl;
  let transform = { x: 0, y: 0, k: 1 };
  let panning = false;
  let panStart = { x: 0, y: 0 };

  // Wall port layout constants
  const WALL_PORT_START_Y = -50;
  const WALL_PORT_SPACING = 40;

  // Interaction state
  let dragging = null;     // { nodeId, startX, startY } or { groupPath, startX, startY }
  let connecting = null;   // { fromNodeId, fromPortId }
  let mouseWorld = { x: 0, y: 0 };

  // ── Derived data ────────────────────────────────────────────────────────

  $: visibleNodes = $nodes.filter(n => n.group === $currentGroup);
  $: childGPs = childGroupPaths($currentGroup, $groups);
  $: groupBoxes = buildGroupBoxes(childGPs, $nodes, $edges, $groups);

  function buildGroupBoxes(paths, nodesVal, edgesVal, groupsVal) {
    const boxes = [];
    let autoX = 0, autoY = -100;
    const boxW = 160, boxH = 70, gap = 30;
    for (const gp of paths) {
      const gNodes = nodesUnderGroup(gp, nodesVal);
      if (gNodes.length === 0) continue;
      const info = groupsVal[gp];
      const x = info?._x ?? autoX;
      const y = info?._y ?? autoY;
      const { inputs, outputs } = groupBoundaryPorts(gp, nodesVal, edgesVal);
      // Count deduplicated ports (same logic as GroupBox)
      const nIn = new Set(inputs.map(e => `${e.from.nodeId}:${e.from.portId}`)).size;
      const nOut = new Set(outputs.map(e => `${e.to.nodeId}:${e.to.portId}`)).size;
      const autoH = Math.max(boxH, 24 + Math.max(nIn, nOut) * 16);
      boxes.push({ path: gp, info, x, y, w: boxW, h: autoH, inputs, outputs });
      autoX += boxW + gap;
      if (autoX > 600) { autoX = 0; autoY -= boxH + gap; }
    }
    return boxes;
  }

  // ── Port positions ──────────────────────────────────────────────────────

  function getPortPos(nodeId, portId, isOutput) {
    const node = findNode(nodeId);
    if (!node) return null;
    const comp = getCompDef(node.type, $registry);
    if (!comp) return null;
    const w = 140;
    const ports = isOutput ? (comp.ports?.out || []) : (comp.ports?.in || []);
    const idx = ports.findIndex(p => p.id === portId);
    if (idx < 0) return null;
    return { x: node.x + (isOutput ? w : 0), y: node.y + 30 + idx * 14 };
  }

  // ── Wall ports (edges crossing current group boundary) ──────────────────

  $: wallPorts = computeWallPorts($edges, $nodes, $currentGroup);

  function computeWallPorts(edgesVal, allNodes, curGroup) {
    if (!curGroup) return { inputs: [], outputs: [] };
    const insideIds = new Set(nodesUnderGroup(curGroup, allNodes).map(n => n.id));
    const inputs = [];   // edges from outside → inside
    const outputs = [];  // edges from inside → outside
    const seenIn = new Set();
    const seenOut = new Set();

    for (const e of edgesVal) {
      const fromIn = insideIds.has(e.from.nodeId);
      const toIn = insideIds.has(e.to.nodeId);
      if (!fromIn && toIn) {
        // Deduplicate by source node
        const key = `${e.from.nodeId}:${e.from.portId}`;
        if (!seenIn.has(key)) {
          seenIn.add(key);
          inputs.push({ key, edges: [] });
        }
        inputs.find(p => p.key === key).edges.push(e);
      } else if (fromIn && !toIn) {
        const key = `${e.to.nodeId}:${e.to.portId}`;
        if (!seenOut.has(key)) {
          seenOut.add(key);
          outputs.push({ key, edges: [] });
        }
        outputs.find(p => p.key === key).edges.push(e);
      }
    }
    return { inputs, outputs };
  }

  // Wall port positions — at the viewport edges in world coordinates
  $: WALL_X_IN = (-transform.x / transform.k) - 10;
  $: WALL_X_OUT = ((-transform.x + (svgEl?.getBoundingClientRect()?.width || 760)) / transform.k) + 10;

  // ── Edge positions ──────────────────────────────────────────────────────

  $: edgePositions = computeEdgePositions($edges, visibleNodes, groupBoxes, $nodes, wallPorts, $currentGroup, WALL_X_IN, WALL_X_OUT);

  function computeEdgePositions(edgesVal, visible, gBoxes, allNodes, walls, curGroup, wallXIn, wallXOut) {
    const visibleIds = new Set(visible.map(n => n.id));
    const results = [];

    // Build lookup: nodeId → which group box contains it (if collapsed)
    const nodeToGroupBox = new Map();
    for (const box of gBoxes) {
      const gNodes = nodesUnderGroup(box.path, allNodes);
      for (const n of gNodes) nodeToGroupBox.set(n.id, box);
    }

    // Track port positions per group box to stack them vertically
    const boxInputCounts = new Map();
    const boxOutputCounts = new Map();

    // Track which edges are handled by wall ports (to avoid duplicates)
    const wallEdgeIds = new Set();

    // Wall port edges: from wall → internal nodes (inside view)

    if (curGroup && walls) {
      walls.inputs.forEach((wp, wpIdx) => {
        const wallY = WALL_PORT_START_Y + wpIdx * WALL_PORT_SPACING;
        wp.edges.forEach(e => {
          wallEdgeIds.add(e.id);
          const toVisible = visibleIds.has(e.to.nodeId);
          const toBox = nodeToGroupBox.get(e.to.nodeId);
          let toPos = null;
          if (toVisible) {
            toPos = getPortPos(e.to.nodeId, e.to.portId, false);
          } else if (toBox) {
            const idx = boxInputCounts.get(toBox.path) || 0;
            boxInputCounts.set(toBox.path, idx + 1);
            toPos = { x: toBox.x, y: toBox.y + 30 + idx * 16 };
          }
          if (toPos) {
            results.push({ edge: e, from: { x: wallXIn, y: wallY }, to: toPos });
          }
        });
      });

      walls.outputs.forEach((wp, wpIdx) => {
        const wallY = WALL_PORT_START_Y + wpIdx * WALL_PORT_SPACING;
        wp.edges.forEach(e => {
          wallEdgeIds.add(e.id);
          const fromVisible = visibleIds.has(e.from.nodeId);
          const fromBox = nodeToGroupBox.get(e.from.nodeId);
          let fromPos = null;
          if (fromVisible) {
            fromPos = getPortPos(e.from.nodeId, e.from.portId, true);
          } else if (fromBox) {
            const idx = boxOutputCounts.get(fromBox.path) || 0;
            boxOutputCounts.set(fromBox.path, idx + 1);
            fromPos = { x: fromBox.x + fromBox.w, y: fromBox.y + 30 + idx * 16 };
          }
          if (fromPos) {
            results.push({ edge: e, from: fromPos, to: { x: wallXOut, y: wallY } });
          }
        });
      });
    }

    // Regular edges — deduplicate first, then assign positions
    // Build deduped list: group edges by (source key → target key)
    const regularEdges = [];
    const seenBoxEdges = new Set();

    for (const e of edgesVal) {
      if (wallEdgeIds.has(e.id)) continue;
      const fromVisible = visibleIds.has(e.from.nodeId);
      const toVisible = visibleIds.has(e.to.nodeId);
      const fromBox = nodeToGroupBox.get(e.from.nodeId);
      const toBox = nodeToGroupBox.get(e.to.nodeId);
      if (fromBox && toBox && fromBox.path === toBox.path) continue;
      if (!fromVisible && !fromBox) continue;
      if (!toVisible && !toBox) continue;

      // Dedup key for group box edges
      const dedupeKey = (fromVisible ? `n${e.from.nodeId}:${e.from.portId}` : `g${fromBox?.path}`) + '→' +
                        (toVisible ? `n${e.to.nodeId}:${e.to.portId}` : `g${toBox?.path}`);
      if ((!fromVisible || !toVisible) && seenBoxEdges.has(dedupeKey)) continue;
      seenBoxEdges.add(dedupeKey);
      regularEdges.push({ e, fromVisible, toVisible, fromBox, toBox });
    }

    // Now assign positions with correct indices
    for (const { e, fromVisible, toVisible, fromBox, toBox } of regularEdges) {
      let fromPos = null, toPos = null;

      if (fromVisible) {
        fromPos = getPortPos(e.from.nodeId, e.from.portId, true);
      } else if (fromBox) {
        const idx = boxOutputCounts.get(fromBox.path) || 0;
        boxOutputCounts.set(fromBox.path, idx + 1);
        fromPos = { x: fromBox.x + fromBox.w, y: fromBox.y + 30 + idx * 16 };
      }

      if (toVisible) {
        toPos = getPortPos(e.to.nodeId, e.to.portId, false);
      } else if (toBox) {
        const idx = boxInputCounts.get(toBox.path) || 0;
        boxInputCounts.set(toBox.path, idx + 1);
        toPos = { x: toBox.x, y: toBox.y + 30 + idx * 16 };
      }

      if (fromPos && toPos) {
        results.push({ edge: e, from: fromPos, to: toPos });
      }
    }
    return results;
  }

  // ── Draft edge ──────────────────────────────────────────────────────────

  $: draftPath = connecting ? computeDraftPath() : '';

  function computeDraftPath() {
    if (!connecting) return '';
    const from = getPortPos(connecting.fromNodeId, connecting.fromPortId, true);
    if (!from) return '';
    const dx = Math.abs(mouseWorld.x - from.x) * 0.5;
    return `M${from.x},${from.y} C${from.x + dx},${from.y} ${mouseWorld.x - dx},${mouseWorld.y} ${mouseWorld.x},${mouseWorld.y}`;
  }

  // ── D3 Zoom ─────────────────────────────────────────────────────────────

  // ── Zoom (wheel) and Pan (background drag) ─────────────────────────────

  function onWheel(e) {
    e.preventDefault();
    const rect = svgEl.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newK = Math.min(4, Math.max(0.1, transform.k * factor));

    // Zoom toward cursor
    transform = {
      x: mx - (mx - transform.x) * (newK / transform.k),
      y: my - (my - transform.y) * (newK / transform.k),
      k: newK,
    };
    viewTransform.set(transform);
  }

  function onBgMouseDown(e) {
    // Only start pan on background (not nodes/ports/groups)
    if (e.target === svgEl || e.target.classList?.contains('grid-bg')) {
      panning = true;
      panStart = { x: e.clientX, y: e.clientY };
      selectedNode.set(null);
    }
  }

  // ── Mouse handlers ──────────────────────────────────────────────────────

  function screenToWorld(clientX, clientY) {
    const rect = svgEl.getBoundingClientRect();
    return {
      x: (clientX - rect.left - transform.x) / transform.k,
      y: (clientY - rect.top - transform.y) / transform.k,
    };
  }

  function onNodeMouseDown(e) {
    const { nodeId, event } = e.detail;
    event.stopPropagation();
    event.preventDefault();
    selectedNode.set(nodeId);
    dragging = { nodeId, startX: event.clientX, startY: event.clientY, moved: false };
  }

  function onPortMouseDown(e) {
    const { nodeId, portId, isOutput, event } = e.detail;
    // Prevent node drag from starting
    dragging = null;
    if (isOutput) {
      connecting = { fromNodeId: nodeId, fromPortId: portId };
    }
  }

  function onGroupMouseDown(e) {
    const { groupPath, event } = e.detail;
    dragging = { groupPath, startX: event.clientX, startY: event.clientY };
  }

  function onGroupDrillIn(e) {
    currentGroup.set(e.detail.groupPath);
  }

  function onCanvasClick(e) {
    // Deselect on click (not mousedown — let D3 handle mousedown for pan)
    if (e.target === svgEl || e.target.classList?.contains('grid-bg')) {
      selectedNode.set(null);
    }
  }

  function onMouseMove(e) {
    mouseWorld = screenToWorld(e.clientX, e.clientY);

    if (panning) {
      transform = {
        x: transform.x + (e.clientX - panStart.x),
        y: transform.y + (e.clientY - panStart.y),
        k: transform.k,
      };
      panStart = { x: e.clientX, y: e.clientY };
      viewTransform.set(transform);
      return;
    }

    if (dragging) {
      const dx = (e.clientX - dragging.startX) / transform.k;
      const dy = (e.clientY - dragging.startY) / transform.k;

      // Only start moving after a small threshold (3px)
      if (!dragging.moved && Math.abs(dx) < 3 && Math.abs(dy) < 3) return;
      dragging.moved = true;

      if (dragging.nodeId) {
        nodes.update(ns => ns.map(n =>
          n.id === dragging.nodeId ? { ...n, x: n.x + dx, y: n.y + dy } : n
        ));
      } else if (dragging.groupPath) {
        const box = groupBoxes.find(b => b.path === dragging.groupPath);
        if (box) {
          box.x += dx;
          box.y += dy;
          // Persist position into group metadata
          groups.update(gs => {
            const g = gs[dragging.groupPath];
            if (g) { g._x = box.x; g._y = box.y; }
            return { ...gs };
          });
        }
      }

      dragging.startX = e.clientX;
      dragging.startY = e.clientY;
    }

    if (connecting) {
      draftPath = computeDraftPath();
    }
  }

  function onMouseUp(e) {
    if (connecting) {
      const target = e.target.closest?.('.port');
      if (target && target.dataset?.isOutput === 'false') {
        const toNodeId = parseInt(target.dataset?.nodeId);
        const toPortId = target.dataset?.portId;
        if (toNodeId && toPortId && toNodeId !== connecting.fromNodeId) {
          // Look up if target port is multi
          const toNode = findNode(toNodeId);
          const toComp = toNode ? getCompDef(toNode.type, $registry) : null;
          const toPort = toComp?.ports?.in?.find(p => p.id === toPortId);
          const isMulti = toPort?.multi;

          if (isMulti) {
            // Multi-port: collects into ordered list, always accept
            addEdge(connecting.fromNodeId, connecting.fromPortId, toNodeId, toPortId);
          } else {
            // Single port: replace existing connection
            edges.update(es => es.filter(
              edge => !(edge.to.nodeId === toNodeId && edge.to.portId === toPortId)
            ));
            addEdge(connecting.fromNodeId, connecting.fromPortId, toNodeId, toPortId);
          }
        }
      }
      connecting = null;
      draftPath = '';
    }
    dragging = null;
    panning = false;
  }

  function onKeyDown(e) {
    if (e.key === 'Delete' && $selectedNode) {
      removeNode($selectedNode);
      selectedNode.set(null);
    }
    if (e.key === 'Escape' && $currentGroup) {
      const dot = $currentGroup.lastIndexOf('.');
      currentGroup.set(dot >= 0 ? $currentGroup.slice(0, dot) : '');
    }
  }

  // ── Fit to view ─────────────────────────────────────────────────────────

  export function fitToView() {
    const allVisible = [...visibleNodes, ...groupBoxes.map(b => ({ x: b.x, y: b.y }))];
    if (allVisible.length === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of allVisible) {
      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, (n.x || 0) + 160);
      maxY = Math.max(maxY, (n.y || 0) + 80);
    }
    const rect = svgEl?.getBoundingClientRect();
    if (!rect) return;
    const pad = 60;
    const w = maxX - minX + pad * 2;
    const h = maxY - minY + pad * 2;
    const scale = Math.min(rect.width / w, rect.height / h, 1.5);
    const tx = (rect.width - w * scale) / 2 - (minX - pad) * scale;
    const ty = (rect.height - h * scale) / 2 - (minY - pad) * scale;
    transform = { x: tx, y: ty, k: scale };
    viewTransform.set(transform);
  }
</script>

<svelte:document on:mousemove={onMouseMove} on:mouseup={onMouseUp} on:keydown={onKeyDown} />
<svelte:window on:blur={() => { dragging = null; connecting = null; panning = false; draftPath = ''; }} />

<div class="canvas-container">
  <Breadcrumbs />

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <svg bind:this={svgEl} class="graph-canvas"
    on:wheel={onWheel}
    on:mousedown={onBgMouseDown}
  >
    <defs>
      <pattern id="grid-small" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/>
      </pattern>
      <pattern id="grid-large" width="100" height="100" patternUnits="userSpaceOnUse">
        <rect width="100" height="100" fill="url(#grid-small)"/>
        <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
      </pattern>
    </defs>

    <rect class="grid-bg" width="10000" height="10000" x="-5000" y="-5000" fill="url(#grid-large)"/>

    <g transform="translate({transform.x}, {transform.y}) scale({transform.k})">
      <!-- Edges -->
      {#each edgePositions as ep (ep.edge.id)}
        <Edge edge={ep.edge} fromPos={ep.from} toPos={ep.to} />
      {/each}

      <!-- Draft edge -->
      {#if connecting && draftPath}
        <path d={draftPath} fill="none" stroke="#4a90d9" stroke-width="2" stroke-dasharray="5,3" />
      {/if}

      <!-- Wall ports (incoming/outgoing edges from parent scope) -->
      {#if $currentGroup}
        {#each wallPorts.inputs as wp, i}
          {@const wy = WALL_PORT_START_Y + i * WALL_PORT_SPACING}
          {@const portId = wp.key.split(':')[1] || 'in'}
          {@const firstEdge = wp.edges[0]}
          {@const srcNode = firstEdge ? findNode(firstEdge.from.nodeId) : null}
          {@const srcComp = srcNode ? getCompDef(srcNode.type, $registry) : null}
          {@const srcPort = srcComp?.ports?.out?.find(p => p.id === firstEdge?.from.portId)}
          {@const dtColor = srcPort ? getDataTypeColor(srcPort.dataType, $registry) : '#4a90d9'}
          {@const rank = srcPort ? portRank(srcPort.shape) : 1}
          {@const ps = portShapeAttrs(rank, WALL_X_IN, wy, 8)}
          <g class="wall-port">
            {#if ps.tag === 'rect'}
              <rect {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {:else if ps.tag === 'polygon'}
              <polygon {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {:else}
              <circle {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {/if}
            <text x={WALL_X_IN + 14} y={wy + 3} fill={dtColor} font-size="9" font-weight="600">
              {portId}
            </text>
          </g>
        {/each}
        {#each wallPorts.outputs as wp, i}
          {@const wy = WALL_PORT_START_Y + i * WALL_PORT_SPACING}
          {@const portId = wp.key.split(':')[1] || 'out'}
          {@const firstEdge = wp.edges[0]}
          {@const dstNode = firstEdge ? findNode(firstEdge.to.nodeId) : null}
          {@const dstComp = dstNode ? getCompDef(dstNode.type, $registry) : null}
          {@const dstPort = dstComp?.ports?.in?.find(p => p.id === firstEdge?.to.portId)}
          {@const dtColor = dstPort ? getDataTypeColor(dstPort.dataType, $registry) : '#e6994a'}
          {@const rank = dstPort ? portRank(dstPort.shape) : 1}
          {@const ps = portShapeAttrs(rank, WALL_X_OUT, wy, 8)}
          <g class="wall-port">
            {#if ps.tag === 'rect'}
              <rect {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {:else if ps.tag === 'polygon'}
              <polygon {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {:else}
              <circle {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" />
            {/if}
            <text x={WALL_X_OUT - 14} y={wy + 3} fill={dtColor} font-size="9" font-weight="600" text-anchor="end">
              {portId}
            </text>
          </g>
        {/each}
      {/if}

      <!-- Group boxes -->
      {#each groupBoxes as box (box.path)}
        <GroupBox
          groupPath={box.path}
          groupInfo={box.info}
          x={box.x} y={box.y} w={box.w} h={box.h}
          inputEdges={box.inputs}
          outputEdges={box.outputs}
          on:drillIn={onGroupDrillIn}
          on:groupMouseDown={onGroupMouseDown}
        />
      {/each}

      <!-- Nodes -->
      {#each visibleNodes as node (node.id)}
        <Node
          {node}
          selected={node.id === $selectedNode}
          on:nodeMouseDown={onNodeMouseDown}
          on:portMouseDown={onPortMouseDown}
        />
      {/each}
    </g>
  </svg>

  <div class="scope-info">
    {$currentGroup || 'Pipeline'}: {visibleNodes.length} nodes, {childGPs.length} groups
  </div>
</div>

<style>
  .canvas-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    position: relative;
    flex: 1;
    min-height: 0;
  }
  .graph-canvas {
    flex: 1;
    width: 100%;
    height: 100%;
    display: block;
    touch-action: none;
    user-select: none;
  }
  .scope-info {
    position: absolute;
    top: 48px;
    right: 12px;
    font-size: 11px;
    color: #888;
    pointer-events: none;
  }
</style>
