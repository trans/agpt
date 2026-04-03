<script>
  import { onMount } from 'svelte';
  import { nodes, edges, groups, findNode, addEdge, removeEdge, removeNode, childGroupPaths, nodesUnderGroup, isGroupRef, groupPathFromRef, getGroupInfo } from '../stores/graph.js';
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
  let selectedEdgeId = null;

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
      boxes.push({ path: gp, info, x, y, w: boxW, h: boxH });
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

  // Position for a declared group port on a group box
  function getDeclaredPortPos(box, portId, isOutput) {
    const slots = isOutput ? (box.info?.ports?.out || []) : (box.info?.ports?.in || []);
    const idx = slots.findIndex(s => s.id === portId);
    if (idx < 0) return null;
    const px = isOutput ? box.x + box.w : box.x;
    return { x: px, y: box.y + 30 + idx * 16 };
  }

  // Resolve any endpoint position — works for both real nodes and group refs
  function resolveEndpointPos(nodeId, portId, isOutput, gBoxes, visibleIds) {
    if (isGroupRef(nodeId)) {
      const gPath = groupPathFromRef(nodeId);
      const box = gBoxes.find(b => b.path === gPath);
      if (box) return getDeclaredPortPos(box, portId, isOutput);
      return null;
    }
    if (visibleIds.has(nodeId)) return getPortPos(nodeId, portId, isOutput);
    return null;
  }

  // Wall port positions — at the viewport edges in world coordinates
  $: WALL_X_IN = (-transform.x / transform.k) - 10;
  $: WALL_X_OUT = ((-transform.x + (svgEl?.getBoundingClientRect()?.width || 760)) / transform.k) + 10;

  // ── Edge positions ──────────────────────────────────────────────────────

  $: edgePositions = computeEdgePositions($edges, visibleNodes, groupBoxes, $nodes, $currentGroup, WALL_X_IN, WALL_X_OUT, $groups);

  function computeEdgePositions(edgesVal, visible, gBoxes, allNodes, curGroup, wallXIn, wallXOut, groupsVal) {
    const visibleIds = new Set(visible.map(n => n.id));
    const results = [];

    // Build group ref → box lookup
    const groupRefToBox = new Map();
    for (const box of gBoxes) {
      groupRefToBox.set("group:" + box.path, box);
    }

    // Track which edges are handled by wall ports (to avoid duplicates)
    const wallEdgeIds = new Set();

    // Wall port edges: edges targeting "group:<currentGroup>" from outside
    // Inside the group, these appear as wall port → internal node (via portMap)
    if (curGroup) {
      const curGroupInfo = groupsVal[curGroup];
      const gRef = "group:" + curGroup;
      const wallInPorts = curGroupInfo?.ports?.in || [];
      const wallOutPorts = curGroupInfo?.ports?.out || [];
      const portMap = curGroupInfo?.portMap || {};

      // Resolve a portMap target to a position (handles both real nodes and group refs)
      function resolveMapTarget(t, output) {
        if (isGroupRef(t.nodeId)) {
          const box = groupRefToBox.get(t.nodeId);
          return box ? getDeclaredPortPos(box, t.portId, output) : null;
        }
        if (visibleIds.has(t.nodeId)) return getPortPos(t.nodeId, t.portId, output);
        // Node inside a collapsed sub-group — find the box containing it
        const box = gBoxes.find(b => nodesUnderGroup(b.path, allNodes).some(n => n.id === t.nodeId));
        return box ? getDeclaredPortPos(box, t.portId, output) : null;
      }

      // Input wall ports: edges where to.nodeId === "group:<curGroup>"
      for (const e of edgesVal) {
        if (e.to.nodeId === gRef) {
          wallEdgeIds.add(e.id);
          const slotIdx = wallInPorts.findIndex(p => p.id === e.to.portId);
          if (slotIdx < 0) continue;
          const wallY = WALL_PORT_START_Y + slotIdx * WALL_PORT_SPACING;

          const targets = portMap[e.to.portId] || [];
          for (let ti = 0; ti < targets.length; ti++) {
            const toPos = resolveMapTarget(targets[ti], false);
            if (toPos) {
              results.push({ key: `${e.id}_in${ti}`, edge: e, from: { x: wallXIn, y: wallY }, to: toPos });
            }
          }
        } else if (e.from.nodeId === gRef) {
          wallEdgeIds.add(e.id);
          const slotIdx = wallOutPorts.findIndex(p => p.id === e.from.portId);
          if (slotIdx < 0) continue;
          const wallY = WALL_PORT_START_Y + slotIdx * WALL_PORT_SPACING;

          const sources = portMap[e.from.portId] || [];
          for (let si = 0; si < sources.length; si++) {
            const fromPos = resolveMapTarget(sources[si], true);
            if (fromPos) {
              results.push({ key: `${e.id}_out${si}`, edge: e, from: fromPos, to: { x: wallXOut, y: wallY } });
            }
          }
        }
      }
    }

    // Regular edges
    for (const e of edgesVal) {
      if (wallEdgeIds.has(e.id)) continue;

      const fromIsGroup = isGroupRef(e.from.nodeId);
      const toIsGroup = isGroupRef(e.to.nodeId);
      const fromVisible = !fromIsGroup && visibleIds.has(e.from.nodeId);
      const toVisible = !toIsGroup && visibleIds.has(e.to.nodeId);
      const fromBox = fromIsGroup ? groupRefToBox.get(e.from.nodeId) : null;
      const toBox = toIsGroup ? groupRefToBox.get(e.to.nodeId) : null;

      // Skip if neither endpoint is visible or a visible group box
      if (!fromVisible && !fromBox) continue;
      if (!toVisible && !toBox) continue;

      let fromPos = null, toPos = null;

      if (fromVisible) {
        fromPos = getPortPos(e.from.nodeId, e.from.portId, true);
      } else if (fromBox) {
        fromPos = getDeclaredPortPos(fromBox, e.from.portId, true);
      }

      if (toVisible) {
        toPos = getPortPos(e.to.nodeId, e.to.portId, false);
      } else if (toBox) {
        toPos = getDeclaredPortPos(toBox, e.to.portId, false);
      }

      if (fromPos && toPos) {
        results.push({ key: e.id, edge: e, from: fromPos, to: toPos });
      }
    }
    return results;
  }

  // ── Draft edge ──────────────────────────────────────────────────────────

  $: draftPath = connecting ? computeDraftPath() : '';

  function computeDraftPath() {
    if (!connecting) return '';
    let from = null;
    if (isGroupRef(connecting.fromNodeId)) {
      const gPath = groupPathFromRef(connecting.fromNodeId);
      const box = groupBoxes.find(b => b.path === gPath);
      if (box) from = getDeclaredPortPos(box, connecting.fromPortId, true);
    } else {
      from = getPortPos(connecting.fromNodeId, connecting.fromPortId, true);
    }
    if (!from) return '';
    const dx = Math.abs(mouseWorld.x - from.x) * 0.5;
    return `M${from.x},${from.y} C${from.x + dx},${from.y} ${mouseWorld.x - dx},${mouseWorld.y} ${mouseWorld.x},${mouseWorld.y}`;
  }

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
      selectedEdgeId = null;
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
    selectedEdgeId = null;
    dragging = { nodeId, startX: event.clientX, startY: event.clientY, moved: false };
  }

  function onEdgeClick(e) {
    const { edgeId } = e.detail;
    selectedEdgeId = edgeId;
    selectedNode.set(null);
  }

  function onPortMouseDown(e) {
    const { nodeId, portId, isOutput, event } = e.detail;
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
        const rawNodeId = target.dataset?.nodeId;
        const toPortId = target.dataset?.portId;
        // Parse nodeId — could be integer (real node) or "group:..." (group port)
        const toNodeId = isGroupRef(rawNodeId) ? rawNodeId : parseInt(rawNodeId);
        if (toNodeId && toPortId && toNodeId !== connecting.fromNodeId) {
          // Look up if target port is multi
          let isMulti = false;
          if (isGroupRef(toNodeId)) {
            const gInfo = getGroupInfo(groupPathFromRef(toNodeId));
            const port = gInfo?.ports?.in?.find(p => p.id === toPortId);
            isMulti = !!port?.multi;
          } else {
            const toNode = findNode(toNodeId);
            const toComp = toNode ? getCompDef(toNode.type, $registry) : null;
            const toPort = toComp?.ports?.in?.find(p => p.id === toPortId);
            isMulti = !!toPort?.multi;
          }

          if (isMulti) {
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
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (selectedEdgeId) {
        removeEdge(selectedEdgeId);
        selectedEdgeId = null;
      } else if ($selectedNode) {
        removeNode($selectedNode);
        selectedNode.set(null);
      }
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
      {#each edgePositions as ep (ep.key)}
        <Edge
          edge={ep.edge}
          fromPos={ep.from}
          toPos={ep.to}
          selected={ep.edge.id === selectedEdgeId}
          on:edgeClick={onEdgeClick}
        />
      {/each}

      <!-- Draft edge -->
      {#if connecting && draftPath}
        <path d={draftPath} fill="none" stroke="#4a90d9" stroke-width="2" stroke-dasharray="5,3" />
      {/if}

      <!-- Wall ports (from declared group ports) -->
      {#if $currentGroup && $groups[$currentGroup]}
        {@const curGroupInfo = $groups[$currentGroup]}
        {@const gRef = "group:" + $currentGroup}
        {@const wallIn = curGroupInfo.ports?.in || []}
        {@const wallOut = curGroupInfo.ports?.out || []}
        {#each wallIn as p, i}
          {@const wy = WALL_PORT_START_Y + i * WALL_PORT_SPACING}
          {@const dtColor = getDataTypeColor(p.dataType, $registry)}
          {@const rank = portRank(p.shape)}
          {@const ps = portShapeAttrs(rank, WALL_X_IN, wy, 8)}
          {@const isConnected = $edges.some(e => e.to.nodeId === gRef && e.to.portId === p.id)}
          <g class="wall-port port port-in"
             data-node-id={gRef}
             data-port-id={p.id}
             data-is-output="false"
             style="cursor: crosshair"
          >
            {#if ps.tag === 'rect'}
              <rect {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {:else if ps.tag === 'polygon'}
              <polygon {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {:else}
              <circle {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {/if}
            <text x={WALL_X_IN + 14} y={wy + 3} fill={dtColor} font-size="9" font-weight="600"
                  opacity={isConnected ? 1 : 0.4}>
              {p.label || p.id}
            </text>
          </g>
        {/each}
        {#each wallOut as p, i}
          {@const wy = WALL_PORT_START_Y + i * WALL_PORT_SPACING}
          {@const dtColor = getDataTypeColor(p.dataType, $registry)}
          {@const rank = portRank(p.shape)}
          {@const ps = portShapeAttrs(rank, WALL_X_OUT, wy, 8)}
          {@const isConnected = $edges.some(e => e.from.nodeId === gRef && e.from.portId === p.id)}
          <g class="wall-port port port-out"
             data-node-id={gRef}
             data-port-id={p.id}
             data-is-output="true"
             style="cursor: crosshair"
          >
            {#if ps.tag === 'rect'}
              <rect {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {:else if ps.tag === 'polygon'}
              <polygon {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {:else}
              <circle {...ps.attrs} fill={dtColor} stroke={dtColor} stroke-width="2" opacity={isConnected ? 1 : 0.4} />
            {/if}
            <text x={WALL_X_OUT - 14} y={wy + 3} fill={dtColor} font-size="9" font-weight="600" text-anchor="end"
                  opacity={isConnected ? 1 : 0.4}>
              {p.label || p.id}
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
