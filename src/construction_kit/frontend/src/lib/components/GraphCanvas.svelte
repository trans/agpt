<script>
  import { onMount } from 'svelte';
  import { zoom as d3Zoom } from 'd3-zoom';
  import { select as d3Select } from 'd3-selection';
  import { nodes, edges, groups, findNode, addEdge, removeNode, childGroupPaths, nodesUnderGroup, groupBoundaryPorts } from '../stores/graph.js';
  import { currentGroup, selectedNode, registry, viewTransform, getCompDef, getDataTypeColor } from '../stores/ui.js';
  import Node from './Node.svelte';
  import Edge from './Edge.svelte';
  import GroupBox from './GroupBox.svelte';
  import Breadcrumbs from './Breadcrumbs.svelte';

  let svgEl;
  let zoomBehavior;
  let transform = { x: 0, y: 0, k: 1 };

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
      // Use stored position or auto-layout
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

  // ── Edge positions ──────────────────────────────────────────────────────

  $: edgePositions = computeEdgePositions($edges, visibleNodes, groupBoxes, $nodes);

  function computeEdgePositions(edgesVal, visible, gBoxes, allNodes) {
    const visibleIds = new Set(visible.map(n => n.id));
    const results = [];

    // Build lookup: nodeId → which group box contains it (if collapsed)
    const nodeToGroupBox = new Map();
    for (const box of gBoxes) {
      const gNodes = nodesUnderGroup(box.path, allNodes);
      for (const n of gNodes) nodeToGroupBox.set(n.id, box);
    }

    for (const e of edgesVal) {
      const fromVisible = visibleIds.has(e.from.nodeId);
      const toVisible = visibleIds.has(e.to.nodeId);
      const fromBox = nodeToGroupBox.get(e.from.nodeId);
      const toBox = nodeToGroupBox.get(e.to.nodeId);

      let fromPos = null, toPos = null;

      if (fromVisible) {
        fromPos = getPortPos(e.from.nodeId, e.from.portId, true);
      } else if (fromBox) {
        // Edge comes from inside a collapsed group — connect to group box right side
        fromPos = { x: fromBox.x + fromBox.w, y: fromBox.y + fromBox.h / 2 };
      }

      if (toVisible) {
        toPos = getPortPos(e.to.nodeId, e.to.portId, false);
      } else if (toBox) {
        // Edge goes into a collapsed group — connect to group box left side
        toPos = { x: toBox.x, y: toBox.y + toBox.h / 2 };
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

  onMount(() => {
    zoomBehavior = d3Zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        transform = { x: event.transform.x, y: event.transform.y, k: event.transform.k };
        viewTransform.set(transform);
      });
    d3Select(svgEl).call(zoomBehavior);
  });

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

  function onCanvasMouseDown(e) {
    if (e.target === svgEl || e.target.classList?.contains('grid-bg')) {
      selectedNode.set(null);
    }
  }

  function onMouseMove(e) {
    mouseWorld = screenToWorld(e.clientX, e.clientY);

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
        if (toNodeId && toPortId) {
          addEdge(connecting.fromNodeId, connecting.fromPortId, toNodeId, toPortId);
        }
      }
      connecting = null;
      draftPath = '';
    }
    dragging = null;
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
    d3Select(svgEl).call(zoomBehavior.transform,
      { x: tx, y: ty, k: scale, __proto__: { rescaleX: x => x, rescaleY: y => y } }
    );
    transform = { x: tx, y: ty, k: scale };
  }
</script>

<svelte:document on:mousemove={onMouseMove} on:mouseup={onMouseUp} on:keydown={onKeyDown} />
<svelte:window on:blur={() => { dragging = null; connecting = null; draftPath = ''; }} />

<div class="canvas-container">
  <Breadcrumbs />

  <svg bind:this={svgEl} class="graph-canvas" on:mousedown={onCanvasMouseDown}>
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
