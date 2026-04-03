// sync.js — Bidirectional sync between Svelte stores and Rete editor

import { ClassicPreset } from 'rete';
import { AreaExtensions } from 'rete-area-plugin';
import { get } from 'svelte/store';
import { nodes, edges, groups, addEdge, removeEdge, updateNode, updateGroup, childGroupPaths, nodesUnderGroup, isGroupRef, getGroupInfo } from '../stores/graph.js';
import { registry, currentGroup } from '../stores/ui.js';
import { createOpNode, createGroupBoxNode, createWallNode } from './nodes.js';
import { initSockets } from './sockets.js';
let syncing = false;

/**
 * Sync the current scope to a Rete editor instance.
 */
export async function syncScopeToRete(editor, area) {
  syncing = true;
  try {
    await editor.clear();

    const curGroup = get(currentGroup);
    const allNodes = get(nodes);
    const allEdges = get(edges);
    const allGroups = get(groups);
    const reg = get(registry);

    initSockets(reg);

    const reteNodeIds = new Set();

    // 1. OpNodes — nodes directly in this group
    const visibleNodes = allNodes.filter(n => n.group === curGroup);
    for (const storeNode of visibleNodes) {
      const compDef = reg?.components?.find(c => c.type === storeNode.type);
      const reteNode = createOpNode(storeNode, compDef);
      await editor.addNode(reteNode);
      await area.translate(reteNode.id, { x: storeNode.x, y: storeNode.y });
      reteNodeIds.add(reteNode.id);
    }

    // 2. GroupBoxNodes — collapsed child groups
    const childPaths = childGroupPaths(curGroup, allGroups);
    let autoX = -100, autoY = -200;
    for (const gp of childPaths) {
      const gNodes = nodesUnderGroup(gp, allNodes);
      if (gNodes.length === 0) continue;
      const groupInfo = allGroups[gp];
      const compDef = reg?.components?.find(c => c.type === groupInfo?.type);
      const reteNode = createGroupBoxNode(gp, groupInfo, compDef);

      await editor.addNode(reteNode);
      const gx = groupInfo?._x ?? autoX;
      const gy = groupInfo?._y ?? autoY;
      await area.translate(reteNode.id, { x: gx, y: gy });
      reteNodeIds.add(reteNode.id);
      autoX += 250;
    }

    // 3. WallPortNodes — boundary when drilled into a group
    if (curGroup) {
      const groupInfo = allGroups[curGroup];
      if (groupInfo) {
        const wallIn = createWallNode(curGroup, groupInfo, true);
        const wallOut = createWallNode(curGroup, groupInfo, false);
        await editor.addNode(wallIn);
        await editor.addNode(wallOut);
        await area.translate(wallIn.id, { x: -400, y: 0 });
        await area.translate(wallOut.id, { x: 900, y: 0 });
        reteNodeIds.add(wallIn.id);
        reteNodeIds.add(wallOut.id);
      }
    }

    // 4. Connections — edges visible at this scope
    for (const edge of allEdges) {
      const resolved = resolveEdge(edge, curGroup, allGroups, reteNodeIds);
      if (!resolved) continue;

      const sourceNode = editor.getNode(resolved.fromId);
      const targetNode = editor.getNode(resolved.toId);
      if (!sourceNode || !targetNode) continue;
      if (!sourceNode.outputs[resolved.fromPort]) continue;
      if (!targetNode.inputs[resolved.toPort]) continue;

      try {
        const conn = new ClassicPreset.Connection(sourceNode, resolved.fromPort, targetNode, resolved.toPort);
        conn.id = String(edge.id);
        await editor.addConnection(conn);
      } catch (e) {
        console.warn('Edge', edge.id, ':', e.message);
      }
    }

    // Fit view after sync
    await AreaExtensions.zoomAt(area, editor.getNodes());

  } finally {
    syncing = false;
  }
}

/**
 * Resolve an edge's endpoints to Rete node IDs at the current scope.
 * Returns { fromId, fromPort, toId, toPort } or null if not visible.
 */
function resolveEdge(edge, curGroup, allGroups, reteNodeIds) {
  const fromId = resolveEndpoint(edge.from.nodeId, edge.from.portId, true, curGroup, allGroups, reteNodeIds);
  const toId = resolveEndpoint(edge.to.nodeId, edge.to.portId, false, curGroup, allGroups, reteNodeIds);
  if (!fromId || !toId) return null;
  return { fromId: fromId.nodeId, fromPort: fromId.portId, toId: toId.nodeId, toPort: toId.portId };
}

function resolveEndpoint(nodeId, portId, isOutput, curGroup, allGroups, reteNodeIds) {
  if (isGroupRef(nodeId)) {
    const gPath = nodeId.slice(6);

    if (gPath === curGroup) {
      // This endpoint targets the current group boundary → wall node
      // Input wall has outputs (data flows in), output wall has inputs (data flows out)
      const wallId = isOutput ? `wall-out:${curGroup}` : `wall-in:${curGroup}`;
      if (reteNodeIds.has(wallId)) return { nodeId: wallId, portId };
      // Try the other wall (edge direction might be inverted from wall perspective)
      const altWallId = isOutput ? `wall-in:${curGroup}` : `wall-out:${curGroup}`;
      if (reteNodeIds.has(altWallId)) return { nodeId: altWallId, portId };
      return null;
    }

    // It's a child group box
    if (reteNodeIds.has(nodeId)) return { nodeId, portId };
    return null;
  }

  // Regular node
  const reteId = String(nodeId);
  if (reteNodeIds.has(reteId)) return { nodeId: reteId, portId };
  return null;
}

/**
 * Set up Rete → Store event listeners.
 */
export function setupReteListeners(editor, area, drillInCallback) {

  editor.addPipe(context => {
    if (syncing) return context;

    if (context.type === 'connectioncreated') {
      const conn = context.data;
      const fromNodeId = parseStoreNodeId(conn.source);
      const toNodeId = parseStoreNodeId(conn.target);
      if (fromNodeId != null && toNodeId != null) {
        addEdge(fromNodeId, conn.sourceOutput, toNodeId, conn.targetInput);
      }
    }

    if (context.type === 'connectionremoved') {
      const conn = context.data;
      const storeEdgeId = parseInt(conn.id);
      if (!isNaN(storeEdgeId)) {
        removeEdge(storeEdgeId);
      }
    }

    return context;
  });

  // Double-click on group box → drill in
  // Rete's pointer handling prevents native dblclick from firing,
  // so we detect double-click via pointerup timing.
  let lastClickTime = 0;
  let lastClickId = null;
  area.container.addEventListener('pointerup', (e) => {
    const nodeEl = e.target.closest('[data-rete-node-id]');
    if (!nodeEl) { lastClickId = null; return; }
    const id = nodeEl.dataset.reteNodeId;
    const now = Date.now();
    if (id === lastClickId && now - lastClickTime < 400) {
      const node = editor.getNode(id);
      if (node?.meta?.isGroup && node.meta.groupPath) {
        drillInCallback?.(node.meta.groupPath);
      }
      lastClickId = null;
    } else {
      lastClickId = id;
      lastClickTime = now;
    }
  });

  area.addPipe(context => {
    if (syncing) return context;

    if (context.type === 'nodetranslated') {
      const { id, position } = context.data;
      const storeId = parseInt(id);
      if (!isNaN(storeId)) {
        updateNode(storeId, { x: position.x, y: position.y });
      }
      if (typeof id === 'string' && id.startsWith('group:')) {
        updateGroup(id.slice(6), { _x: position.x, _y: position.y });
      }
    }

    return context;
  });
}

function parseStoreNodeId(reteId) {
  if (reteId.startsWith('wall-in:') || reteId.startsWith('wall-out:')) {
    return 'group:' + reteId.split(':').slice(1).join(':');
  }
  if (reteId.startsWith('group:')) return reteId;
  const num = parseInt(reteId);
  return isNaN(num) ? reteId : num;
}
