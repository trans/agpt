// graph.js — Flat graph state store
// All nodes in one flat list, organized by group tags.
// All edges direct primitive-to-primitive. No boundaries.

import { writable, derived } from 'svelte/store';

// ── Core State ──────────────────────────────────────────────────────────────

export const nodes = writable([]);
export const edges = writable([]);
export const groups = writable({});   // { "coop.xfmr_a": { label, type, params } }

let nextId = 1;
export function newId() { return nextId++; }
export function setNextId(n) { nextId = n; }

// ── Node Operations ─────────────────────────────────────────────────────────

export function addNode(type, group, params = {}, x = 0, y = 0) {
  const id = newId();
  const node = { id, type, group, params, x, y };
  nodes.update(ns => [...ns, node]);
  return node;
}

export function removeNode(id) {
  nodes.update(ns => ns.filter(n => n.id !== id));
  edges.update(es => es.filter(e => e.from.nodeId !== id && e.to.nodeId !== id));
}

export function findNode(id) {
  let found = null;
  nodes.subscribe(ns => { found = ns.find(n => n.id === id) || null; })();
  return found;
}

export function updateNode(id, updates) {
  nodes.update(ns => ns.map(n => n.id === id ? { ...n, ...updates } : n));
}

// ── Edge Operations ─────────────────────────────────────────────────────────

export function addEdge(fromNodeId, fromPortId, toNodeId, toPortId) {
  const id = newId();
  const edge = { id, from: { nodeId: fromNodeId, portId: fromPortId }, to: { nodeId: toNodeId, portId: toPortId } };
  edges.update(es => [...es, edge]);
  return edge;
}

export function removeEdge(id) {
  edges.update(es => es.filter(e => e.id !== id));
}

// ── Group Operations ────────────────────────────────────────────────────────

export function addGroup(path, label, type, params = {}, ports = null) {
  const info = { label, type, params };
  if (ports) info.ports = ports;
  groups.update(gs => ({ ...gs, [path]: info }));
}

export function updateGroup(path, updates) {
  groups.update(gs => {
    if (!gs[path]) return gs;
    return { ...gs, [path]: { ...gs[path], ...updates } };
  });
}

export function removeGroup(path) {
  groups.update(gs => {
    const next = { ...gs };
    delete next[path];
    const prefix = path + '.';
    for (const key of Object.keys(next)) {
      if (key.startsWith(prefix)) delete next[key];
    }
    return next;
  });
}

export function getGroupInfo(path) {
  let info = null;
  groups.subscribe(gs => { info = gs[path] || null; })();
  return info;
}

// ── Group Port Helpers ──────────────────────────────────────────────────

export function isGroupRef(nodeId) {
  return typeof nodeId === 'string' && nodeId.startsWith('group:');
}

export function groupPathFromRef(nodeId) {
  return nodeId.slice(6);
}

// ── Param Inheritance ───────────────────────────────────────────────────────

export function resolveGroupParam(groupPath, paramName, groupsValue) {
  let path = groupPath;
  while (true) {
    const info = groupsValue[path];
    if (info?.params?.[paramName] !== undefined) return info.params[paramName];
    if (!path) break;
    const dot = path.lastIndexOf('.');
    path = dot >= 0 ? path.slice(0, dot) : '';
  }
  const defaults = { d: 64, stream_dim: 64, ff_mult: 4, n_heads: 4 };
  return defaults[paramName];
}

export function resolvedParams(groupPath, groupsValue) {
  const d = resolveGroupParam(groupPath, 'd', groupsValue) || 64;
  const ff_mult = resolveGroupParam(groupPath, 'ff_mult', groupsValue) || 4;
  const stream_dim = resolveGroupParam(groupPath, 'stream_dim', groupsValue) || d;
  const n_heads = resolveGroupParam(groupPath, 'n_heads', groupsValue) || Math.max(1, Math.floor(d / 16));
  return { d, ff_mult, d_ff: d * ff_mult, stream_dim, n_heads };
}

// ── Serialization ───────────────────────────────────────────────────────────

export function serializeGraph() {
  let result;
  const unsub1 = nodes.subscribe(ns => {
    const unsub2 = edges.subscribe(es => {
      const unsub3 = groups.subscribe(gs => {
        result = {
          nodes: ns.map(n => ({ id: n.id, type: n.type, group: n.group, x: n.x, y: n.y, params: n.params })),
          edges: es.map(e => ({ id: e.id, from: e.from, to: e.to })),
          groups: { ...gs },
        };
      });
      unsub3();
    });
    unsub2();
  });
  unsub1();
  return result;
}

export function deserializeGraph(data) {
  const ns = (data.nodes || []).map(n => ({
    id: n.id, type: n.type, group: n.group || '', x: n.x || 0, y: n.y || 0, params: n.params || {},
  }));
  const es = (data.edges || []).map(e => ({
    id: e.id || newId(), from: e.from, to: e.to,
  }));

  // Update nextId
  for (const n of ns) { if (n.id >= nextId) nextId = n.id + 1; }
  for (const e of es) { if (e.id >= nextId) nextId = e.id + 1; }

  nodes.set(ns);
  edges.set(es);
  groups.set(data.groups || {});
}

export function clearGraph() {
  nodes.set([]);
  edges.set([]);
  groups.set({});
  nextId = 1;
}

// ── Derived Queries ─────────────────────────────────────────────────────────

// Direct child group paths of a given group
export function childGroupPaths(currentGroup, groupsValue) {
  const result = [];
  const seen = new Set();
  const prefix = currentGroup ? currentGroup + '.' : '';
  const prefixLen = prefix.length;

  for (const path of Object.keys(groupsValue)) {
    if (currentGroup && !path.startsWith(prefix)) continue;
    if (!currentGroup && path.includes('.')) {
      const root = path.split('.')[0];
      if (!seen.has(root) && groupsValue[root]) { seen.add(root); result.push(root); }
      continue;
    }
    if (path === currentGroup) continue;
    const rest = path.slice(prefixLen);
    const dot = rest.indexOf('.');
    const childPath = prefix + (dot >= 0 ? rest.slice(0, dot) : rest);
    if (!seen.has(childPath) && groupsValue[childPath]) { seen.add(childPath); result.push(childPath); }
  }
  return result;
}

// Nodes in a group or any subgroup
export function nodesUnderGroup(groupPath, nodesValue) {
  if (!groupPath) return nodesValue;
  const prefix = groupPath + '.';
  return nodesValue.filter(n => n.group === groupPath || n.group.startsWith(prefix));
}

