// ui.js — UI state stores

import { writable } from 'svelte/store';

export const currentGroup = writable('');      // dotted group path being viewed
export const selectedNode = writable(null);    // node ID or null
export const registry = writable(null);        // components.json data

// Zoom/pan state (managed by D3 but exposed for components)
export const viewTransform = writable({ x: 0, y: 0, k: 1 });

// Get component definition by type
export function getCompDef(type, registryValue) {
  if (!registryValue) return null;
  return registryValue.components.find(c => c.type === type) || null;
}

// Get data type color
export function getDataTypeColor(dataType, registryValue) {
  return registryValue?.dataTypes?.[dataType]?.color || '#888';
}
