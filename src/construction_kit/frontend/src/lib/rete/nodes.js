// nodes.js — Rete node factories for our three node types

import { ClassicPreset } from 'rete';
import { getSocket } from './sockets.js';

/**
 * Create an OpNode (regular computation node) from a store node + component definition.
 * @param {object} storeNode - { id, type, group, params, x, y }
 * @param {object} compDef - component definition from registry
 * @returns {ClassicPreset.Node}
 */
export function createOpNode(storeNode, compDef) {
  const node = new ClassicPreset.Node(compDef?.label || storeNode.type);
  // Use stringified store ID so Rete can use it
  node.id = String(storeNode.id);

  // Attach metadata for our custom rendering
  node.meta = {
    storeId: storeNode.id,
    type: storeNode.type,
    color: compDef?.color || '#888',
    params: storeNode.params,
  };

  // Add input ports
  for (const port of (compDef?.ports?.in || [])) {
    const socket = getSocket(port.dataType || 'matrix');
    const input = new ClassicPreset.Input(socket, port.label || port.id, !!port.multi);
    input.index = (compDef.ports.in.indexOf(port));
    node.addInput(port.id, input);
  }

  // Add output ports
  for (const port of (compDef?.ports?.out || [])) {
    const socket = getSocket(port.dataType || 'matrix');
    const output = new ClassicPreset.Output(socket, port.label || port.id, true);
    output.index = (compDef.ports.out.indexOf(port));
    node.addOutput(port.id, output);
  }

  return node;
}

/**
 * Create a GroupBoxNode (collapsed sub-group) from group metadata.
 * Dynamic ports from group.ports.in / group.ports.out.
 * @param {string} groupPath - e.g., "coop.xfmr_a"
 * @param {object} groupInfo - { label, type, params, ports, portMap }
 * @param {object} compDef - component definition for the group type (optional)
 * @returns {ClassicPreset.Node}
 */
export function createGroupBoxNode(groupPath, groupInfo, compDef) {
  const node = new ClassicPreset.Node(groupInfo?.label || groupPath.split('.').pop());
  node.id = "group:" + groupPath;

  node.meta = {
    isGroup: true,
    groupPath,
    color: compDef?.color || '#666',
    type: groupInfo?.type,
  };

  // Dynamic ports from declared group ports
  for (const [i, port] of (groupInfo?.ports?.in || []).entries()) {
    const socket = getSocket(port.dataType || 'matrix');
    const input = new ClassicPreset.Input(socket, port.label || port.id, !!port.multi);
    input.index = i;
    node.addInput(port.id, input);
  }

  for (const [i, port] of (groupInfo?.ports?.out || []).entries()) {
    const socket = getSocket(port.dataType || 'matrix');
    const output = new ClassicPreset.Output(socket, port.label || port.id, true);
    output.index = i;
    node.addOutput(port.id, output);
  }

  return node;
}

/**
 * Create a WallPortNode (boundary ports when drilled into a group).
 * @param {string} groupPath - current group being viewed
 * @param {object} groupInfo - group metadata with declared ports
 * @param {boolean} isInput - true for input wall (left side), false for output wall (right side)
 * @returns {ClassicPreset.Node}
 */
export function createWallNode(groupPath, groupInfo, isInput) {
  const label = isInput ? '← In' : 'Out →';
  const node = new ClassicPreset.Node(label);
  // Wall nodes use the group ref as their ID
  // Input and output walls share the same group ref but have different ports
  node.id = isInput ? `wall-in:${groupPath}` : `wall-out:${groupPath}`;

  node.meta = {
    isWall: true,
    isWallInput: isInput,
    groupPath,
  };

  if (isInput) {
    // Wall input node has OUTPUT ports (data flows from wall into the scope)
    for (const [i, port] of (groupInfo?.ports?.in || []).entries()) {
      const socket = getSocket(port.dataType || 'matrix');
      const output = new ClassicPreset.Output(socket, port.label || port.id, true);
      output.index = i;
      node.addOutput(port.id, output);
    }
  } else {
    // Wall output node has INPUT ports (data flows from scope into the wall)
    for (const [i, port] of (groupInfo?.ports?.out || []).entries()) {
      const socket = getSocket(port.dataType || 'matrix');
      const input = new ClassicPreset.Input(socket, port.label || port.id, true);
      input.index = i;
      node.addInput(port.id, input);
    }
  }

  return node;
}
