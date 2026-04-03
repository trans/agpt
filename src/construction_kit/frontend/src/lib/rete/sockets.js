// sockets.js — Rete socket types from our data type registry

import { ClassicPreset } from 'rete';

const socketCache = new Map();

/** Get or create a socket for a data type */
export function getSocket(dataType) {
  if (!socketCache.has(dataType)) {
    socketCache.set(dataType, new ClassicPreset.Socket(dataType));
  }
  return socketCache.get(dataType);
}

/** Initialize sockets from registry data types */
export function initSockets(registry) {
  if (!registry?.dataTypes) return;
  for (const dt of Object.keys(registry.dataTypes)) {
    getSocket(dt);
  }
}

/** Check if an output socket can connect to an input socket */
export function canConnect(outSocket, inSocket) {
  // For now, allow same-type connections and any matrix-compatible types
  if (outSocket.name === inSocket.name) return true;
  // TODO: add type compatibility rules (e.g., vector → matrix broadcast)
  return true;
}
