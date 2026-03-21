// cards.js — Engine card state

import { writable } from 'svelte/store';

export const cards = writable([]);
export const activeCardIdx = writable(0);

export function createCard(name) {
  return {
    id: crypto.randomUUID(),
    name: name || 'Engine',
    starred: true,
    engine: {
      built: false,
      training: false,
      lossHistory: [],
      modelHash: null,
      summary: null,
      totalSteps: 0,
    },
    graph: null,  // serialized graph snapshot
  };
}
