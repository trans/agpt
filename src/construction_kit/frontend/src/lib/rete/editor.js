// editor.js — Rete editor instance factory

import { NodeEditor } from 'rete';
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin';
import { SveltePlugin, Presets as SveltePresets } from 'rete-svelte-plugin';
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin';
import NodeView from './views/NodeView.svelte';
import SocketView from './views/SocketView.svelte';

/**
 * Create a new Rete editor instance mounted on a container element.
 */
export async function createEditor(container) {
  const editor = new NodeEditor();
  const area = new AreaPlugin(container);
  const connection = new ConnectionPlugin();
  const render = new SveltePlugin();

  // Custom rendering — our dark-themed node and socket components
  render.addPreset(SveltePresets.classic.setup({
    customize: {
      node() { return NodeView; },
      socket() { return SocketView; },
    },
  }));

  connection.addPreset(ConnectionPresets.classic.setup());

  editor.use(area);
  area.use(connection);
  area.use(render);

  // Suppress Rete's default zoom-on-dblclick so native dblclick events propagate
  area.addPipe(context => {
    if (context.type === 'zoom' && context.data.source === 'dblclick') return;
    return context;
  });

  // Multi-select with Ctrl
  AreaExtensions.selectableNodes(area, AreaExtensions.selector(), {
    accumulating: AreaExtensions.accumulateOnCtrl(),
  });
  AreaExtensions.simpleNodesOrder(area);

  return {
    editor,
    area,
    async destroy() {
      area.destroy();
    },
  };
}
