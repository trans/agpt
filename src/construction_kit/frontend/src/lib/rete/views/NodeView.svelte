<script>
  import { Ref } from 'rete-svelte-plugin';

  export let data;
  export let emit;

  // In Svelte 5 + rete-svelte-plugin, data IS the node (no .payload wrapper)
  $: node = data;
  $: meta = node?.meta || {};
  $: label = node?.label || '?';
  $: color = meta.color || '#888';
  $: isGroup = !!meta.isGroup;
  $: isWall = !!meta.isWall;

  $: inputs = node ? Object.entries(node.inputs || {}) : [];
  $: outputs = node ? Object.entries(node.outputs || {}) : [];

  $: maxPorts = Math.max(inputs.length, outputs.length);
  $: nodeHeight = Math.max(50, 30 + maxPorts * 24);
  $: nodeWidth = isWall ? 120 : (isGroup ? 180 : 160);
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="node"
  class:group-node={isGroup}
  class:wall-node={isWall}
  class:selected={node?.selected}
  data-rete-node-id={node?.id}
  style="
    --node-color: {color};
    width: {nodeWidth}px;
    min-height: {nodeHeight}px;
  "
>
  <div class="label">{label}</div>

  <div class="ports">
    <div class="inputs">
      {#each inputs as [key, input]}
        <div class="port-row input-row">
          <Ref
            class="input-socket"
            init={(el) => emit({ type: 'render', data: { type: 'socket', side: 'input', key, nodeId: node.id, element: el, payload: input.socket } })}
            unmount={(el) => emit({ type: 'unmount', data: { element: el } })}
          />
          <span class="port-label">{input.label || key}</span>
        </div>
      {/each}
    </div>
    <div class="outputs">
      {#each outputs as [key, output]}
        <div class="port-row output-row">
          <span class="port-label">{output.label || key}</span>
          <Ref
            class="output-socket"
            init={(el) => emit({ type: 'render', data: { type: 'socket', side: 'output', key, nodeId: node.id, element: el, payload: output.socket } })}
            unmount={(el) => emit({ type: 'unmount', data: { element: el } })}
          />
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .node {
    background: color-mix(in srgb, var(--node-color) 15%, #1a1a2e);
    border: 1.5px solid var(--node-color);
    border-radius: 8px;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    user-select: none;
    position: relative;
  }
  .node.selected {
    border-width: 2.5px;
    filter: brightness(1.3);
  }
  .node.group-node {
    border-style: dashed;
    border-width: 1.5px;
    background: color-mix(in srgb, var(--node-color) 8%, #1a1a2e);
  }
  .node.wall-node {
    border-style: dotted;
    border-color: #555;
    background: #16213e;
  }
  .label {
    text-align: center;
    padding: 6px 8px 2px;
    font-size: 11px;
    font-weight: 600;
    color: #ddd;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .group-node .label { color: var(--node-color); }
  .wall-node .label { color: #888; font-size: 10px; }

  .ports {
    display: flex;
    justify-content: space-between;
    padding: 4px 0 6px;
  }
  .inputs, .outputs {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .outputs { align-items: flex-end; }

  .port-row {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0 4px;
  }
  .port-label {
    font-size: 9px;
    color: #aaa;
  }
  .output-row .port-label { text-align: right; }
</style>
