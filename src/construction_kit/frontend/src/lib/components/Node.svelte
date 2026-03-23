<script>
  import { createEventDispatcher } from 'svelte';
  import { registry } from '../stores/ui.js';
  import { getCompDef, getDataTypeColor } from '../stores/ui.js';
  import Port from './Port.svelte';

  export let node;        // { id, type, group, params, x, y }
  export let selected = false;

  const dispatch = createEventDispatcher();

  $: comp = getCompDef(node.type, $registry);
  $: color = comp?.color || '#888';
  $: label = comp?.label || node.type;
  $: inPorts = comp?.ports?.in || [];
  $: outPorts = comp?.ports?.out || [];
  $: maxPorts = Math.max(inPorts.length, outPorts.length);
  $: w = 140;
  $: h = Math.max(50, 26 + maxPorts * 14);

  function onMouseDown(e) {
    dispatch('nodeMouseDown', { nodeId: node.id, event: e });
  }

  function onDblClick(e) {
    dispatch('nodeDblClick', { nodeId: node.id, event: e });
  }

  function onPortMouseDown(e, portDef, isOutput) {
    e.stopPropagation();
    dispatch('portMouseDown', { nodeId: node.id, portId: portDef.id, isOutput, event: e });
  }
</script>

<g
  class="node"
  class:selected
  transform="translate({node.x}, {node.y})"
  on:mousedown={onMouseDown}
  on:dblclick={onDblClick}
  style="cursor: grab"
>
  <!-- Background -->
  <rect
    width={w} height={h} rx="6"
    fill="{color}22"
    stroke={color}
    stroke-width={selected ? 2.5 : 1.5}
  />

  <!-- Label -->
  <text
    x={w / 2} y="18"
    text-anchor="middle"
    fill="#ddd"
    font-size="11"
    font-weight="600"
    pointer-events="none"
  >{label}</text>

  <!-- Input ports -->
  {#each inPorts as p, i}
    {@const py = 30 + i * 14}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    <Port
      portDef={{ ...p, _nodeId: node.id }}
      isOutput={false}
      px={0}
      py={py}
      color={dtColor}
      on:mousedown={(e) => onPortMouseDown(e.detail?.event || e, p, false)}
    />
  {/each}

  <!-- Output ports -->
  {#each outPorts as p, i}
    {@const py = 30 + i * 14}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    <Port
      portDef={{ ...p, _nodeId: node.id }}
      isOutput={true}
      px={w}
      py={py}
      color={dtColor}
      on:mousedown={(e) => onPortMouseDown(e.detail?.event || e, p, true)}
    />
  {/each}
</g>

<style>
  .node { transition: opacity 0.1s; }
  .node:hover rect { filter: brightness(1.2); }
  .node.selected rect { filter: brightness(1.3); }
</style>
