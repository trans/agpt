<script>
  import { createEventDispatcher } from 'svelte';
  import { registry } from '../stores/ui.js';
  import { getCompDef, getDataTypeColor } from '../stores/ui.js';
  import { edges } from '../stores/graph.js';
  import { portRank, portShapeAttrs } from '../utils/portShapes.js';

  export let groupPath;
  export let groupInfo;
  export let x = 0;
  export let y = 0;
  export let w = 160;
  export let h = 70;

  const dispatch = createEventDispatcher();

  $: comp = getCompDef(groupInfo?.type, $registry);
  $: color = comp?.color || '#666';
  $: label = groupInfo?.label || groupPath.split('.').pop();

  // Read declared ports from group metadata
  $: portsIn = groupInfo?.ports?.in || [];
  $: portsOut = groupInfo?.ports?.out || [];

  // Check which ports have external connections (for dimming)
  $: gRef = "group:" + groupPath;
  $: connectedPorts = computeConnectedPorts($edges, portsIn, portsOut, gRef);
  $: autoH = Math.max(h, 24 + Math.max(portsIn.length, portsOut.length) * 16);

  function computeConnectedPorts(allEdges, inputs, outputs, groupRef) {
    const connected = new Set();
    for (const p of inputs) {
      if (allEdges.some(e => e.to.nodeId === groupRef && e.to.portId === p.id))
        connected.add(`in:${p.id}`);
    }
    for (const p of outputs) {
      if (allEdges.some(e => e.from.nodeId === groupRef && e.from.portId === p.id))
        connected.add(`out:${p.id}`);
    }
    return connected;
  }

  function onDblClick(e) {
    e.stopPropagation();
    dispatch('drillIn', { groupPath });
  }

  function onMouseDown(e) {
    e.stopPropagation();
    dispatch('groupMouseDown', { groupPath, event: e });
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<g
  class="group-box"
  transform="translate({x}, {y})"
  style="cursor: pointer"
  on:dblclick={onDblClick}
  on:mousedown={onMouseDown}
>
  <rect
    width={w} height={autoH} rx="8"
    fill="{color}11"
    stroke={color}
    stroke-width="1.5"
    stroke-dasharray="6,3"
  />
  <text
    x={w / 2} y="16"
    text-anchor="middle"
    fill={color}
    font-size="11"
    font-weight="600"
    pointer-events="none"
  >{label}</text>

  <!-- Declared input ports (left side) -->
  {#each portsIn as p, i}
    {@const py = 30 + i * 16}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    {@const rank = portRank(p.shape)}
    {@const ps = portShapeAttrs(rank, 0, py, 5)}
    {@const isConnected = connectedPorts.has(`in:${p.id}`)}
    <g class="port port-in"
       data-node-id={"group:" + groupPath}
       data-port-id={p.id}
       data-is-output="false"
       style="cursor: crosshair"
    >
      <circle cx={0} cy={py} r={10} fill="transparent" stroke="none" />
      {#if ps.tag === 'rect'}
        <rect {...ps.attrs} fill="transparent" stroke={dtColor}
              stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {:else if ps.tag === 'polygon'}
        <polygon {...ps.attrs} fill="transparent" stroke={dtColor}
                 stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {:else}
        <circle {...ps.attrs} fill="transparent" stroke={dtColor}
                stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {/if}
      <text x="10" y={py + 3} fill={dtColor} font-size="7"
            opacity={isConnected ? 0.7 : 0.3} pointer-events="none">
        {p.label || p.id}
      </text>
    </g>
  {/each}

  <!-- Declared output ports (right side) -->
  {#each portsOut as p, i}
    {@const py = 30 + i * 16}
    {@const dtColor = getDataTypeColor(p.dataType, $registry)}
    {@const rank = portRank(p.shape)}
    {@const ps = portShapeAttrs(rank, w, py, 5)}
    {@const isConnected = connectedPorts.has(`out:${p.id}`)}
    <g class="port port-out"
       data-node-id={"group:" + groupPath}
       data-port-id={p.id}
       data-is-output="true"
       style="cursor: crosshair"
    >
      <circle cx={w} cy={py} r={10} fill="transparent" stroke="none" />
      {#if ps.tag === 'rect'}
        <rect {...ps.attrs} fill={dtColor} stroke={dtColor}
              stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {:else if ps.tag === 'polygon'}
        <polygon {...ps.attrs} fill={dtColor} stroke={dtColor}
                 stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {:else}
        <circle {...ps.attrs} fill={dtColor} stroke={dtColor}
                stroke-width="2" opacity={isConnected ? 1 : 0.4} pointer-events="none" />
      {/if}
      <text x={w - 10} y={py + 3} text-anchor="end" fill={dtColor} font-size="7"
            opacity={isConnected ? 0.7 : 0.3} pointer-events="none">
        {p.label || p.id}
      </text>
    </g>
  {/each}
</g>
