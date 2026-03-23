<script>
  import { portRank, portShapeAttrs } from '../utils/portShapes.js';

  export let portDef;     // { id, label, dataType, shape, multi }
  export let isOutput;    // boolean
  export let px;          // x position
  export let py;          // y position
  export let color;       // data type color
  export let r = 6;       // port radius

  let showTooltip = false;

  $: rank = portRank(portDef.shape);
  $: shape = portShapeAttrs(rank, px, py, r);
  $: fill = isOutput ? color : 'transparent';
  $: strokeWidth = isOutput ? 2 : 2.5;
  $: isMulti = !!portDef.multi;
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<g
  class="port"
  class:port-in={!isOutput}
  class:port-out={isOutput}
  style="cursor: crosshair"
  data-node-id={portDef._nodeId}
  data-port-id={portDef.id}
  data-is-output={isOutput}
  on:mouseenter={() => showTooltip = true}
  on:mouseleave={() => showTooltip = false}
  on:mousedown
>
  <!-- Invisible hit area -->
  <circle cx={px} cy={py} r={r + 5} fill="transparent" stroke="none" />

  <!-- Visible shape -->
  {#if shape.tag === 'rect'}
    <rect {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {:else if shape.tag === 'polygon'}
    <polygon {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {:else}
    <circle {...shape.attrs} {fill} stroke={color} stroke-width={strokeWidth} pointer-events="none" />
  {/if}

  <!-- Multi-port indicator (stacked marks) -->
  {#if isMulti}
    <line x1={px - r - 2} y1={py - r - 3} x2={px + r + 2} y2={py - r - 3}
          stroke={color} stroke-width="1.5" opacity="0.5" pointer-events="none" />
    <line x1={px - r - 2} y1={py - r - 6} x2={px + r + 2} y2={py - r - 6}
          stroke={color} stroke-width="1.5" opacity="0.3" pointer-events="none" />
  {/if}

  <!-- Label -->
  {#if isOutput}
    <text x={px - r - 4} y={py + 3} text-anchor="end" fill={color} font-size="8" opacity="0.7" pointer-events="none">
      {portDef.label || portDef.id}
    </text>
  {:else}
    <text x={px + r + 4} y={py + 3} fill={color} font-size="8" opacity="0.7" pointer-events="none">
      {portDef.label || portDef.id}
    </text>
  {/if}
</g>

<!-- Tooltip -->
{#if showTooltip}
  {@const tx = isOutput ? px - 150 : px + 15}
  {@const ty = py - 20}
  <g class="port-tooltip-g" pointer-events="none">
    <rect x={tx} y={ty} width="140" height="32" rx="4" fill="#1a1a2eee" stroke="#444" />
    <text x={tx + 4} y={ty + 12} fill="#ddd" font-size="10" font-weight="600">
      {portDef.id}
      <tspan fill={color}> {portDef.dataType || ''}</tspan>
    </text>
    <text x={tx + 4} y={ty + 24} fill="#888" font-size="9">
      {isOutput ? 'output' : 'input'}{isMulti ? ' (multi)' : ''}
    </text>
  </g>
{/if}
