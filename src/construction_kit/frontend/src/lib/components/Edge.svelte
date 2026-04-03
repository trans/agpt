<script>
  import { createEventDispatcher } from 'svelte';

  export let edge;        // { id, from: {nodeId, portId}, to: {nodeId, portId} }
  export let fromPos;     // { x, y } or null
  export let toPos;       // { x, y } or null
  export let color = '#555';
  export let selected = false;

  const dispatch = createEventDispatcher();

  $: visible = fromPos && toPos;
  $: d = visible ? bezierPath(fromPos, toPos) : '';

  function bezierPath(from, to) {
    const dx = Math.abs(to.x - from.x) * 0.5;
    return `M${from.x},${from.y} C${from.x + dx},${from.y} ${to.x - dx},${to.y} ${to.x},${to.y}`;
  }

  function onClick(e) {
    e.stopPropagation();
    dispatch('edgeClick', { edgeId: edge.id });
  }
</script>

{#if visible}
  <!-- Wide invisible hit area for easy clicking -->
  <path
    {d}
    fill="none"
    stroke="transparent"
    stroke-width="12"
    style="cursor: pointer"
    on:click={onClick}
  />
  <!-- Visible edge -->
  <path
    class="edge"
    class:selected
    {d}
    fill="none"
    stroke={selected ? '#4a90d9' : color}
    stroke-width={selected ? 2.5 : 1.5}
    pointer-events="none"
    data-edge-id={edge.id}
  />
{/if}
