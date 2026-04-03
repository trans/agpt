<script>
  export let data;
  // data.payload is the ClassicPreset.Socket instance

  const colorMap = {
    token_ids: '#7bc67e',
    stream: '#4a90d9',
    matrix: '#b8a038',
    logits: '#e6994a',
    vector: '#c49a38',
    scalar: '#d9d94a',
    raw_text: '#d9d9d9',
    tensor: '#a08828',
  };

  // data IS the socket (no .payload wrapper in Svelte 5 compat)
  $: socketName = data?.name || 'matrix';
  $: color = colorMap[socketName] || '#888';
</script>

<div
  class="socket"
  title={socketName}
  style="background: {color}; border-color: {color};"
></div>

<style>
  .socket {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid;
    box-sizing: border-box;
    cursor: crosshair;
  }
</style>
