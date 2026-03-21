<script>
  import { onMount } from 'svelte';
  import { registry } from './lib/stores/ui.js';
  import { nodes, edges, groups } from './lib/stores/graph.js';

  let loaded = false;

  onMount(async () => {
    const resp = await fetch('/components.json');
    const data = await resp.json();
    registry.set(data);
    loaded = true;
  });
</script>

<div id="app">
  {#if loaded}
    <div id="left-panel">
      <h3 style="padding: 12px; color: #4a90d9;">microGPT</h3>
      <p style="padding: 0 12px; font-size: 12px; color: #888;">
        Svelte frontend — under construction.
      </p>
      <p style="padding: 0 12px; font-size: 11px; color: #666;">
        Nodes: {$nodes.length} | Edges: {$edges.length} | Groups: {Object.keys($groups).length}
      </p>
    </div>
    <div id="canvas-wrap">
      <p style="color: #555; text-align: center; margin-top: 40vh;">
        GraphCanvas component coming next...
      </p>
    </div>
    <div id="right-panel">
      <h3 style="padding: 12px; color: #4a90d9;">Engines</h3>
    </div>
  {:else}
    <p style="color: #888; text-align: center; margin-top: 40vh;">Loading...</p>
  {/if}
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100vh;
    overflow: hidden;
  }
  #app {
    display: flex;
    height: 100%;
  }
  #left-panel {
    width: 250px;
    background: #16213e;
    border-right: 1px solid #333;
    flex-shrink: 0;
  }
  #canvas-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  #right-panel {
    width: 260px;
    background: #16213e;
    border-left: 1px solid #333;
    flex-shrink: 0;
  }
</style>
