<script>
  import { onMount, onDestroy } from 'svelte';
  import { registry, currentGroup } from './lib/stores/ui.js';
  import { nodes, edges, groups } from './lib/stores/graph.js';
  import { createDemoGraph } from './lib/defaults.js';
  import { createEditor } from './lib/rete/editor.js';
  import { syncScopeToRete, setupReteListeners } from './lib/rete/sync.js';
  import Breadcrumbs from './lib/components/Breadcrumbs.svelte';

  let loaded = false;
  let containerEl;
  let editorInstance = null;

  onMount(async () => {
    const resp = await fetch('/components.json');
    const data = await resp.json();
    registry.set(data);
    createDemoGraph();
    loaded = true;
  });

  let lastSyncedGroup = null;

  // When loaded + container ready, create Rete editor
  $: if (loaded && containerEl && !editorInstance) {
    initEditor();
  }

  // Re-sync when currentGroup changes (but not on initial load — initEditor handles that)
  $: if (editorInstance && loaded && lastSyncedGroup !== $currentGroup) {
    syncScope($currentGroup);
  }

  async function initEditor() {
    editorInstance = await createEditor(containerEl);
    setupReteListeners(editorInstance.editor, editorInstance.area, handleDrillIn);
    lastSyncedGroup = $currentGroup;
    await syncScopeToRete(editorInstance.editor, editorInstance.area);
  }

  async function syncScope(group) {
    if (!editorInstance) return;
    lastSyncedGroup = group;
    await syncScopeToRete(editorInstance.editor, editorInstance.area);
  }

  function handleDrillIn(groupPath) {
    currentGroup.set(groupPath);
  }

  onDestroy(() => {
    editorInstance?.destroy();
  });
</script>

<div id="app">
  {#if loaded}
    <div id="left-panel">
      <div class="panel-header">
        <h3>microGPT</h3>
      </div>
      <div class="panel-body">
        <p class="info">
          {$nodes.length} nodes &middot; {$edges.length} edges &middot; {Object.keys($groups).length} groups
        </p>
      </div>
    </div>

    <div id="canvas-wrap">
      <Breadcrumbs />
      <div class="rete-container" bind:this={containerEl}></div>
    </div>

    <div id="right-panel">
      <div class="panel-header">
        <h3>Engines</h3>
      </div>
      <div class="panel-body">
        <p class="info">Card management coming soon.</p>
      </div>
    </div>
  {:else}
    <div class="loading">Loading...</div>
  {/if}
</div>

<style>
  :global(html, body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100%;
    overflow: hidden;
  }
  :global(*) { box-sizing: border-box; }

  #app {
    display: flex;
    height: 100vh;
  }
  #left-panel, #right-panel {
    width: 260px;
    background: #16213e;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #left-panel { border-right: 1px solid #333; }
  #right-panel { border-left: 1px solid #333; }
  #canvas-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .rete-container {
    flex: 1;
    width: 100%;
    height: 100%;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid #333;
  }
  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    color: #4a90d9;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .panel-body {
    padding: 12px 16px;
    flex: 1;
    overflow-y: auto;
  }
  .info {
    font-size: 11px;
    color: #666;
  }
  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #666;
  }
</style>
