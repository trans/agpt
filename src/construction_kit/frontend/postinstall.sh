#!/bin/sh
# Patch rete-svelte-plugin to use Svelte 5 compat layer
RENDERER="node_modules/rete-svelte-plugin/svelte/renderer.js"
if [ -f "$RENDERER" ]; then
  sed -i "s|./compat/svelte3-4|./compat/svelte5.svelte.js|" "$RENDERER"
  echo "Patched rete-svelte-plugin for Svelte 5"
fi
