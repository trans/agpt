// defaults.js — Default graph generators for standard architectures
// Creates flat nodes + edges with group tags.

import { addNode, addEdge, addGroup, updateGroup, clearGraph, newId } from './stores/graph.js';

function _node(type, group, params = {}, x = 0, y = 0) {
  return addNode(type, group, params, x, y);
}

function _edge(fromNode, fromPort, toNode, toPort) {
  return addEdge(fromNode.id, fromPort, toNode.id, toPort);
}

export function createDemoGraph() {
  clearGraph();
  const sd = 64;

  // Pipeline-level data nodes
  const tok = _node('char_tokenizer', '', {}, -200, 100);
  const win = _node('sequential_window', '', { seq_len: 128 }, 50, 100);
  const init = _node('zero_init', '', {}, 50, -60);
  const loss = _node('loss', '', {}, 900, 100);
  const opt = _node('optimizer', '', {
    algorithm: 'adam', learning_rate: 0.0003,
    beta1: 0.9, beta2: 0.999, weight_decay: 0.01,
  }, 900, -60);

  // Cooperative group — declare ports up front
  addGroup('coop', 'Cooperative Ensemble', 'cooperative', { stream_dim: sd }, {
    in: [
      { id: "ids", dataType: "token_ids", shape: [null], label: "tokens" },
      { id: "stream_in", dataType: "stream", shape: [null, "stream_dim"], label: "stream" },
    ],
    out: [
      { id: "logits_out", dataType: "logits", shape: [null, "vocab_size"], label: "logits" },
    ],
  });

  // Two transformer experts
  const expertA = createTransformerExpert('coop.xfmr_a', 'Transformer A', sd, 3, 100, 40);
  const expertB = createTransformerExpert('coop.xfmr_b', 'Transformer B', sd, 3, 100, 600);
  const router = _node('global_router', 'coop', { epsilon: 0.2 }, 500, 300);

  // Coop portMap — routes to transformer sub-group ports (not raw internal nodes)
  updateGroup('coop', {
    portMap: {
      "ids": [
        { nodeId: expertA.gRef, portId: "ids" },
        { nodeId: expertB.gRef, portId: "ids" },
      ],
      "stream_in": [
        { nodeId: expertA.gRef, portId: "stream_in" },
        { nodeId: router.id, portId: "stream_in" },
      ],
      "logits_out": [
        { nodeId: router.id, portId: "logits_out" },
      ],
    }
  });

  // Pipeline wiring — external edges target group ports
  _edge(tok, 'token_ids', win, 'token_ids');
  addEdge(win.id, 'input_ids', 'group:coop', 'ids');
  addEdge(init.id, 'stream_out', 'group:coop', 'stream_in');

  // Internal coop wiring — through transformer group ports
  addEdge(expertA.gRef, 'stream_out', expertB.gRef, 'stream_in');
  addEdge(expertA.gRef, 'logits_out', router.id, 'logits_in');
  addEdge(expertB.gRef, 'logits_out', router.id, 'logits_in');

  // Output — from group port to pipeline
  addEdge('group:coop', 'logits_out', loss.id, 'logits_in');
  _edge(win, 'target_ids', loss, 'targets');
}

function createTransformerExpert(groupPath, label, d, nLayers, baseX, baseY) {
  // Declare ports for the transformer group
  addGroup(groupPath, label, 'transformer', { d, n_layers: nLayers }, {
    in: [
      { id: "ids", dataType: "token_ids", shape: [null], label: "tokens" },
      { id: "stream_in", dataType: "stream", shape: [null, "stream_dim"], label: "stream" },
    ],
    out: [
      { id: "logits_out", dataType: "logits", shape: [null, "vocab_size"], label: "logits" },
      { id: "stream_out", dataType: "stream", shape: [null, "stream_dim"], label: "stream" },
    ],
  });

  const gRef = "group:" + groupPath;
  const xStep = 180, yStep = 160;
  let row = 0;

  // Embedding
  addGroup(`${groupPath}.emb`, 'Embedding', 'embedding', {});
  const embTable = _node('embedding_table', `${groupPath}.emb`, { vocab_size: 65 }, baseX, baseY + row * yStep);
  const lookup = _node('lookup', `${groupPath}.emb`, {}, baseX + xStep, baseY + row * yStep);
  _edge(embTable, 'out', lookup, 'table');

  // Stream read
  const streamRead = _node('stream_proj_internal', groupPath, { d_in: d, d_out: d }, baseX + xStep * 2, baseY + row * yStep);

  // Add: embedding + stream
  const embedAdd = _node('add', groupPath, {}, baseX + xStep * 3, baseY + row * yStep);
  _edge(lookup, 'out', embedAdd, 'a');
  _edge(streamRead, 'out', embedAdd, 'b');

  row++;
  let lastNode = embedAdd, lastPort = 'out';

  // Layers
  for (let i = 0; i < nLayers; i++) {
    const lg = `${groupPath}.layer_${i}`;
    addGroup(lg, `Layer ${i}`, 'layer', {});
    const y = baseY + row * yStep;

    const lnA = _node('layer_norm', lg, {}, baseX, y);
    const att = _node('attention_layer', lg, { n_heads: Math.max(1, Math.floor(d / 16)) }, baseX + xStep, y);
    const lnF = _node('layer_norm', lg, {}, baseX + xStep * 2, y);
    const ffn = _node('ffn_layer', lg, { ff_mult: 4 }, baseX + xStep * 3, y);

    _edge(lnA, 'out', att, 'in');
    _edge(att, 'out', lnF, 'in');
    _edge(lnF, 'out', ffn, 'in');
    _edge(lastNode, lastPort, lnA, 'in');

    lastNode = ffn; lastPort = 'out';
    row++;
  }

  // Output head
  addGroup(`${groupPath}.out`, 'Output Head', 'output_head', {});
  const y = baseY + row * yStep;
  const outW = _node('weight_param', `${groupPath}.out`, {}, baseX, y);
  const outMM = _node('matmul', `${groupPath}.out`, {}, baseX + xStep, y);
  const outB = _node('bias_param', `${groupPath}.out`, {}, baseX, y + 80);
  const outAdd = _node('add_bias', `${groupPath}.out`, {}, baseX + xStep, y + 80);
  _edge(lastNode, lastPort, outMM, 'x');
  _edge(outW, 'out', outMM, 'W');
  _edge(outMM, 'out', outAdd, 'x');
  _edge(outB, 'out', outAdd, 'b');

  // Stream write
  const streamWrite = _node('stream_proj_internal', groupPath, { d_in: d, d_out: d }, baseX + xStep * 2, y);
  _edge(lastNode, lastPort, streamWrite, 'in');

  // Set portMap — maps declared ports to internal nodes
  updateGroup(groupPath, {
    portMap: {
      "ids": [{ nodeId: lookup.id, portId: "ids" }],
      "stream_in": [{ nodeId: streamRead.id, portId: "in" }],
      "logits_out": [{ nodeId: outAdd.id, portId: "out" }],
      "stream_out": [{ nodeId: streamWrite.id, portId: "out" }],
    }
  });

  return { gRef, lookup, streamRead, streamWrite, outputLast: outAdd };
}
