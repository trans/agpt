// portShapes.js — SVG port shape helpers

// Determine rank from shape definition
export function portRank(shapeDef) {
  if (!Array.isArray(shapeDef)) return 1;
  const dims = shapeDef.filter(d => d !== null);
  return dims.length;
}

// Returns SVG element attributes for a port shape
// rank: 0=scalar(circle), 1=vector(diamond), 2=matrix(square), 3+=tensor(circle)
export function portShapeAttrs(rank, px, py, r) {
  if (rank === 2) {
    const size = r * 2;
    return {
      tag: 'rect',
      attrs: { x: px - size/2, y: py - size/2, width: size, height: size, rx: 2 },
    };
  } else if (rank === 1) {
    return {
      tag: 'polygon',
      attrs: { points: `${px},${py-r} ${px+r},${py} ${px},${py+r} ${px-r},${py}` },
    };
  } else {
    return {
      tag: 'circle',
      attrs: { cx: px, cy: py, r },
    };
  }
}
