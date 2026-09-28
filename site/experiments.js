(() => {
  const list = document.querySelector('#experiment-list');
  if (!list) return;

  const rows = [...list.querySelectorAll('.experiment-row')];
  const filters = [...document.querySelectorAll('[data-filter]')];
  const search = document.querySelector('#experiment-search');
  const sort = document.querySelector('#date-sort');
  const count = document.querySelector('#result-count');
  const empty = document.querySelector('#no-results');

  function update() {
    const query = search.value.trim().toLowerCase();
    const selected = Object.fromEntries(filters.map(input => [input.dataset.filter, input.value]));
    let visible = 0;

    for (const row of rows) {
      const matches = (!query || row.dataset.search.includes(query))
        && Object.entries(selected).every(([key, value]) => !value
          || (key === 'tag' ? row.dataset.tags.split('|').includes(value) : row.dataset[key] === value));
      row.hidden = !matches;
      if (matches) visible++;
    }

    rows.sort((a, b) => {
      const cmp = a.dataset.updated.localeCompare(b.dataset.updated)
        || a.querySelector('h2').textContent.localeCompare(b.querySelector('h2').textContent);
      return sort.value === 'oldest' ? cmp : -cmp;
    });
    list.append(...rows);
    count.textContent = `${visible} of ${rows.length} experiments`;
    empty.hidden = visible !== 0;

    const url = new URL(location.href);
    for (const [key, value] of Object.entries(selected)) {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    if (query) url.searchParams.set('q', search.value.trim());
    else url.searchParams.delete('q');
    if (sort.value === 'oldest') url.searchParams.set('sort', 'oldest');
    else url.searchParams.delete('sort');
    history.replaceState(null, '', url);
  }

  function clear() {
    search.value = '';
    filters.forEach(input => { input.value = ''; });
    sort.value = 'newest';
    update();
  }

  const params = new URLSearchParams(location.search);
  search.value = params.get('q') || '';
  sort.value = params.get('sort') === 'oldest' ? 'oldest' : 'newest';
  filters.forEach(input => {
    const value = params.get(input.dataset.filter) || '';
    input.value = [...input.options].some(option => option.value === value) ? value : '';
    input.addEventListener('change', update);
  });
  search.addEventListener('input', update);
  sort.addEventListener('change', update);
  document.querySelector('#clear-filters').addEventListener('click', clear);
  document.querySelector('[data-clear]').addEventListener('click', clear);
  update();
})();
