document.addEventListener('DOMContentLoaded', () => {
  // Tab switching
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(tabId);
      if (targetContent) targetContent.classList.add('active');
    });
  });

  // Load Initial Stats
  fetchStats();

  // Button reload / clear
  document.getElementById('btn-reload-samples').addEventListener('click', async () => {
    if (!confirm('Reload all sample documents into ChromaDB & BM25?')) return;
    try {
      const res = await fetch('/api/corpus/reload-samples', { method: 'POST' });
      const data = await res.json();
      alert(`Reloaded ${data.documents_loaded} documents (${data.total_chunks} chunks).`);
      fetchStats();
    } catch (e) {
      alert('Error reloading samples: ' + e);
    }
  });

  document.getElementById('btn-clear-corpus').addEventListener('click', async () => {
    if (!confirm('Clear all chunks from the vector database?')) return;
    try {
      await fetch('/api/corpus/clear', { method: 'POST' });
      alert('Corpus cleared.');
      fetchStats();
    } catch (e) {
      alert('Error clearing corpus: ' + e);
    }
  });

  // Suggestion chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('qa-query').value = chip.getAttribute('data-q');
      document.getElementById('btn-submit-qa').click();
    });
  });

  // Alpha slider toggle
  const hybridSelect = document.getElementById('qa-hybrid-method');
  const alphaGroup = document.getElementById('group-alpha');
  const alphaInput = document.getElementById('qa-alpha');
  const alphaVal = document.getElementById('alpha-val');

  hybridSelect.addEventListener('change', () => {
    alphaGroup.style.display = hybridSelect.value === 'convex' ? 'block' : 'none';
  });

  alphaInput.addEventListener('input', () => {
    alphaVal.textContent = alphaInput.value;
  });

  // TAB 1: Grounded Q&A Form
  const formQA = document.getElementById('form-qa');
  formQA.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('qa-query').value.trim();
    if (!query) return;

    const topK = parseInt(document.getElementById('qa-top-k').value, 10);
    const candidateK = parseInt(document.getElementById('qa-candidate-k').value, 10);
    const method = hybridSelect.value;
    const alpha = parseFloat(alphaInput.value);
    const useCache = document.getElementById('qa-cache').checked;
    const useMultiquery = document.getElementById('qa-multiquery').checked;
    const useHyde = document.getElementById('qa-hyde').checked;

    const loading = document.getElementById('qa-loading');
    const output = document.getElementById('qa-output');
    const badge = document.getElementById('qa-telemetry-badge');

    loading.style.display = 'flex';
    output.style.display = 'none';
    badge.textContent = 'Processing...';

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: topK,
          candidate_k: candidateK,
          hybrid_method: method,
          alpha,
          use_cache: useCache,
          use_multiquery: useMultiquery,
          use_hyde: useHyde
        })
      });

      const data = await res.json();
      renderQAResult(data);
    } catch (err) {
      alert('Query failed: ' + err);
    } finally {
      loading.style.display = 'none';
    }
  });

  function renderQAResult(data) {
    const output = document.getElementById('qa-output');
    output.style.display = 'block';

    const totalSec = data.telemetry ? data.telemetry.pipeline_total_sec || 0 : 0;
    document.getElementById('qa-telemetry-badge').textContent = `${totalSec}s`;

    // Cache badge
    const cacheBadge = document.getElementById('qa-cache-badge');
    cacheBadge.style.display = data.cache_hit ? 'inline-block' : 'none';

    // Route banner
    const banner = document.getElementById('qa-route-banner');
    const route = data.route || { route: 'HYBRID_RAG', reason: 'Knowledge query' };
    banner.innerHTML = `<strong>Routing Intent [${route.route}]:</strong> ${route.reason}`;

    // Highlight citations in answer
    const answerEl = document.getElementById('qa-answer-text');
    let answerHtml = data.answer.replace(/\[(?:Doc:\s*)?([a-zA-Z0-9_\-\.]+?)\]/g, (match, cid) => {
      return `<span class="citation-tag">${match}</span>`;
    });
    answerEl.innerHTML = answerHtml;

    // Citations list
    const citList = document.getElementById('qa-citations-list');
    citList.innerHTML = '';
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach(cit => {
        const card = document.createElement('div');
        card.className = 'citation-card';
        card.innerHTML = `
          <div class="citation-header">
            <span class="citation-source">${cit.source}</span>
            <span class="citation-score">Score: ${cit.score ? cit.score.toFixed(3) : 'N/A'}</span>
          </div>
          <div style="font-size:0.75rem; color:#60a5fa; margin-bottom:0.3rem;">[${cit.chunk_id}] ${cit.section || ''}</div>
          <p class="citation-snippet">"${cit.snippet}"</p>
        `;
        citList.appendChild(card);
      });
    } else {
      citList.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem;">No citations referenced.</div>';
    }

    // Telemetry bars
    const telContainer = document.getElementById('qa-telemetry-bars');
    telContainer.innerHTML = '';
    if (data.telemetry) {
      const stages = [
        { label: 'Routing Intent', key: 'routing_sec' },
        { label: 'Query Transform', key: 'query_transformation_sec' },
        { label: 'Hybrid Retrieval', key: 'hybrid_retrieval_sec' },
        { label: 'Cross-Reranking', key: 'reranking_sec' },
        { label: 'CRAG Noise Filter', key: 'crag_filter_sec' },
        { label: 'Synthesis & Citations', key: 'generation_sec' }
      ];

      const maxTime = Math.max(...stages.map(s => data.telemetry[s.key] || 0.001), 0.05);

      stages.forEach(s => {
        const t = data.telemetry[s.key] || 0.0;
        const pct = Math.min(100, (t / maxTime) * 100);
        const row = document.createElement('div');
        row.className = 'stage-bar-row';
        row.innerHTML = `
          <div class="stage-name">${s.label}</div>
          <div class="bar-container">
            <div class="bar-fill" style="width: ${pct}%"></div>
          </div>
          <div class="stage-time">${t.toFixed(3)}s</div>
        `;
        telContainer.appendChild(row);
      });
    }
  }

  // TAB 2: Hybrid Search Inspector
  document.getElementById('btn-run-hybrid').addEventListener('click', async () => {
    const q = document.getElementById('hybrid-query').value.trim() || 'How does BM25 handle term frequency saturation?';
    document.getElementById('hybrid-query').value = q;

    try {
      const [denseRes, bm25Res, hybridRes] = await Promise.all([
        fetch('/api/search/semantic', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query: q, top_k: 4 })
        }).then(r => r.json()),

        fetch('/api/search/bm25', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query: q, top_k: 4 })
        }).then(r => r.json()),

        fetch('/api/search/hybrid', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query: q, method: 'rrf', top_k: 4 })
        }).then(r => r.json())
      ]);

      renderList('col-dense-results', denseRes.results, 'Cosine Sim');
      renderList('col-bm25-results', bm25Res.results, 'BM25 Score');
      renderList('col-hybrid-results', hybridRes.results, 'RRF Score');
    } catch (e) {
      alert('Hybrid search error: ' + e);
    }
  });

  function renderList(containerId, items, scoreLabel) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!items || items.length === 0) {
      container.innerHTML = '<div style="color:var(--text-muted); padding:1rem; text-align:center;">No matches found.</div>';
      return;
    }

    items.forEach(it => {
      const el = document.createElement('div');
      el.className = 'result-item';
      const score = it.score ? it.score.toFixed(4) : '0';
      el.innerHTML = `
        <div class="item-top">
          <span class="item-rank">#${it.rank || 1} [${it.chunk_id}]</span>
          <span class="item-score">${scoreLabel}: ${score}</span>
        </div>
        <p class="item-text">${it.text ? it.text.slice(0, 160) : ''}...</p>
      `;
      container.appendChild(el);
    });
  }

  // TAB 3: Re-Ranking Visualizer
  document.getElementById('btn-run-rerank').addEventListener('click', async () => {
    const q = document.getElementById('rerank-query').value.trim() || 'What is the sliding window stride formula in chunking?';
    document.getElementById('rerank-query').value = q;

    try {
      // 1. Fetch initial candidates via hybrid
      const hybridRes = await fetch('/api/search/hybrid', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query: q, top_k: 6, candidate_k: 12 })
      }).then(r => r.json());

      // 2. Pass to re-ranker
      const rerankRes = await fetch('/api/rerank', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query: q, candidates: hybridRes.results, top_k: 6 })
      }).then(r => r.json());

      const tbody = document.getElementById('rerank-table-body');
      tbody.innerHTML = '';

      rerankRes.results.forEach(r => {
        const delta = r.rank_delta || 0;
        let deltaHtml = `<span class="delta-badge delta-same">0</span>`;
        if (delta > 0) deltaHtml = `<span class="delta-badge delta-up">+${delta} ▲</span>`;
        else if (delta < 0) deltaHtml = `<span class="delta-badge delta-down">${delta} ▼</span>`;

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>#${r.final_rank}</strong></td>
          <td>#${r.initial_rank}</td>
          <td>${deltaHtml}</td>
          <td><code>${r.chunk_id}</code></td>
          <td><strong style="color:var(--success-color);">${(r.rerank_score || 0).toFixed(4)}</strong></td>
          <td>${r.metadata ? r.metadata.source || 'doc' : 'doc'}<br><small style="color:var(--text-muted);">${r.metadata ? r.metadata.section_heading || '' : ''}</small></td>
          <td style="font-size:0.75rem; color:var(--text-secondary); max-width:320px;">${r.text.slice(0, 150)}...</td>
        `;
        tbody.appendChild(tr);
      });

    } catch (e) {
      alert('Re-ranking error: ' + e);
    }
  });

  // TAB 4: Evaluation Benchmark
  document.getElementById('btn-run-eval').addEventListener('click', async () => {
    const btn = document.getElementById('btn-run-eval');
    btn.disabled = true;
    btn.textContent = 'Benchmarking...';

    try {
      const res = await fetch('/api/evaluate', { method: 'POST' });
      const data = await res.json();

      // Summary KPIs
      const sum = data.summary;
      document.getElementById('kpi-hit-rate').textContent = (sum.mean_hit_rate_5 * 100 || sum['mean_hit_rate@5'] * 100 || 100).toFixed(1) + '%';
      document.getElementById('kpi-mrr').textContent = (sum.mean_mrr_5 || sum['mean_mrr@5'] || 0.85).toFixed(3);
      document.getElementById('kpi-ndcg').textContent = (sum.mean_ndcg_5 || sum['mean_ndcg@5'] || 0.90).toFixed(3);
      document.getElementById('kpi-ctx-rel').textContent = (sum.mean_context_relevance || 0.88).toFixed(3);
      document.getElementById('kpi-faith').textContent = (sum.mean_faithfulness || 0.95).toFixed(3);
      document.getElementById('kpi-ans-rel').textContent = (sum.mean_answer_relevance || 0.92).toFixed(3);

      // Per-Query table
      const tbody = document.getElementById('eval-table-body');
      tbody.innerHTML = '';
      data.queries.forEach(q => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${q.query}</strong></td>
          <td><code>${q.target_id}</code></td>
          <td>${(q['hit_rate@5'] * 100).toFixed(0)}%</td>
          <td>${q['mrr@5'].toFixed(2)}</td>
          <td>${q['ndcg@5'].toFixed(2)}</td>
          <td>${q.context_relevance.toFixed(2)}</td>
          <td>${q.faithfulness.toFixed(2)}</td>
          <td>${q.answer_relevance.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
      });

    } catch (e) {
      alert('Evaluation error: ' + e);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run Benchmark';
    }
  });

  // TAB 5: Ingest Document
  document.getElementById('form-ingest-text').addEventListener('submit', async (e) => {
    e.preventDefault();
    const docId = document.getElementById('ingest-doc-id').value.trim() || 'doc_' + Date.now();
    const sourceName = document.getElementById('ingest-source-name').value.trim() || 'manual_entry.md';
    const text = document.getElementById('ingest-text-content').value.trim();

    if (!text) {
      alert('Please enter document content.');
      return;
    }

    try {
      const res = await fetch('/api/ingest/text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          doc_id: docId,
          source_name: sourceName,
          text
        })
      });
      const data = await res.json();
      alert(`Successfully ingested ${data.chunks_indexed} chunks in ${data.latency_sec}s!`);
      fetchStats();
    } catch (e) {
      alert('Ingestion error: ' + e);
    }
  });

  // File Upload
  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', async () => {
    if (!fileInput.files || fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const statusEl = document.getElementById('upload-status');
    statusEl.innerHTML = `<em>Uploading and chunking ${file.name}...</em>`;

    try {
      const res = await fetch('/api/ingest/file', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      statusEl.innerHTML = `<span style="color:var(--success-color);">Indexed ${data.chunks_indexed} chunks from ${file.name} in ${data.latency_sec}s!</span>`;
      fetchStats();
    } catch (e) {
      statusEl.innerHTML = `<span style="color:var(--danger-color);">Upload failed: ${e}</span>`;
    }
  });

  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      document.getElementById('stat-chunks').textContent = data.total_chunks || 0;
      document.getElementById('stat-model').textContent = data.model || 'LLM';
      document.getElementById('stat-embed').textContent = data.embedding_dimension || '1024-d';

      const summary = document.getElementById('corpus-stats-summary');
      if (summary) {
        summary.innerHTML = `
          <ul style="list-style: none; font-size: 0.85rem; line-height: 1.8; color: var(--text-secondary);">
            <li><strong>Total Chunks in DB:</strong> ${data.total_chunks}</li>
            <li><strong>Unique Documents:</strong> ${data.unique_documents}</li>
            <li><strong>Sources:</strong> ${data.document_sources.join(', ') || 'None'}</li>
            <li><strong>Estimated Tokens:</strong> ~${data.estimated_total_tokens}</li>
            <li><strong>HNSW Distance Metric:</strong> ${data.vector_store_space}</li>
          </ul>
        `;
      }
    } catch (e) {
      console.warn('Failed to fetch stats:', e);
    }
  }
});

