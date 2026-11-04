document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneDefault = document.getElementById("dropzoneDefault");
  const fileSelectedInfo = document.getElementById("fileSelectedInfo");
  const selectedFileName = document.getElementById("selectedFileName");
  const clearFileBtn = document.getElementById("clearFileBtn");
  const uploadBtn = document.getElementById("uploadBtn");
  const refreshBtn = document.getElementById("refreshBtn");
  const clearCorpusBtn = document.getElementById("clearCorpusBtn");
  const corpusBadge = document.getElementById("corpusBadge");
  const indexedStats = document.getElementById("indexedStats");
  const documentsList = document.getElementById("documentsList");

  const retrieveInput = document.getElementById("retrieveInput");
  const retrieveTopK = document.getElementById("retrieveTopK");
  const runRetrievalBtn = document.getElementById("runRetrievalBtn");
  const retrieveOutput = document.getElementById("retrieveOutput");
  const retrieveCount = document.getElementById("retrieveCount");
  const retrieveResultsList = document.getElementById("retrieveResultsList");

  const askInput = document.getElementById("askInput");
  const askTopK = document.getElementById("askTopK");
  const askRagBtn = document.getElementById("askRagBtn");
  const askOutput = document.getElementById("askOutput");
  const answerText = document.getElementById("answerText");
  const answerModelBadge = document.getElementById("answerModelBadge");
  const sourcesToggle = document.getElementById("sourcesToggle");
  const sourcesCount = document.getElementById("sourcesCount");
  const sourcesContent = document.getElementById("sourcesContent");

  const modelBadge = document.getElementById("modelBadge");
  const modelDropdown = document.getElementById("modelDropdown");
  const activeModelName = document.getElementById("activeModelName");
  const toastContainer = document.getElementById("toastContainer");

  let currentSelectedFile = null;

  // Toast Notification
  function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  // Model Selection
  modelBadge.addEventListener("click", (e) => {
    e.stopPropagation();
    modelBadge.classList.toggle("open");
    modelDropdown.classList.toggle("show");
  });

  document.addEventListener("click", () => {
    modelBadge.classList.remove("open");
    modelDropdown.classList.remove("show");
  });

  document.querySelectorAll(".model-option").forEach((opt) => {
    opt.addEventListener("click", async () => {
      const model = opt.getAttribute("data-model");
      try {
        const res = await fetch("/api/model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model })
        });
        if (res.ok) {
          activeModelName.textContent = model;
          document.querySelectorAll(".model-option").forEach(o => o.classList.remove("active"));
          opt.classList.add("active");
          showToast(`Active model changed to: ${model}`);
        }
      } catch (err) {
        showToast("Failed to switch model", "error");
      }
    });
  });

  // Dropzone and File Picker
  dropzone.addEventListener("click", () => {
    if (!currentSelectedFile) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectFile(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  });

  function selectFile(file) {
    currentSelectedFile = file;
    selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    dropzoneDefault.style.display = "none";
    fileSelectedInfo.style.display = "block";
  }

  clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentSelectedFile = null;
    fileInput.value = "";
    dropzoneDefault.style.display = "block";
    fileSelectedInfo.style.display = "none";
  });

  // Upload and Index
  uploadBtn.addEventListener("click", async () => {
    if (!currentSelectedFile) {
      showToast("Please choose a file to upload first.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", currentSelectedFile);

    const originalBtnHtml = uploadBtn.innerHTML;
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = `<span class="spinner"></span> <span>Indexing document...</span>`;

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      showToast(`Indexed "${data.filename}" (${data.chunks_count} chunks)`, "success");
      clearFileBtn.click();
      await fetchDocuments();
      await fetchStatus();
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = originalBtnHtml;
    }
  });

  // Fetch Documents
  async function fetchDocuments() {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      renderDocuments(data);
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  }

  // Fetch Status
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      corpusBadge.textContent = data.badge_text;
      if (data.is_empty) {
        corpusBadge.classList.remove("active");
      } else {
        corpusBadge.classList.add("active");
      }
      if (data.active_model) {
        activeModelName.textContent = data.active_model;
      }
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  }

  function renderDocuments(data) {
    indexedStats.textContent = `${data.total_docs} docs · ${data.total_chunks} chunks`;

    if (!data.documents || data.documents.length === 0) {
      documentsList.innerHTML = `<div class="empty-state">No indexed documents.</div>`;
      return;
    }

    documentsList.innerHTML = "";
    data.documents.forEach(doc => {
      const item = document.createElement("div");
      item.className = "doc-item";
      item.innerHTML = `
        <div class="doc-info">
          <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
          <span class="doc-meta">${doc.chunks_count} chunks</span>
        </div>
        <button class="doc-delete-btn" data-doc="${encodeURIComponent(doc.filename)}" title="Remove document">✕</button>
      `;

      item.querySelector(".doc-delete-btn").addEventListener("click", async (e) => {
        const docName = decodeURIComponent(e.target.getAttribute("data-doc"));
        if (confirm(`Remove "${docName}" from corpus?`)) {
          try {
            const delRes = await fetch(`/api/documents/${encodeURIComponent(docName)}`, {
              method: "DELETE"
            });
            if (delRes.ok) {
              showToast(`Removed "${docName}"`);
              await fetchDocuments();
              await fetchStatus();
            }
          } catch (err) {
            showToast("Failed to delete document", "error");
          }
        }
      });

      documentsList.appendChild(item);
    });
  }

  // Refresh Button
  refreshBtn.addEventListener("click", async () => {
    refreshBtn.disabled = true;
    await fetchDocuments();
    await fetchStatus();
    refreshBtn.disabled = false;
    showToast("Corpus refreshed");
  });

  // Clear Corpus Button
  clearCorpusBtn.addEventListener("click", async () => {
    if (confirm("Are you sure you want to clear all indexed documents from the corpus?")) {
      try {
        const res = await fetch("/api/clear", { method: "POST" });
        if (res.ok) {
          showToast("Corpus cleared successfully.");
          await fetchDocuments();
          await fetchStatus();
          retrieveOutput.classList.remove("visible");
          askOutput.classList.remove("visible");
        }
      } catch (err) {
        showToast("Failed to clear corpus", "error");
      }
    }
  });

  // Run Retrieval
  runRetrievalBtn.addEventListener("click", async () => {
    const query = retrieveInput.value.trim();
    if (!query) {
      showToast("Please enter a retrieval query", "error");
      return;
    }

    const topK = parseInt(retrieveTopK.value, 10) || 4;
    const originalBtn = runRetrievalBtn.innerHTML;
    runRetrievalBtn.disabled = true;
    runRetrievalBtn.innerHTML = `<span class="spinner" style="border-top-color:#090d16;"></span> <span>Retrieving...</span>`;

    try {
      const res = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Retrieval failed");

      renderRetrievalResults(data.results);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      runRetrievalBtn.disabled = false;
      runRetrievalBtn.innerHTML = originalBtn;
    }
  });

  function renderRetrievalResults(results) {
    retrieveOutput.classList.add("visible");
    retrieveCount.textContent = results.length;

    if (!results || results.length === 0) {
      retrieveResultsList.innerHTML = `<div class="empty-state">No matching chunks found in corpus.</div>`;
      return;
    }

    retrieveResultsList.innerHTML = "";
    results.forEach((match, idx) => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML = `
        <div class="chunk-top">
          <span class="chunk-doc">${match.doc_name} · Chunk #${match.chunk_index + 1}</span>
          <span class="chunk-score">Score: ${match.similarity}</span>
        </div>
        <div class="chunk-text">${escapeHtml(match.text)}</div>
      `;
      retrieveResultsList.appendChild(card);
    });
  }

  // Ask RAG
  askRagBtn.addEventListener("click", async () => {
    const query = askInput.value.trim();
    if (!query) {
      showToast("Please enter a question to ask", "error");
      return;
    }

    const topK = parseInt(askTopK.value, 10) || 4;
    const model = activeModelName.textContent.trim();

    const originalBtn = askRagBtn.innerHTML;
    askRagBtn.disabled = true;
    askRagBtn.innerHTML = `<span class="spinner"></span> <span>Synthesizing...</span>`;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK, model })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "RAG query failed");

      renderAnswer(data);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      askRagBtn.disabled = false;
      askRagBtn.innerHTML = originalBtn;
    }
  });

  function renderAnswer(data) {
    askOutput.classList.add("visible");
    answerModelBadge.textContent = data.model_used;
    answerText.innerHTML = formatMarkdown(data.answer);

    if (data.sources && data.sources.length > 0) {
      sourcesCount.textContent = data.sources.length;
      sourcesToggle.style.display = "inline-flex";
      sourcesContent.innerHTML = "";
      data.sources.forEach((s) => {
        const item = document.createElement("div");
        item.className = "chunk-card";
        item.style.marginTop = "8px";
        item.innerHTML = `
          <div class="chunk-top">
            <span class="chunk-doc">${s.doc_name} (Chunk #${s.chunk_index + 1})</span>
            <span class="chunk-score">Score: ${s.similarity}</span>
          </div>
          <div class="chunk-text">${escapeHtml(s.text)}</div>
        `;
        sourcesContent.appendChild(item);
      });
    } else {
      sourcesToggle.style.display = "none";
      sourcesContent.innerHTML = "";
    }
  }

  // Sources toggle
  sourcesToggle.addEventListener("click", () => {
    const isOpen = sourcesContent.classList.toggle("show");
    sourcesToggle.querySelector("span").textContent = isOpen 
      ? `▼ Hide retrieved context sources (${sourcesCount.textContent})`
      : `▶ View retrieved context sources (${sourcesCount.textContent})`;
  });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Simple bold
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Code blocks / inline code
    safe = safe.replace(/`([^`]+)`/g, "<code style='background:rgba(255,255,255,0.08);padding:2px 4px;border-radius:4px;font-family:monospace;'>$1</code>");
    // Line breaks to paragraphs
    const paragraphs = safe.split("\n\n").map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    return paragraphs;
  }

  // Initial Load
  fetchStatus();
  fetchDocuments();
});

