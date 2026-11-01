# Data Science RAG — Session Log

## Session 1 — 2026-10-31

### Notebook review (`datascience_rag.ipynb`, 3 cells)
Setup cell is solid: documented constants, fail-fast env checks, 1536 dims (under the 2,000-dim HNSW limit), 2000/200 char chunking.

Fixes needed:
1. **Model string mismatch**: `"openai/gpt-6-luna"` is LiteLLM format, but `ChatOpenAI` expects a bare model name. Drop the `openai/` prefix, or switch to `litellm` / `ChatLiteLLM`.
2. **Missing imports**: a PDF loader (`PyMuPDFLoader`) and `RecursiveCharacterTextSplitter` (`langchain_text_splitters`).
3. **Supabase schema not covered yet**: needs the `vector` extension, a `documents` table with `vector(1536)`, and a `match_documents` function.
4. **Key type**: `SUPABASE_KEY` should be the service-role key for ingestion (RLS blocks anon inserts). Keep it in `.env` / Streamlit secrets only.
5. `json` and the t-SNE/plotly imports are unused for now (needed at the visualization step).

### Knowledge base findings
| PDF | Pages | Characters |
|---|---|---|
| matplotlib 3.11.2 | 130 | 256K |
| NumPy 2.5.4 | 151 | 277K |
| pandas 3.1.0 | 220 | 416K |
| Plotly 7.1.0 | 158 | 302K |
| scikit-learn 1.9.1 | 234 | 544K |
| SciPy 1.18.1 | 153 | 322K |
| **Total** | **1,046** | **~2.1M** |

- All text-based (LaTeX via pandoc), none scanned.
- **pypdf glues words together**: 14,341 glued words (25+ letters) vs 104 with `pdftotext`. Don't use `PyPDFLoader`.
- Page headers ("pandas 3.1.0 reference RESHAPING AND PIVOT TABLES"): parse into `library` / `version` / `section` metadata, then strip.
- 74 `Source: https://...` lines: carry forward as `source_url` metadata for citations.
- Unicode math italic (𝑅𝐺𝐵): apply `unicodedata.normalize("NFKC", text)`.
- Estimate: ~1,200 chunks, ~530K tokens, ~$0.07 to embed with text-embedding-3-large.

**Lesson:** always look at extracted text before embedding. Most RAG quality problems come from ingestion, not the model.

### PDF library decision: PyMuPDF
Install (from `llm_engineering`):
    uv add pymupdf        # or: pip install pymupdf

Loader cell:
    from langchain_community.document_loaders import PyMuPDFLoader

    docs = []
    for pdf in sorted(KNOWLEDGE_BASE_PATH.glob("*.pdf")):
        pages = PyMuPDFLoader(str(pdf)).load()    # one Document per page
        for d in pages:
            d.metadata["library"] = pdf.stem.replace("_reference", "")
        docs.extend(pages)

    print(f"{len(docs)} pages loaded")             # expect 1,046

Alternatives considered: `pypdf` (rejected: glued words), `pymupdf4llm` (Markdown output, good for header-based splitting later), `pdftotext` (clean, but a system binary, which makes deployment harder).

⚠️ PyMuPDF not yet tested on these files. Rerun the glued-word check after installing.

### Build plan
1. Fix config cell
2. Supabase schema SQL (extension, table, HNSW index, `match_documents`)
3. Load PDFs with metadata (library, source, page, section, source_url)
4. Chunk + inspect chunk-length distribution
5. Embed + upsert in batches (token/cost estimate first)
6. Retrieval sanity test (top-k + scores)
7. t-SNE embedding visualization (plotly, colored by library)
8. RAG chain with Pydantic structured output (`answer`, `sources`)
9. Evaluation: hit rate@k and MRR on a small question set
10. Streamlit app (chat, citations, library filter)

### Open items
- [ ] Install PyMuPDF and run the glued-word verification
- [ ] Supabase check: which project holds the `documents` table?
- [ ] Step 1: config cell edits (show current and proposed side by side)

## Session 2 — 2026-10-31 (evening)

### Notebook state
- Imports cell: added `re`, `unicodedata`, `Counter`, `pandas`, `PyMuPDFLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`.
- Config cell: `MODEL = "gpt-6-luna"` (bare name for ChatOpenAI). Restart & Run All passes the env and path checks.
- Pydantic `Result` / `Chunk` cell moved to the bottom as "Experiment: LLM chunking" (not run). `Result` duplicates LangChain `Document`; `Chunk.as_result` still needs fixing (returns nothing, treats Document as dict). Revisit after baseline eval.
- Refer to cells by their first line, not by number.

### Ingestion: done ✅
| Check | Result |
|---|---|
| Pages loaded | 1,040 (6 contents pages skipped, 1 per PDF) |
| Glued words (25+ letters) | 104, all real class names (e.g. HistGradientBoostingClassifier) |
| Pages with section | 1,040 / 1,040 |
| Pages with source_url | 74 |
| Private Use Area chars | 0 |

PyMuPDF confirmed as the loader. Fixes made in `clean_page`, all found by inspecting output:
1. **Two-line header**: PyMuPDF splits "pandas 3.1.0 reference" and "SECTION NAME" onto separate lines. Read the section from the next line if it is uppercase.
2. **Chapter openers** have no running header: title line(s), then `Source:`. Use the title as the section. Other openers fall back to the first line (<60 chars).
3. **Private Use Area glyphs** (unmapped font chars): `str.maketrans` map U+EE47→": ", U+EE4E→"(", U+EE4F→")", U+EE55→"-", U+EE6C→"×". Applied before header parsing so section names are clean too.
4. **Version** for chapter openers taken from the PDF `title` metadata.
5. **Contents filter**: drop only pages starting with "Contents".

**Lesson:** a `". . ." > 10` filter wrongly dropped 18 real pages (doctest `...` in scipy/sklearn API pages). It was caught by uneven drop counts per library and a fall in glued words (TransformedTargetRegressor 9→3). Validate every heuristic filter against what it actually removed.

### Chunking: done ✅
- Metadata trimmed to `library, version, section, page, source, source_url` (+ `chunk_id`; section chunks also get `page_end`).
- Splitter: RecursiveCharacterTextSplitter 2000/200, separators `["\n\n", "\n", " ", ""]`.

| Metric | Page-level (`page_chunks`) | Section-level (`chunks`) |
|---|---|---|
| Chunks | 1,652 | **1,294** |
| Chunks < 1,600 chars | 775 (47%) | 124 (10%) |
| Chunks 1,800–2,000 chars | 759 (46%) | 1,142 (88%) |
| P25 / median chars | 793 / 1,686 | 1,931 / 1,964 |
| Std dev chars | 627 | 362 |
| Tiny (<200) | 14 | 1 |
| Tokens / est. cost | 606,823 / $0.08 | 632,452 / $0.08 |

- **Why page-level was bimodal:** average page ≈ 2,150 chars, so most pages gave one full chunk plus a 300–800-char leftover that cut thoughts at page breaks.
- **Fix:** `itertools.groupby` merges consecutive pages with the same (library, section) into 170 section runs, then splits. Remaining short chunks ≈ one per section end (expected). Citations become page ranges (`page` to `page_end`).
- Tokens rose slightly with fewer chunks because chunk boundaries now carry the 200-char overlap where page breaks had none.
- Cost assumes text-embedding-3-large @ $0.13/1M tokens; confirm current pricing.
- `page_chunks` kept as the baseline to compare against `chunks` (hit rate@k / MRR) at step 9.
- Optional refactor: the two stats/histogram cells are duplicates; wrap them in `chunk_report(chunks, label)`.

### Open items (pick up here next session)
- [x] Install PyMuPDF and run the glued-word verification
- [x] Step 1: config cell edits
- [x] Steps 3–4: load, clean, chunk (section-level, 1,294 chunks)
- [ ] Tidy: delete the obsolete skipped-pages diagnostic cell and empty cells
- [ ] Supabase check: which project holds the `documents` table? (Claude can list projects read-only on request)
- [ ] **Next: Step 2, Supabase schema SQL** (vector extension, `documents` table vector(1536), HNSW index, `match_documents`)
- [ ] Step 5: embed + upsert in batches (~1,294 chunks, ~$0.08)
- [ ] Step 6: retrieval sanity test
