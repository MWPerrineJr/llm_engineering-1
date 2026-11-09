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

## Session 3 — 2026-11-01

### Notebook tidy: done ✅
- Empty cells deleted. The PUA / no-section diagnostic was kept as a regression check (it now prints 0 and an empty Counter).
- The two duplicate stats cells were replaced by `chunk_report(chunks, label, show=True)`, which returns a per-chunk DataFrame (`library, chars, tokens, strategy`). It is called as `page_df = chunk_report(page_chunks, "page-level")` and `section_df = chunk_report(chunks, "section-level")`.
- New comparison cell, `def summarize(df):`, with a side-by-side table and an overlaid histogram ("Effect of merging pages into section runs").
  - Fixed bins (`size=50`, 0–2,000) on every histogram, so differences reflect the data rather than different bin edges.
  - Each row is rounded inside `summarize`, not with a table-wide `.round(1)` (that made both costs show as $0.1).
  - Bug fixed: `return df` was inside `if show:`, so `show=False` returned None.
- Results are unchanged: 1,652 vs 1,294 chunks, std dev 627 vs 362, cost $0.079 vs $0.082.

### chunk_id fix ✅
- Both strategies used to number from 0, so IDs collided (which would break an upsert keyed on chunk_id).
- Now `f"{library}-p{i:04d}"` (page-level) and `f"{library}-s{i:04d}"` (section-level).
- New cell `# --- Sanity check: chunk_ids are unique across both strategies ---` asserts uniqueness and prints `matplotlib-p0000 matplotlib-s0000`.

**Lesson:** `page_chunks = chunks` binds a second name to the same list; it does not copy it. The assert failed because the Join cell hadn't been rerun after the Trim cell, so both names pointed to the page-level list. Always Restart & Run All (and save) before any step that writes data.

### Supabase findings (read-only)
- Project: **data-science-rag** (`qrerekdrynpjzaqxokbj`, us-east-2, Postgres 17).
- `vector` 0.8.2 is installed in the **public** schema (so use `vector(1536)`, not `extensions.vector`). 0.8 supports iterative index scans for filtered queries.
- `public.documents` already exists with the LangChain default shape: `id bigint` (sequence), `content`, `metadata jsonb`, `embedding vector`, RLS on, **0 rows**.
- Still unknown: the embedding dimension, whether `match_documents` exists, and whether there's an HNSW index.

### Open items (pick up here next session)
- [x] Tidy notebook
- [x] Supabase check: which project holds `documents`
- [x] Step 2: `documents` recreated; schema, HNSW index and `match_documents` applied (see below)
- [x] Step 5: embed + upsert (see below)

### Step 2: Supabase schema ✅
Decision: **drop and recreate** `documents`. The old LangChain default (`id bigint` sequence) couldn't hold string chunk_ids, so re-runs would have duplicated rows. It was empty, so nothing was lost.

| Object | Definition |
|---|---|
| `id` | `text` primary key, = `chunk_id` (makes upserts idempotent; matches LangChain's `ids=` argument) |
| `content`, `metadata` | `text not null`, `jsonb not null` |
| `embedding` | `vector(1536) not null` (`vector` 0.8.2 lives in `public`) |
| `library` | `text generated always as (metadata->>'library') stored`, btree-indexed |
| `documents_embedding_hnsw` | `hnsw (embedding vector_cosine_ops)` |
| `match_documents(query_embedding, match_count=5, filter jsonb='{}')` | returns `id, content, metadata, similarity`, filters with `metadata @> filter` (LangChain-compatible signature) |
| RLS | on (ingestion uses the service-role key) |

**Error and fix:** `42P13 cannot change return type of existing function`. An older `match_documents` returned `id bigint`, and `create or replace` can't change OUT columns. Fixed by adding `drop function if exists public.match_documents(vector, integer, jsonb);` first. To list overloads: `select oid::regprocedure from pg_proc where proname = 'match_documents';`

### Step 5: embed + upsert ✅
- `OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)`. The smoke test returned length 1536 with norm 1.0.
- `upsert_chunks(chunk_list, strategy)` runs in batches of 100, adds `metadata["strategy"]`, and upserts on `id`. 13 batches took 24 s.
- Loaded **section-level only** (`strategy="section"`). Page-level comes later with `upsert_chunks(page_chunks, "page")`, compared via `filter={"strategy": ...}` at Step 9.

| library | rows | sections |
|---|---|---|
| matplotlib | 161 | 33 |
| numpy | 171 | 22 |
| pandas | 249 | 21 |
| plotly | 183 | 37 |
| scikit-learn | 334 | 31 |
| scipy | 196 | 26 |
| **total** | **1,294** | **170** |

All vectors are 1536-dim with average norm 1.000. The counts match `section_df` exactly (verified in the notebook and with SQL).

**Lessons**
- `PGRST125 Invalid path specified in request URL`: `SUPABASE_URL` must be the bare `https://<ref>.supabase.co`, with no trailing `/` and no `/rest/v1`.
- PostgREST returns at most **1,000 rows** by default. `res.count` (with `count="exact"`) is exact, but `res.data` is truncated (it showed scipy = 2). Count per library on the server with `select("id", count="exact", head=True).eq("library", lib)`.
- Unsaved notebook ≠ what happened. Save before reviewing.

### Open items (pick up here next session)
- [ ] **Next: Step 6, retrieval sanity test.** `retrieve(question, k=5, **filters)` calling the `match_documents` RPC, with 5 probe questions plus a filtered plotly call. Check top-1 library, section sense, the sim gap from rank 1 to 5, and that filtered queries still return k rows
- [ ] Step 7: t-SNE embedding visualization (plotly, colored by library)
- [ ] Step 8: RAG chain with Pydantic structured output (`answer`, `sources`)
- [ ] Step 9: evaluation, hit rate@k and MRR; load `page_chunks` as `strategy="page"` for the baseline
- [ ] Step 10: Streamlit app (chat, citations, library filter)

### Step 6: retrieval sanity test ✅
`retrieve(question, k=5, **filters)` calls the `match_documents` RPC with `filter={"strategy": "section", **filters}`.

**The first run exposed an ingestion bug: section names split by casing.** Chapter openers keep their original title ("Reshaping and pivot tables"), while running headers are uppercase and passed through `.title()` ("Reshaping And Pivot Tables"). `groupby` treated them as different keys, so **every chapter's opening page became its own one-page section run**, cut off from its content.

Fix: a new cell after the load/clean cell, `# --- Canonicalize section names ... ---`. It groups by `(library, section.casefold())` and keeps the variant that isn't plain `.title()` output (the opener's wording, which also restores `scipy.optimize`). 69 sections had mixed casing.

| | Before | After |
|---|---|---|
| Section runs | 170 | **101** |
| Section-level chunks | 1,294 | **1,264** |
| Chunks < 1,600 chars | 9.6% | **5.9%** |
| Std dev chars | 362 | **269** |
| Tokens / cost | 632,452 / $0.082 | 633,383 / $0.082 |

Reloaded with delete-then-upsert in the upsert cell:
    supabase.table(TABLE_NAME).delete().eq("metadata->>strategy", "section").execute()
    upsert_chunks(chunks, "section")
SQL check: 1,264 rows, 1,264 distinct ids, 101 sections (the same count when case-insensitive), no stale rows.
Per library: matplotlib 155, numpy 167, pandas 246, plotly 179, scikit-learn 325, scipy 192.

**Scorecard (after the fix)**
| Probe | Top-1 | Right library in top 5 | Sim @1 → @5 |
|---|---|---|---|
| Pivot long → wide | pandas · Reshaping and pivot tables (100–111) | 5/5 | 0.587 → 0.482 |
| Tune hyperparameters with CV | scikit-learn · Cross-validation (58–72) | 5/5 | 0.622 → 0.521 |
| Minimize with constraints | scipy · Optimization (scipy.optimize) (5–35) | 5/5 | 0.561 → 0.529 |
| Broadcast arrays | numpy · Broadcasting (62–67) | 5/5 | 0.682 → 0.541 |
| Secondary y-axis (ambiguous) | mixed plotly / matplotlib | — | 0.464 → 0.448 (flat) |
| Same question, `library="plotly"` | plotly only, 5 rows | ✓ | 0.476 → 0.447 |

- Clear questions score about 0.56–0.68 at rank 1, and the ambiguous one stays flat around 0.45. A cutoff around 0.5 could drive a "not confident" answer at Step 8; tune it at Step 9. The PDFs may not cover `twinx` / `secondary_y` directly (a knowledge-base gap).
- The filter returns all k rows, so post-filtering after the HNSW search isn't losing results at this scale.
- The same chunk scored 0.467 vs 0.485 across two `embed_query` calls (OpenAI embeddings aren't exactly deterministic). **At Step 9, embed each eval question once and reuse it.**
- Single-query score shifts after re-chunking are noise; judge chunking changes with hit rate@k / MRR.

**Lessons**
- Inspecting retrieval output (section names plus page ranges) found an ingestion bug that the chunk-length stats didn't show. Look at real results, not just aggregates.
- An upsert never deletes. When chunk ids can change, delete by `strategy` first so stale rows don't linger.

### Open items (supersedes the list above)
- [x] Step 6: retrieval sanity test, and the section-casing fix
- [ ] **Next: Step 7, t-SNE.** Fetch embeddings paged past the 1,000-row cap (`.range()`), `json.loads` each vector (pgvector returns a string), `TSNE(metric="cosine", perplexity=30, init="pca", random_state=42)`, plotly colored by library with section/id hover. Try perplexity 15 and 50
- [ ] Step 8: RAG chain with Pydantic structured output (`answer`, `sources`) plus a low-confidence cutoff
- [ ] Step 9: evaluation, hit rate@k and MRR; load `page_chunks` as `strategy="page"` for the baseline; embed questions once
- [ ] Step 10: Streamlit app (chat, citations, library filter)

### Step 7: t-SNE visualization ✅
- Embeddings are fetched from Supabase in pages (`.range(start, start+999)`) to get past the 1,000-row cap. Each vector goes through `json.loads`, because pgvector comes back as a string.
- `plot_tsne(X, meta, dims=2|3, perplexity=30, seed=42)` uses cosine t-SNE with `init="pca"`, plotly colored by library, hover showing section and id, and returns the coordinates. 2D and 3D need separate fits (3D is the Barnes-Hut maximum).
- **10-NN same-library rate on the t-SNE layout: 0.975 (both 2D and 3D).** scikit-learn 1.000, pandas 0.99, plotly 0.97, scipy 0.96, matplotlib 0.95, numpy 0.95.
- Overlap pairs: **matplotlib ↔ plotly** and **numpy ↔ scipy** (consistent in 2D and 3D, so real rather than a projection artifact). The outliers are mostly `API quick reference` chunks (signature lists look alike across libraries).
- t-SNE keeps local neighborhoods, not distances between clusters. For a check in the real geometry, use `NearestNeighbors(metric="cosine")` on `X`.

### Ingestion fixes found via t-SNE and eval prep ✅
1. **TOC continuation pages.** The `Contents` filter only caught pages *starting* with "Contents". Each table of contents runs onto a second page with "CONTENTS" as the running header, so it got parsed as section `Contents`: 19 dot-leader chunks. Fix: drop `section.casefold() == "contents"` after canonicalizing. Dropped 10 pages (matplotlib 2, numpy 1, pandas 2, plotly 2, scikit-learn 2, scipy 1).
   - **Bug on the way:** `.casefold != "contents"` (missing `()`) compares a method object to a string, which is always True, so nothing was dropped and no error was raised. It was caught only because the drop-count print showed `{}`.
2. **Hyphenation and spacing splits.** `rc- Params` / `Rcparams`, `(ufunc) basics` / `(Ufunc)Basics`, `prac- tices` / `Practices`. Fix: match on `section_key(s) = re.sub(r"[^a-z0-9]", "", s.casefold())` and rejoin line-break hyphens with `re.sub(r"(\w)- (\w)", r"\1\2", name)`. 72 variant spellings merged.
- Also: rerunning a single cell isn't a fresh run. The canonicalize cell printed `0 sections had mixed casing` because it ran on already-fixed `docs`. Restart & Run All.

**Current data (notebook and Supabase agree)**
| | Value |
|---|---|
| Pages | 1,030 |
| Section runs | **92** |
| Section-level chunks (`strategy="section"`) | **1,243** (matplotlib 151, numpy 164, pandas 243, plotly 174, scikit-learn 321, scipy 190) |
| Page-level chunks (`strategy="page"`) | **1,631** |
| `Contents` / dot-leader / hyphen-broken rows | 0 / 0 / 0 |
| Distinct sections (normalized = raw) | 92 = 92 for both strategies |

### Step 8: RAG chain ✅
- `Answer(BaseModel)`: `answer: str`, `cited: list[int]` via `ChatOpenAI(model=MODEL).with_structured_output(Answer)`.
- **The LLM cites block numbers only; citations (library, version, section, pages, url, sim) are built in code from Supabase metadata**, so it can't invent pages or URLs.
- Confidence gate **before** the LLM: if top-1 sim < `MIN_SIM`, return "not confident" without a model call.
- `dedupe` uses `setdefault` to keep the *first* (best-scoring) chunk per section. A dict comprehension kept the last one (0.487 instead of 0.579).
- **`gpt-6-luna` rejects `temperature=0`** (only the default of 1 is supported). Answers aren't deterministic, so grade answer quality over several runs or from saved outputs. Retrieval metrics don't involve the LLM.
- Smoke test: pivot answered correctly with a citation (pandas · Reshaping and pivot tables, 100–111). Secondary y-axis (0.459) and capital of France (0.10) were refused at a cutoff of 0.50.
- Known limitation: citation pages cover the whole section run (e.g. API quick reference 149–220). Future fix: `add_start_index=True`, then map the offset to a page.

### Step 9: evaluation, first run ✅
48 questions (43 answerable with an expected `(library, section)`, 5 out-of-scope). Each is embedded once (`embed_documents`) and reused across strategies.

| Metric | page (1,631) | section (1,243) |
|---|---|---|
| hit@1 | **0.884** (38/43) | 0.837 (36/43) |
| hit@3 | **0.953** | 0.930 |
| hit@5 | 1.000 | 1.000 |
| hit@10 | 1.000 | 1.000 |
| MRR | **0.922** | 0.892 |
| library@1 | 0.953 | **0.977** |

- **The difference is 2 questions at hit@1, which is within noise for n=43** (1 question ≈ 2.3 pts). Page-level also has more chunks per section ("more lottery tickets") under a section-label metric, and the metric doesn't measure how complete the retrieved context is, which is the reason section-level chunking exists. Not decided yet; needs a paired comparison and answer-quality grading.
- Section-level misses at rank > 3: KD-tree nearest neighbors (rank 5), subplot labels overlapping (rank 4), dark theme for all figures (rank 4, top-1 = matplotlib). The dark-theme question doesn't name plotly, and matplotlib's style sheets answer it too, so it's a labeling ambiguity rather than a retrieval failure.

**Confidence cutoff (section strategy)**
| cutoff | answerable kept | unanswerable rejected |
|---|---|---|
| 0.35 | 1.000 | 0.8 |
| **0.40** | **1.000** | **1.0** |
| 0.45 | 0.953 | 1.0 |
| 0.50 | 0.837 | 1.0 |
| 0.55 | 0.581 | 1.0 |

- **Set `MIN_SIM = 0.40`.** The original 0.50 would have refused 16% of answerable questions. The margin is thin (one unanswerable question scores between 0.35 and 0.40), and it rests on only 5 unanswerable questions; add more.

**Lessons**
- Bugs that raise no error (`.casefold` without parentheses) are caught only by printing what a filter removed.
- Eval labels must be unambiguous: a question that doesn't name a library can be correctly answered by another library.
- Small eval sets: read the misses before trusting differences of a few points.
- `groupby(...).apply(metrics)` raises a pandas FutureWarning. Pass `include_groups=False`.

### Open items (supersedes the lists above)
- [x] Steps 1–8, Step 9 first run
- [ ] Set `MIN_SIM = 0.40` in the RAG chain cell
- [ ] Step 9b: relabel or reword ambiguous questions (dark theme → "in Plotly"), add ~10 questions in your own words and ~10 more out-of-scope ones, run a paired page-vs-section comparison (per-question rank diff), and grade answer quality with LLM-as-judge over 2–3 runs
- [ ] Decide on a chunking strategy from 9b
- [ ] Optional: per-chunk page citations (`add_start_index=True`)
- [ ] Step 10: Streamlit app (chat, citations, library filter, "not confident" state)

## Session 4 — 2026-11-08

### MIN_SIM ✅
- `MIN_SIM = 0.40` in the RAG chain cell (was 0.50). The Step 8 markdown cell still says 0.50, so update the text.

### Step 9b: eval set v2 ✅
- 71 questions: 56 answerable (43 + 13 new in casual phrasing, aimed at sections not covered yet) and 15 out-of-scope (5 + 10 new **near-misses**: XGBoost, Dask, seaborn, statsmodels, Polars, Dash, Prophet, PyTorch GPU, SQL window, git).
- **Multi-label targets** (Option 2): `ALSO` dict → `eval_df["targets"]` = list of acceptable (library, section) pairs; hit = any target in the top k; library@1 checks against the target libraries. 5 multi-label questions: dark theme (+matplotlib rcParams), PNG export (+matplotlib Intro to Figures), Ax = b (+numpy API), NaN before training (+pandas missing data), subplot overlap (+plotly Subplots).
- Label assert: every target must exist in `chunks` metadata. It caught `Analyzing` vs the PDF's **`Analysing one sample`**.
- `pd.concat(..., ignore_index=True)`; the groupby selects columns explicitly (FutureWarning gone).

| Metric | page | section |
|---|---|---|
| hit@1 | 0.839 (47/56) | 0.804 (45/56) |
| hit@3 | 0.946 | 0.964 |
| hit@5 / @10 | 1.000 | 1.000 |
| MRR | 0.897 | 0.880 |
| library@1 | 0.964 | 0.964 |

- All 13 casual questions are within the top 3 (section). The only rank > 3 misses: KD-tree (5, top = scipy API quick reference) and subplot overlap (4, top = Text in Matplotlib).

### Paired page vs section ✅
- Ranks per question (miss = 11): **44 ties, 6 page better, 6 section better; Wilcoxon p = 0.69 (n = 12).**
- **Retrieval doesn't distinguish the strategies.** Decide chunking on answer quality (completeness of context), not hit rate.

### Confidence gate with near-misses
| cutoff | answerable kept | out-of-scope rejected |
|---|---|---|
| 0.35 | 1.000 | 0.467 |
| **0.40** | **1.000** | **0.733** |
| 0.45 | 0.929 | 0.800 |
| 0.50 | 0.804 | 0.867 |
| 0.55 | 0.518 | 1.000 |

- Leaks at 0.40: XGBoost early stopping 0.530 (→ sklearn Ensembles), Dask groupby 0.515 (→ pandas Group by), seaborn pairplot 0.471 (→ plotly Express), statsmodels OLS p-values 0.421 (→ scipy API).
- **The distributions overlap; no similarity cutoff separates them** (0 leaks costs 48% of answerable questions). Keep 0.40 as a cheap first filter.
- The 0.45 row moved 0.946 → 0.929 between identical runs (re-embedding is nondeterministic). Cache the question vectors to disk for stable comparisons.

**Lessons**
- Easy out-of-scope questions (capital of France) made the gate look perfect. Near-misses are the real test.
- When pasting a replacement, paste *over* the old lines. The leftover old 3 lines silently rebuilt `eval_df` without `targets`; it only failed one cell later.
- A label typo is a silent miss, not an error. Assert labels against the data.

### Open items (supersedes the lists above)
- [ ] Update the Step 8 markdown cell to 0.40; delete the empty cell after the paired-comparison cell
- [ ] Cache `eval_df` question vectors (e.g. parquet/npy) so reruns don't re-embed
- [ ] **Next: second layer of refusal.** Add `in_context: bool` to `Answer`; the LLM sets it False when the blocks don't answer the question. Test on the 4 leaks plus all answerable questions
- [ ] LLM-as-judge (2–3 runs): faithfulness, completeness, correct refusal, page vs section → decide chunking
- [ ] Optional: per-chunk page citations (`add_start_index=True`)
- [ ] Step 10: Streamlit app (chat, citations, library filter, "not confident" state)

## Session 4b — 2026-11-08 (afternoon)

### Second refusal layer: `in_context` ✅
- `Answer` now has `in_context: bool` as its **first** field, so the model decides whether the context covers the question before it writes an answer. `cited` is empty when the field is False.
- The system prompt says each block is labeled with its library, and if the question asks about a library or tool none of the blocks come from, set `in_context=False` instead of answering with a different library's equivalent. **The leak libraries are deliberately not named in the prompt**, so the prompt isn't fitted to the eval set.
- `ask()` returns `refused_by` = `"gate"` | `"llm"` | `None`, plus `llm_note` (the model's explanation) when the LLM refuses.
- `search()`/`ask()` accept `qv=` to reuse cached question vectors (`eval_df.qv`), so eval runs don't re-embed.
- Bug: the rpc key was typed `query_embeddings` (plural) → PGRST202 "function not found in schema cache". PostgREST finds a function by its argument names, so the key must be `query_embedding`.

### Results (full eval set, section strategy, cached vectors)
| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Answerable answered | 56/56 | 56/56 | 56/56 |
| Out-of-scope refused | 15/15 | 15/15 (gate 11, LLM 4) | 15/15 (gate 11, LLM 4) |
| Cost | — | $0.0224 | $0.0225 |

- The LLM layer caught all 4 leaks (XGBoost, Dask, seaborn, statsmodels) with **0 false refusals**. With the gate alone at 0.40, the rejection rate was 0.733.
- Caveat: 0/15 leaks still allows a true leak rate of up to about 20% (the rule of three, 95%). Add more near-miss questions before relying on it.

### Cost tracking ✅
- `with_structured_output(Answer, include_raw=True)` → `res["raw"].usage_metadata`; `llm_cost()` uses a `PRICES` dict (gpt-6-luna $0.10 in / $0.01 cached / $0.50 out per 1M; check against OpenAI's pricing page). Gate refusals cost $0 (`NO_USAGE`).
- Run 2: 60 LLM calls out of 71 questions (the gate skipped 11), 172,541 in / 10,361 out tokens, **$0.0224 total, $0.000374 per call, $0.37 per 1,000 questions.**
- Answers: about 2,874 in / 181 out tokens. LLM refusals: about 2,899 in / 61 out.

**Lessons**
- A typo in an rpc argument name doesn't give a "bad argument" error; it gives "function not found". Read the error's `hint`.
- Save the notebook (Cmd+S) before asking Claude to read outputs. The file on disk lags behind the kernel.

### Open items (supersedes the lists above)
- [x] Third stability run: identical decisions 3/3 (input tokens identical, so retrieval with cached vectors is deterministic; output tokens 10,361 → 10,522)
- [ ] Prompt: stop putting block numbers (`[1, 4]`) in the answer text (they're already in `cited`)
- [ ] Cache `eval_df` question vectors to disk (parquet/npy) so they survive a kernel restart
- [ ] **Next: LLM-as-judge, page vs section.** Generate answers for the 56 answerable questions with both strategies (`ask(q, strategy='page')`); have `ask()` also return its context; judge with a stronger model (GPT-6 Sol) to avoid self-preference bias; hide the strategy and randomize A/B order (position bias); score faithfulness, completeness (1–5) and a pairwise preference; report win/tie/loss, a sign test, and quality per $. Run 2–3 times. Estimated $0.80–$1 per run.
- [ ] More near-miss out-of-scope questions (to tighten the leak-rate bound)
- [ ] Optional: per-chunk page citations (`add_start_index=True`)
- [ ] Step 10: Streamlit app (chat, citations, library filter, "not confident" state)

## Session 4c — 2026-11-08 (evening)

### Small fixes ✅
- `ask()` returns `context` (None for gate refusals) so the judge can check faithfulness. The first paste left the new returns *under* the old ones, where they were dead code. Fixed by replacing them.
- `llm_cost` key fixed: `cache_read` (it was `cached_read`, so cached tokens were billed at the full input price).
- Prompt: "don't answer with a different library's equivalent" restored, and block numbers are now kept out of the answer text (they go only in `cited`). The smoke test confirms no `[1, 4]` in answers.
- Still to fix: `answer` description says "In in_context" (should be "If"), and there's a stray `"` in the `cited` description.

### Step 9d: LLM-as-judge, page vs section ✅
- 9d-1: answer all 56 answerable questions with both strategies (`ask(q, qv=..., strategy=...)`). 0 refusals under either strategy.
- 9d-2: judge `gpt-6-sol` (stronger than `gpt-6-luna`, which wrote the answers, to avoid self-preference bias). It's **blind** to the strategy, with the **A/B order randomized** per question (`np.random.default_rng(42 + run)`). `Judgment` puts `reason` first, then faithful_a/b (checked against *each answer's own* context), complete_a/b (1–5), and preferred A/B/tie.
- 9d-3: majority vote over 3 runs, sign test (ties dropped), quality means, position-bias and consistency checks. 9d-4: lists unfaithful answers, plus a swap × preferred crosstab.
- Judge cost: $2.02 for 3 runs (168 judgments, about $0.012 each). The cached price for Sol is set equal to its input price until verified.

| | page | section |
|---|---|---|
| Majority wins (33 ties) | **14** | 9 |
| Wins across all 168 judgments | **47** | 29 |
| Faithful | **0.970** | 0.952 |
| Completeness (1–5) | **4.76** | 4.66 |
| Input tokens per 56 answers | **131K** | 162K (+23%) |

- Sign test p = 0.405 (n = 23): **not significant**, but page is never behind on any measure. (Run 0 alone: 17–11 for page, p = 0.345.)
- **No position bias:** when page was in slot A it won 20–15; when page was in slot B it won 27–14. Overall A 34 / B 42. The judge follows the strategy, not the slot.
- The judge is noisy: the same winner in all 3 runs for 62.5% of questions. **Caveat:** 9d-1 was re-run after run 0, so run 0 judged different answers than runs 1–2. The 62.5% mixes judge variability with answer variability.
- **Faithfulness flags (12 of 336) are almost all *correct but ungrounded* facts**, where the model filled gaps from its training: `best_params_`, `font.size`, `jac` optional in SLSQP constraints, `RdBu_r`, ridge = L2. The one genuinely weak answer: integer-array indexing (under both strategies).

### Decision: **page-level chunking** ✅
- Retrieval tied (Session 4: Wilcoxon p = 0.69) and answer quality tied (p = 0.405, with page slightly ahead). Page uses about 19% fewer input tokens per answer and is simpler (no section merging). Section chunking's intended benefit, more complete context, didn't show up in either retrieval or answers.

**Lessons**
- **Prompt caching changes costs between runs.** The second generation hit the cache: same input tokens, but the cost went from $0.0175 to $0.0062. Compare strategies on **token counts**, not dollars; the first-run cost is the realistic per-query figure.
- Re-running an upstream cell (9d-1) silently changes what later "runs" are measuring. Freeze the inputs (save them to disk) before doing repeated judge runs.
- Faithfulness and correctness are different things. A strict judge flags correct facts that aren't in the context. Decide whether that's acceptable for the product.

### Open items (supersedes the lists above)
- [ ] Save `gen` and `judged` to parquet so the $2 judge run never has to be repeated
- [ ] Switch `search()`/`ask()` default from `'section'` to `'page'`; re-run the two-layer refusal check on page (0.40 was tuned on section top-1 sims, so re-check the cutoff sweep for page)
- [ ] Fix the two description typos in `Answer`
- [ ] Optional: tighten the prompt against outside knowledge ("if a detail isn't in the blocks, leave it out") and re-judge faithfulness only
- [ ] Optional: the integer-array indexing answer is weak under both strategies, so check whether the PDF covers the single-index-array case
- [ ] Cache `eval_df` question vectors to disk
- [ ] More near-miss out-of-scope questions
- [ ] Step 10: Streamlit app (chat, citations, library filter, "not confident" state)
