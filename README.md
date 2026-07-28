## Selected retrieval configuration

The selected retrieval configuration is `compact`.

- Embedding model: `intfloat/multilingual-e5-small`
- Chunk maximum: 220 tokens
- Chunk overlap: 40 tokens
- Vector index: FAISS `IndexFlatIP`
- Similarity: cosine similarity through L2-normalised vectors
- Candidate pool: 40
- Final retrieval: Top 5 for evaluation, Top 4 for answer generation
- Maximum chunks per source: 1

The configuration was selected using development-set retrieval metrics,
manual chunk inspection, programme-specific retrieval performance and
multi-source recall. The holdout set was not used during configuration
selection.

Increasing candidate_k from 40 to 100 did not improve source recall.
It exposed more incorrectly boosted programme candidates, reducing
Hit@3, MRR and source-group accuracy. The final configuration therefore
uses candidate_k=40 with explicit catalogue and regulatory routing.

# University of Luxembourg Student Services RAG

A source-grounded Retrieval-Augmented Generation prototype for answering University of Luxembourg student-services questions using official University and Luxembourg public-service sources.

## Project status

The project currently has a completed and frozen retrieval pipeline.

Completed:

* Official source catalogue
* Automated HTML and PDF collection
* Raw-source caching
* Content cleaning and validation
* Structure-aware chunking
* Multilingual embeddings
* FAISS vector indexing
* Metadata-aware query routing
* Retrieval evaluation
* Compact-versus-broad chunking comparison
* Final retrieval-configuration selection

Remaining:

* Grounded answer generation
* Source-citation formatting
* Unsupported-question abstention
* Streamlit interface
* Final frozen holdout evaluation

---

## Project overview

University information is distributed across admissions pages, programme pages, regulations, academic-matters pages, housing pages and external government services.

Students often need to search several sources to answer a single question, such as:

* How many programmes can I apply for?
* Which documents are required?
* What language proof must I provide?
* How do I re-enrol?
* What happens if my study progression is insufficient?
* Can I appeal an examination decision?
* Is University accommodation guaranteed?
* Which residence-permit procedure applies to a non-EU student?
* Which regulations apply to a particular Bachelor or Master programme?

This project creates a retrieval system that searches a curated collection of official sources, selects the most relevant passages and, in the next development stage, will provide those passages to a language model for grounded answer generation.

The system is not intended to replace the University of Luxembourg, Guichet.lu, CNS, MengStudien or another competent authority. It is a portfolio prototype demonstrating source collection, information retrieval, metadata-aware search, evaluation and grounded RAG design.

---

## Target users

The current project scope covers:

* Prospective Bachelor students
* Prospective Master students
* Current Bachelor students
* Current Master students

The assistant is designed to answer in English.

English sources are preferred, but official French regulations are retained where they are the authoritative source. A multilingual embedding model allows English questions to retrieve relevant French regulatory passages without creating a separate translation pipeline.

---

## Scope

### Included

The current corpus covers:

* Bachelor and Master admissions
* Admission eligibility
* Required application documents
* Proof of language skills
* Diploma recognition
* Recognition of prior experience
* University faculties and programme structure
* Selected Bachelor and Master programme pages
* Programme duration, ECTS and teaching languages
* Re-enrolment
* Payment and student status
* Study progression
* Special arrangements
* Academic conduct
* Appeals
* Official study regulations
* Bachelor and Master regulatory annexes
* Academic calendar
* University accommodation
* Residence permits and authorisation to stay
* Health insurance
* Financial support and AideFi
* Official external Luxembourg public-service information

### Excluded

The current version does not attempt to cover:

* Doctoral programmes and PhD administration
* Continuing education
* Guest-student procedures
* Staff administration
* Individual course pages across every programme
* Research-centre content
* News and events
* Unofficial blogs or student forums
* Personalised legal or administrative decisions
* Arbitrary user-uploaded documents
* Production authentication or user accounts

---

## Source policy

Only approved official sources are included.

The corpus contains:

* University of Luxembourg webpages
* University of Luxembourg regulatory PDFs
* Official Luxembourg public-service pages where Uni.lu refers students to an external authority

External official sources include relevant information from services such as:

* Guichet.lu
* CNS
* MengStudien

Discovered links are not automatically added to the corpus. They must be reviewed, triaged and explicitly approved in the source catalogue.

This prevents the index from being populated with:

* Navigation pages
* Duplicate language versions
* News pages
* Privacy pages
* Outdated documents
* Unrelated programme pages
* Unofficial advice

---

## Corpus snapshot

The frozen retrieval experiment uses:

| Item                 | Count |
| -------------------- | ----: |
| Approved sources     |    62 |
| Cleaned HTML sources |    59 |
| Cleaned PDF sources  |     3 |
| Compact chunks       | 3,414 |
| Broad chunks         | 2,057 |
| Embedding dimensions |   384 |

The three PDF sources contain official French regulatory material:

* General Study Regulations
* Bachelor regulatory annex
* Master regulatory annex

Each cleaned source retains YAML front matter and structured metadata such as:

* `source_id`
* `title`
* `source_url`
* `source_group`
* `audience`
* `degree_level`
* `faculty`
* `programme_name`
* `language`
* `is_external`
* document type
* retrieval and collection metadata

PDF chunks additionally retain page metadata where available.

---

## System architecture

```text
Official webpages and PDFs
        ↓
Source catalogue and approval
        ↓
Raw HTML/PDF cache
        ↓
Cleaned Markdown documents
        ↓
Structure-aware chunks
        ↓
Multilingual E5 embeddings
        ↓
FAISS IndexFlatIP
        ↓
Deterministic query routing
        ↓
Metadata-aware score adjustment
        ↓
Source-diverse top-k passages
        ↓
Grounded answer generation
```

The current repository has completed the pipeline through retrieval.

---

## Data collection pipeline

The collection workflow is implemented as numbered scripts:

```text
001_collect_sources.py
002_review_links.py
003_triage_discovered_links.py
004_validate_collection.py
005_final_compilation_sources.py
006_status_counts.py
```

The process:

1. Reads approved source records from `data/catalog/sources.csv`.
2. Fetches and caches official HTML and PDF sources.
3. Records redirects, content type, status, hashes and extraction information.
4. Cleans HTML into structured Markdown.
5. Extracts text and page markers from digital PDFs.
6. Discovers linked source candidates.
7. Keeps discovered links separate from the approved corpus.
8. Validates expected files, extraction status and metadata.
9. Records collection issues and validation results.

The raw files are preserved so that cleaned output can be regenerated without repeatedly requesting the original websites.

---

## Retrieval pipeline

The retrieval workflow is implemented as:

```text
007_prepare_eval.py
008_build_chunks.py
009_inspect_chunks.py
010_build_index.py
011_search_index.py
012_run_retrieval_eval.py
013_compare_retrieval_runs.py
```

### Chunk construction

Chunking is structure-aware rather than plain fixed-width splitting.

The splitter:

* Parses YAML front matter separately
* Tracks Markdown heading hierarchy
* Recognises PDF page markers
* Recognises legal chapter and article headings
* Preserves lists where possible
* Preserves Markdown tables or splits them by complete rows
* Removes obvious PDF header and footer noise
* Prevents embedding-model truncation
* Adds controlled source context to embedding text
* Creates deterministic chunk IDs
* Retains original source text separately from embedding text

Each chunk contains both:

```text
chunk_text
```

and:

```text
embedding_text
```

`chunk_text` contains the original cleaned source passage.

`embedding_text` adds factual retrieval context, such as:

```text
Title
Source group
Audience
Degree level
Faculty
Programme
Language
Section
Page
```

No LLM-generated summary is introduced during indexing.

---

## Embedding model

The selected embedding model is:

```text
intfloat/multilingual-e5-small
```

Reasons for selection:

* Designed for asymmetric query-to-passage retrieval
* Supports multilingual retrieval
* Produces 384-dimensional embeddings
* Allows English questions to retrieve occasional French regulatory passages
* Small enough for practical local development
* Supports substantially longer inputs than the originally considered multilingual paraphrase model

The model is used with its required formats:

```text
query: <student question>
```

```text
passage: <source chunk>
```

All embeddings are converted to `float32` and L2-normalised.

Similarity scores are used for ranking. They are not interpreted as calibrated probabilities or confidence percentages.

---

## Vector index

The selected vector index is:

```text
FAISS IndexFlatIP
```

The index performs exact exhaustive inner-product search.

Because document and query embeddings are L2-normalised, inner product corresponds to cosine similarity.

`IndexFlatIP` was selected because the corpus is small enough that approximate indexes are unnecessary. It provides:

* Exact retrieval
* No index-training stage
* No approximation error
* Simple reproducibility
* Straightforward debugging
* Very low retrieval latency for the current corpus

The FAISS index is accompanied by:

* Saved embedding matrix
* Ordered chunk metadata
* Build manifest
* Package versions
* Chunk-file hash
* Model and index settings

The following integrity condition is enforced:

```text
chunk count
=
embedding rows
=
FAISS index.ntotal
=
metadata rows
```

---

## Metadata-aware retrieval

Dense semantic similarity is combined with deterministic metadata routing.

The router detects, where possible:

* Exact programme
* Faculty
* Bachelor or Master level
* Prospective or current audience
* Admissions intent
* Programme-catalogue intent
* Re-enrolment
* Academic matters
* Regulations
* Accommodation
* Immigration
* Health insurance
* Financial support
* Diploma recognition
* Academic calendar

Hard filtering is used cautiously.

Exact programme names and explicit faculty references can safely restrict retrieval. General topics are primarily treated as score preferences so that regulations and supporting sources are not accidentally excluded.

The initial candidate set is reranked using metadata matches and then diversified by source.

---

## Explicit intent routing

Two previously ambiguous intents are handled separately.

### Programme-catalogue intent

Catalogue boosting is applied only to explicit catalogue-navigation questions such as:

* Compare study programmes
* Browse programmes
* Programme overview
* Programme catalogue
* List available programmes

It is not applied merely because a question contains the word `programme`.

This distinction prevents admissions or regulation questions from being incorrectly dominated by programme pages.

### Regulatory-document intent

Explicit terms such as the following route a question toward regulations:

* Official document
* Study regulations
* Regulatory provisions
* Bachelor annex
* Master annex
* General study rules
* Legal basis

This prevents programme pages from outranking official regulatory sources when the user is asking which regulation or annex applies.

---

## Source diversification

The final retrieval configuration allows:

```text
Maximum chunks per source: 1
```

during initial source selection.

This was introduced because earlier retrieval runs frequently returned two nearly identical chunks from the same source. Duplicate results consumed top-k positions and prevented useful alternative sources from entering the result set.

Selecting one initial chunk per source improves:

* Source diversity
* Multi-source retrieval
* Exact-source evaluation
* Context coverage for the later answer generator

The later answer-generation stage may expand neighbouring chunks from a selected source when additional local context is required.

---

## Evaluation design

Retrieval is evaluated independently from answer generation.

The retrieval evaluation asks:

```text
Did the system retrieve the correct official source?
```

It does not yet evaluate:

* Final answer wording
* Answer completeness
* Citation presentation
* Hallucination
* LLM abstention

### Development set

The current development set contains:

```text
35 answerable questions
```

It includes:

* Direct questions
* Lists
* Paraphrases
* Near-miss questions
* Programme-specific questions
* Multi-source questions
* Cross-language regulatory questions

The development set was used for:

* Chunk-size comparison
* Metadata-routing corrections
* Gold-source auditing
* Source-diversity tuning
* Retrieval-error analysis

### Holdout set

The holdout set was not used during retrieval configuration selection.

It must remain frozen until the complete system—including generation and abstention—is ready for final evaluation.

---

## Gold-source evaluation

Evaluation questions contain one or more expected source IDs.

Two source modes are supported:

### `any`

Any listed source is sufficient.

This is used when the same official fact is repeated across several valid sources.

Example:

```text
gold_source_ids:
SRC_ADM_001|SRC_ADM_002|SRC_ADM_004

gold_source_mode:
any
```

### `all`

All listed sources contribute required information.

This is used for genuinely multi-source questions.

Example:

```text
gold_source_ids:
SRC_HOUSING_001|SRC_HOUSING_004

gold_source_mode:
all
```

This distinction was added after discovering that the original evaluation treated alternative sources and jointly required sources in the same way.

---

## Retrieval metrics

The following metrics are recorded:

### Hit@k

Whether at least one acceptable gold source appears in the top `k` results.

Reported at:

* Hit@1
* Hit@3
* Hit@5

### Mean Reciprocal Rank

Rewards placing the first correct source as high as possible.

Examples:

```text
Correct at rank 1 → 1.0
Correct at rank 2 → 0.5
Correct at rank 3 → 0.333
Correct at rank 4 → 0.25
Correct at rank 5 → 0.20
```

### Recall@5

Measures how much of the required gold-source set appears in the top five.

For `gold_source_mode=any`, retrieving any acceptable source is complete.

For `gold_source_mode=all`, every required source contributes to recall.

### Metadata metrics

The project also reports:

* Source-group accuracy@3
* Programme accuracy@3
* Faculty accuracy@3
* External-source accuracy@3

These metrics help distinguish an exact-source failure from a broader routing failure.

---

## Chunking experiment

Two configurations were evaluated with the same:

* Corpus
* Embedding model
* FAISS index type
* Metadata
* Query router
* Candidate pool
* Development questions
* Source-diversification policy

### Compact

```text
Maximum chunk size: 220 tokens
Overlap: 40 tokens
Chunk count: 3,414
```

### Broad

```text
Maximum chunk size: 350 tokens
Overlap: 60 tokens
Chunk count: 2,057
```

---

## Final development-set results

| Metric                     |     Compact |        Broad |
| -------------------------- | ----------: | -----------: |
| Hit@1                      |  **82.86%** |       77.14% |
| Hit@3                      | **100.00%** |       97.14% |
| Hit@5                      | **100.00%** |       97.14% |
| MRR@5                      |  **90.48%** |       86.19% |
| Recall@5                   |  **99.05%** |       96.19% |
| Source-group accuracy@3    | **100.00%** |       97.14% |
| Programme accuracy@3       |     100.00% |      100.00% |
| Faculty accuracy@3         |     100.00% |      100.00% |
| External-source accuracy@3 |     100.00% |      100.00% |
| Mean retrieval latency     |    26.67 ms | **24.23 ms** |
| Median retrieval latency   |    19.44 ms | **17.19 ms** |

Question-level reciprocal-rank comparison:

| Result       | Questions |
| ------------ | --------: |
| Compact wins |         4 |
| Broad wins   |         0 |
| Ties         |        31 |

The approximately two-millisecond latency advantage of broad was not considered meaningful because both configurations are already effectively instantaneous compared with language-model generation.

---

## Selected retrieval configuration

The selected retrieval configuration is:

```text
compact
```

Frozen parameters:

* Embedding model: `intfloat/multilingual-e5-small`
* Embedding dimension: 384
* Maximum chunk size: 220 tokens
* Chunk overlap: 40 tokens
* Vector index: FAISS `IndexFlatIP`
* Similarity: cosine similarity through L2-normalised vectors
* Candidate pool: 40
* Maximum chunks per source: 1
* Evaluation retrieval depth: top 5
* Planned answer-generation depth: top 4

The compact configuration was selected because it:

* Achieved higher Hit@1
* Achieved perfect Hit@3 on the answerable development set
* Achieved perfect Hit@5 on the answerable development set
* Achieved higher MRR
* Achieved higher multi-source recall
* Achieved perfect source-group accuracy
* Won four question-level comparisons
* Lost no question-level comparison to broad
* Produced more focused passages for later answer generation
* Maintained negligible retrieval latency

The holdout set was not used during this selection.

---

## Problems encountered and fixes

### 1. Duplicate chunks occupied top-k positions

#### Problem

Earlier runs allowed two chunks from the same source.

This produced result lists such as:

```text
SRC_ADM_002
SRC_ADM_002
SRC_ADM_001
SRC_ADM_004
```

Repeated or near-identical chunks consumed valuable top-k positions and pushed other valid sources downward.

#### Fix

The final configuration uses:

```text
max_chunks_per_source = 1
```

This improved source diversity and moved relevant alternative sources into the top three.

---

### 2. Incomplete gold-source labels

#### Problem

Some official facts appeared on more than one valid Uni.lu page, but the evaluation initially listed only one source as correct.

For example, the requirement to demonstrate competence in programme teaching languages appeared across:

* Main admissions information
* Admission criteria
* Proof of language skills

A correct retrieval was therefore marked as a failure.

#### Fix

Gold sources were audited against the official corpus.

Questions now support:

```text
gold_source_mode = any
```

for alternative valid sources and:

```text
gold_source_mode = all
```

for genuinely multi-source answers.

The original evaluation files were backed up so the correction remains auditable.

---

### 3. Programme-catalogue routing failure

#### Problem

The question:

```text
Where can I compare the University's Bachelor and Master study programmes?
```

did not initially route to the programme catalogue.

The router recognised singular `programme` forms but did not reliably recognise plural and catalogue-navigation intent.

#### Fix

The router was expanded to recognise explicit catalogue intent, including:

* Compare programmes
* Programme overview
* Programme catalogue
* Browse programmes
* List programmes

The University-wide Study Programme Overview is boosted only for genuine catalogue-navigation questions.

---

### 4. Catalogue boosting was initially too broad

#### Problem

A first routing correction boosted programme overview pages whenever the query contained terms such as `programme` or `programmes`.

This incorrectly affected questions such as:

* How many programmes can I apply for?
* Does the University offer a spring or summer intake?
* Which official document contains programme-specific regulations?

Those questions concern admissions or regulations, not programme browsing.

#### Fix

Catalogue intent was separated into an explicit Boolean routing decision.

The catalogue boost now fires only when the question clearly asks to compare, browse, find or list programmes.

Generic appearances of the word `programme` do not trigger the boost.

---

### 5. Admissions-intake routing failure

#### Problem

A broad retrieval run misclassified the question:

```text
Does the University offer a spring or summer intake for
Bachelor and Master programmes?
```

as a programme-catalogue question.

It consequently retrieved only programme pages and missed the main admissions source.

#### Fix

The admissions routing vocabulary was expanded with:

* Application cycle
* Intake
* Annual intake
* Spring intake
* Summer intake
* September intake

This distinguishes programme availability from admissions-cycle questions.

---

### 6. Regulatory-document routing failure

#### Problem

For the question:

```text
Which official document contains programme-specific regulatory
provisions for Bachelor programmes for 2026–2027?
```

the correct Bachelor Annex had the highest raw semantic score.

However, programme metadata boosts promoted programme pages above it.

This demonstrated that:

* The chunk was correct
* The embedding was correct
* FAISS found the correct candidate
* The metadata reranker overturned the correct semantic ranking

#### Fix

Explicit regulatory intent was added for phrases such as:

* Official document
* Study regulations
* Regulatory provisions
* Bachelor annex
* Master annex
* General study rules
* Legal basis

Regulatory-document intent takes priority over generic programme terms.

---

### 7. Candidate-pool experiment

#### Baseline

```text
candidate_k = 40
```

#### Experiment

```text
candidate_k = 100
```

#### Result

Increasing the candidate pool did not improve source recall.

Instead, it exposed additional incorrectly boosted programme candidates. This reduced:

* Hit@3
* MRR
* Source-group accuracy

The experiment showed that the problem was not insufficient FAISS search coverage. `IndexFlatIP` had already found the correct semantic candidate. The problem was metadata reranking.

#### Decision

The final configuration retains:

```text
candidate_k = 40
```

The rejected experiment is documented to prevent repeating the same tuning path without new evidence.

---

### 8. Compact versus broad trade-off

Broad chunks preserve more surrounding context but combine more concepts into each vector.

Compact chunks provide more precise retrieval units but create a larger index.

The evaluation showed that broad chunks:

* Did not improve programme or faculty accuracy
* Had lower Hit@1
* Had lower Hit@3
* Had lower Hit@5
* Had lower MRR
* Had lower Recall@5
* Won no question-level comparison

Its small latency advantage was not sufficient to justify lower retrieval quality.

The compact configuration was therefore frozen.

---

## Remaining retrieval limitation

The compact configuration achieved:

```text
Recall@5 = 99.05%
```

rather than 100%.

This indicates that at least one genuinely multi-source question retrieved some, but not every, required source within the top five.

This is different from an exact-source Hit@5 failure: every development question retrieved at least one correct source, but one multi-source question was not completely covered.

Possible future improvements include:

* Dynamic candidate expansion when fewer than five unique sources remain
* Hybrid lexical and dense retrieval
* Lightweight reranking
* Query decomposition for explicitly multi-part questions
* Neighbouring-chunk expansion after initial source selection

These changes were not added to the frozen baseline because the current compact retriever already performs strongly, and additional complexity should be supported by holdout evidence.

---

## Latency

Final compact retrieval latency:

```text
Mean:   26.67 ms
Median: 19.44 ms
```

The embedding model was loaded on `cuda:0` during the recorded experiment.

The project uses the `faiss-cpu` package for exact vector search. This is sufficient because the corpus contains only a few thousand vectors, and retrieval is already well below one tenth of a second.

Model-loading time is not included in the per-question retrieval latency because the model is loaded once and reused.

---

## Recorded retrieval environment

The selected compact index was built with:

```text
sentence-transformers: 5.6.0
transformers:          5.12.1
torch:                 2.12.1+cu126
faiss-cpu:             1.14.3
numpy:                 2.5.0
```

Index details:

```text
Index type:             IndexFlatIP
Embedding dimension:    384
Normalised embeddings:  true
Indexed chunks:         3,414
Indexed documents:      62
Embedding batch size:   32
Model maximum length:   512
```

The build manifest and content hashes are stored with the index for reproducibility.

---

## Running the retrieval pipeline

### Validate the collected corpus

```bash
python scripts/004_validate_collection.py
```

### Build both chunking experiments

```bash
python scripts/008_build_chunks.py --config all
```

### Generate manual chunk-review files

```bash
python scripts/009_inspect_chunks.py --config compact
python scripts/009_inspect_chunks.py --config broad
```

### Build indexes

```bash
python scripts/010_build_index.py --config compact
python scripts/010_build_index.py --config broad
```

### Search interactively

```bash
python scripts/011_search_index.py \
  --config compact
```

### Search one question

```bash
python scripts/011_search_index.py \
  --config compact \
  --candidate-k 40 \
  --max-chunks-per-source 1 \
  --query "How do I re-enrol for the next semester?"
```

### Run compact development evaluation

```bash
python scripts/012_run_retrieval_eval.py \
  --config compact \
  --candidate-k 40 \
  --max-chunks-per-source 1
```

### Run broad development evaluation

```bash
python scripts/012_run_retrieval_eval.py \
  --config broad \
  --candidate-k 40 \
  --max-chunks-per-source 1
```

### Compare retrieval runs

```bash
python scripts/013_compare_retrieval_runs.py
```

---

## Retrieval outputs

Chunk outputs:

```text
data/processed/chunks_compact.jsonl
data/processed/chunks_broad.jsonl
data/processed/chunking_stats_compact.json
data/processed/chunking_stats_broad.json
data/processed/chunk_review_compact.md
data/processed/chunk_review_broad.md
```

Index outputs:

```text
faiss_index/compact/index.faiss
faiss_index/compact/embeddings.npy
faiss_index/compact/metadata.jsonl
faiss_index/compact/build_manifest.json
```

Evaluation outputs:

```text
results/retrieval_compact_per_question.csv
results/retrieval_compact_summary.json
results/retrieval_compact_error_analysis.csv
results/retrieval_broad_per_question.csv
results/retrieval_broad_summary.json
results/retrieval_broad_error_analysis.csv
results/retrieval_comparison_summary.json
results/retrieval_comparison_per_question.csv
```

---

## Current limitations

### Development-set tuning

The retrieval system was tuned using the development questions.

The 100% compact Hit@3 score must therefore be interpreted as development-set performance, not an unbiased estimate of performance on unseen questions.

### No unanswerable questions in the current retrieval evaluation

The evaluated development file contains:

```text
35 answerable questions
0 unanswerable questions
```

Therefore:

```text
unanswerable_top_1_score_mean = 0.0
```

is only a default empty-set value. It does not demonstrate successful abstention.

A separate unsupported-question development set is still required before a retrieval-confidence or abstention rule can be designed.

### No final answer evaluation

The current metrics evaluate source retrieval only.

They do not yet measure:

* Answer correctness
* Answer completeness
* Groundedness
* Citation correctness
* Hallucination
* Abstention
* User-interface quality

### Source freshness

The assistant will answer from the collected source snapshot.

University and government information may change, including:

* Application deadlines
* Re-enrolment periods
* Fees
* Programme structures
* Housing procedures
* Residence requirements
* Academic regulations

Every final answer should display its source and the source snapshot date where relevant.

### Prototype status

This project is not an official University of Luxembourg service.

Users must confirm consequential decisions using the linked official source or the relevant University/public authority.

---

## Next development stage

The next stage will add:

1. A grounded prompt template
2. Groq-based answer generation
3. English answers from English and French evidence
4. Source attribution
5. Page-aware PDF citations
6. Unsupported-question abstention
7. Streamlit interface
8. Retrieval-debug panel
9. Final answer-quality evaluation
10. Frozen holdout evaluation

The selected compact retriever will remain unchanged unless final system evaluation exposes a reproducible defect.

---

## Final frozen retrieval configuration

```text
Configuration name:             compact
Embedding model:               intfloat/multilingual-e5-small
Embedding dimension:           384
Maximum chunk size:            220 tokens
Chunk overlap:                 40 tokens
FAISS index:                   IndexFlatIP
Similarity:                    cosine via L2-normalised inner product
Candidate pool:                40
Maximum initial chunks/source: 1
Evaluation retrieval depth:    5
Answer-generation depth:       4
```

This configuration is the retrieval baseline for the remainder of the project.
