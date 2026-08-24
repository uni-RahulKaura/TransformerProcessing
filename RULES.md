# TransformerProcessing

Section detection, abstractive summarisation and topic labelling for commercial documents
(agreements, statements, quarterly reports, correspondence). Rules and tooling only — no
customer documents, no run output, no credentials. See [What is deliberately absent](#what-is-deliberately-absent).

## What this does

A document goes in as Markdown. Three things come out:

1. **An outline** — the document's sections, nested, with the untitled ones named rather than dropped.
2. **A summary per section** — reworded by the model, not the section's own sentences quoted back.
3. **Topic labels per section** — up to three, with a confidence, from a 116-topic vocabulary.

## The rules

| file | what it decides |
|---|---|
| `pipeline/survey.py` | which rule set a document gets. A lease and a phone bill do not have the same structure, and applying lease rules to the bill produced 110 sections for a document with a dozen. |
| `pipeline/sections.py` | where a section begins. Markdown headings, bold leads, clause numbering, decimal numbering, capitalised titles, block leads, and titled lettered sub-items. |
| `pipeline/chunking.py` | how a long section is split so all of it is summarised, not just its opening. Splits on lettered items when the document has them. |
| `pipeline/summarise.py` | the generation budget. Scales with the section and is bounded by the input. |
| `pipeline/safe_abstractive.py` | whether a rewrite is safe to publish, and what to do when it is not. |
| `pipeline/polarity.py` | whether a summary reversed a duty the section imposes. |
| `pipeline/topics.py` | topic labelling, and the grounding gate that suppresses a label whose vocabulary the section never uses. |
| `pipeline/categories.py` | the five original regex facets: money, dates, parties, permissions, obligations. |
| `pipeline/verify.py`, `tables.py`, `figures.py` | faithfulness checks, table arithmetic, figure descriptions. |

## Why a summary is sometimes refused

The summariser produces two drafts per section — one that leans on the source's wording and one
forced away from it — and `safe_abstractive.choose()` decides which is published. Every section
records the route it took, so the abstraction rate is a count rather than a claim:

| route | meaning |
|---|---|
| `abstractive` | reworded, clean on every check |
| `abstractive-repaired` | reworded; a figure or date absent from the section was deleted and the rewrite kept |
| `abstractive-depolarised` | reworded; a sentence that reversed the section's meaning was removed |
| `extractive-short` | the section is too short for a rewrite to shorten it, so it is quoted |
| `faithful` | the rewrite failed a check, so the closer draft was published |
| `verbatim` | neither draft was safe, so the section is quoted directly |

The defect that motivated the polarity check: a summary rendered *"the Recipient **shall be**
responsible for any breach"* as *"the Recipient **is not** responsible"*. Every word of the wrong
version appears in the section, so no figure, name or spelling check could see it.

A first version of that fix **deleted the stray `not` instead of dropping the sentence**, and on
real text it turned three correct prohibitions into mandates — worse than the defect it was
chasing. It is now drop-only, and never edits a negation. `test_polarity.py` pins that.

## Topic labelling

116 topics from three sources, all going through the same grounding gate:

- **54 derived** from the corpus's own section titles — clustered with `BAAI/bge-small-en-v1.5`,
  Ward linkage, consolidated by hand.
- **41 from CUAD** — the canonical clause categories of the
  [Contract Understanding Atticus Dataset](https://github.com/TheAtticusProject/cuad), read from
  `category_descriptions.csv` rather than transcribed, so the wording is theirs. Non-Compete,
  Exclusivity, Cap on Liability, Source Code Escrow, ROFR/ROFO/ROFN and 36 others.
- **21 for statements** — written against the section titles telecom bills actually print.

Scoring is zero-shot NLI with `facebook/bart-large-mnli`: the section is the premise, each topic
becomes a one-sentence hypothesis. **The number beside a label is that entailment probability.**
Scores are independent, so they do not sum to 1 — a section can be 1.00 confidentiality and 0.69
data protection at once. **Confidence is not accuracy: the model can be 0.96 and wrong.**

The grounding gate is what makes the low band usable. A label is suppressed unless the section uses
one of that topic's own words. Without it, an NDA's parties clause scored 0.68 for *"a loan or
credit facility: advances, lenders, agents and repayment"*, and a two-word heading reading
`10 MISCELLANEOUS` scored 0.94 for confidentiality.

## Tests

```
python test_names.py      # 18 cases -- figure, date and name support
python test_polarity.py   # 17 cases -- reversed and dropped negations
python test_tidy.py       # 12 cases -- surface repair without breaking real words
```

Every case is taken from real output. The **false-alarm cases matter as much as the true ones**:
this check decides whether a section publishes a rewrite or a quotation, so a false alarm costs
exactly what the work was for. Cases that must NOT fire include an article added to a defined term
(`The Agreement`), a possessive (`TESLA's`), a defined term written with a space (`Sub Tenant`
against a lease defining `Subtenant`), a table of contents matched as one 20-word party name, and
`No.` before a digit read as a negation.

## Auditing

`judge/` builds blind audit packets and runs specialised judges over them. A judge sees the raw
source document and the published output, and is barred from the parser, from the internal route
tags, and from other judges' verdicts. Every high-severity finding then faces three independent
skeptics instructed to refute it, and survives only on a 2-of-3 majority.

- `make_packets.py` / `make_readable.py` — blind packets, and human-readable output
- `judge-workflow.js` — six remits over summaries: figures, obligations, entities, coverage,
  copying, topic accuracy
- `outline-workflow.js` — three remits over outlines: sections invented, sections missing, nesting
- `compare_runs.py` — one run against another, asserting rather than assuming that a
  summary-only change leaves the outline untouched
- `threshold_sweep.py` — what the abstraction rate would be at each short-section threshold,
  computed from a run's own output instead of re-running the corpus once per candidate

## What is deliberately absent

No customer documents, no run output, no answer keys, no credentials. The test corpus is executed
commercial agreements and telecom bills — named counterparties, addresses, account numbers,
DocuSign envelope IDs, and in the bills real subscriber names and phone numbers. Derived output is
no safer than the source, because a summary quotes the clause it summarises.

Example text inside comments and test fixtures has had party names replaced with neutral
equivalents. The linguistic property each case exercises is unchanged — a near-miss is still a
near-miss, a possessive still a possessive — but no counterparty is identifiable.

`.gitignore` enforces this. If you add a document directory, add it there first.

## Requirements

Python 3.11+, `transformers`, `torch`. Models are used offline (`HF_HUB_OFFLINE=1`):
`facebook/bart-large-cnn` for summaries, `facebook/bart-large-mnli` for topics,
`BAAI/bge-small-en-v1.5` for the clustering that produced the derived topics.

```bash
ABSTRACTIVE=1 ABS_N=6 TOPICS=1 TOPIC_GATE=1 python run_all.py
```

| variable | default | effect |
|---|---|---|
| `SHORT_WORDS` | 25 | sections at or below this are quoted rather than reworded; `0` disables |
| `SUMMARY_SHARE` | 0.30 | summary length as a share of the section |
| `TOPIC_MIN` | 0.55 | confidence floor for publishing a topic |
| `TOPIC_GATE` | 1 | the grounding gate; `0` disables it |
| `COVER` | 1 | split long sections so all of them is summarised |
