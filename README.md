# Transformer Processing — building a navigable index of a document

Turn a long document into an index an agent can navigate: every section, what kind of thing
is in it, and a summary of each — with **no LLM, no API calls, and nothing leaving the
machine**. Four transformer models are compared on identical input.

```bash
pip install -r requirements.txt
python run.py examples/sample_sow.md --no-summaries   # structure + labels, <1 second
python run.py examples/sample_sow.md                  # add summaries from all four models
```

**Want one file instead of the package?** [`transformer_index.py`](transformer_index.py) is
the whole thing — all five stages, the verification, and the runner — in a single
self-contained script with no local imports. Same code, same output. Download that one file
and run it:

```bash
python transformer_index.py yourdoc.md --no-summaries
```

`--no-summaries` deliberately imports neither torch nor transformers, so the structure and
label check runs on a bare Python install.

Start with `--no-summaries`. It answers the question that matters first on a new document:
**did the section rules find the structure at all?** If the section count is wrong, no
amount of summarising will fix it.

---

## Why this exists

A contract is long and most of it is irrelevant to any given question. When someone asks
*"when does this end?"* or *"are we allowed to subcontract?"*, today a person reads until
they find it. If every section carries a label and a summary, that stops being necessary —
you open the two sections that matter and ignore the other fifty.

## The pipeline

A transformer never sees a document. It sees whatever a rule handed it. Those rules decide
the shape of everything downstream, so they are the substance of this repository — the model
call is four lines.

| Stage | What it does | Where |
|---|---|---|
| 1 | document → sections (three heading rules) | `pipeline/sections.py` |
| 2 | sections → subsections (`4.1` under `4`) | `pipeline/sections.py` |
| 3 | split any section over the model's reading limit | `pipeline/chunking.py` |
| 4 | summarise each section and each part | `pipeline/summarise.py` |
| 5 | label each section: money / dates / parties / permissions / obligations | `pipeline/categories.py` |
| — | check the summaries against the source, mechanically | `pipeline/verify.py` |

Stages 3 and 4 make no assumptions about the kind of document. Stages 1, 2 and 5 do — see
*Other document types* below.

Every rule is documented in the docstring of the file that runs it, including the three
that were **wrong before they were right**. Those are recorded rather than quietly fixed,
because each one failed in a way that looked like a bad model:

- **The uppercase test.** Heading detection originally required a line to be over 70%
  uppercase. It kept `TIMELINE` and threw away real numbered clauses written in Title Case.
  Case was the wrong discriminator; length and shape is the right one.
- **The 40-character floor.** Headings with almost no body were dropped as noise. They are
  not noise — a heading that runs straight into `4.1` is what makes `4.1` a subsection.
  Dropping it left subsections attached to the wrong parent. They are kept and marked
  `container_only`.
- **LongT5's embeddings.** That checkpoint stores one embedding table under `shared.weight`
  but declares `tie_word_embeddings=False`, so the library randomly initialises the
  embeddings and the model emits word salad. Real table standard deviation 10.09, randomly
  initialised 1.0002. Two lines fix it.

## Results

Measured on a 44-section document, weighting usefulness 50%, faithfulness 30%, speed 20%:

| Model | Tells you what the section is | Sticks to the source | Speed | Sec/section | Overall |
|---|---|---|---|---|---|
| **BART-large-CNN** | 8/10 | 8/10 | 4/10 | 5.64 s | **7.2** |
| DistilBART-CNN | 7/10 | 8/10 | 6/10 | 3.15 s | **7.1** |
| MiniLM-L6 extractive | 3/10 | 10/10 | 10/10 | 0.07 s | **6.5** |
| LongT5-16384 | 4/10 | 2/10 | 7/10 | 2.37 s | **4.0** |

**BART is the recommendation. DistilBART is 0.1 behind and 1.8× faster** — at scale that is
the one to pick. Two findings are more useful than the ranking:

**Splitting long sections beat reading them whole.** LongT5 read a 3,089-word section in one
pass and invented a detail that was not in the document. BART read the same section in five
parts and every part summary was accurate. Reading more of the document made the output
worse, not better.

**Quoting is safe but useless.** The extractive model cannot invent, because every word is
quoted. But "most typical sentence" in a formal document is the boilerplate, so it reliably
returns the least informative line in the section.

## What the verification found — read this before trusting any output

`pipeline/verify.py` checks summaries against their source three ways, because each catches
something the others structurally cannot. On 36 section summaries and 21 part summaries it
found **nine wrong words**.

**Not one was an invented fact.** No fabricated obligation, party, amount or deadline
anywhere. Every one was a corrupted *label*: a party name pluralised, a space inserted
inside an acronym, an acronym's case changed, and — twice — a clause number that either does
not exist or points at the wrong clause.

That distinction is the finding, because it tells you exactly how far to trust the output:

- **Safe** — using a summary to decide which section to open.
- **Not safe** — quoting a summary, or letting anything act on a clause number inside one.

We under-counted this at first, and the reason is instructive. The initial pass checked only
section-level summaries, in lower case, and found three problems. It could not see the four
inside part summaries, could not see an acronym written `AbCd` where the source says `ABCD`, because case was folded away, and
could not see `13.1` written for `13.1.1` because `13.1` genuinely does appear in the
section. Three separate checks are needed; one is not enough.

### A summary can be accurate and still useless

On 7 of 36 sections the summary had nothing wrong with it and still would not help anyone
decide whether to open the section — it latched onto the section's opening line, or a
footnote, instead of its substance. **The worst case was the pricing section**, which
contained the entire fee schedule; the summary described which month the financial year
starts and said invoices are paid per the master agreement. Every word true, not one number.

This is why the category labels exist alongside the summaries. The word-search rules tag
that section `money` from the amounts in the table even though the summary never mentions
money. Where the summary fails, the label still routes you correctly. Neither is sufficient
alone.

## Other document types — will this work on yours?

Honestly: **partly, and it is worth knowing which part.** Everything above was measured on
supplier agreements. It has not been tested on anything else.

**The document-agnostic half carries over unchanged.** Stage 3 (splitting to fit a reading
limit) and stage 4 (the model calls) assume nothing about the content.

**Amendments should mostly work.** Same clause structure, same numbering, same vocabulary.
One specific risk: in an amendment the clause number *is* the payload — *"Section 4.2 is
amended to read…"* — and the verification above caught clause-number corruption twice in 36
sections. On a contract that is an annoyance; on an amendment it changes which clause you
think was amended. For amendments, extract the clause number by rule and never read it from
a summary.

**Invoices will not work, and it is not a tuning problem.** An invoice has no clause
headings and no `4.1`-under-`4` hierarchy, so stages 1 and 2 have nothing to key on. And the
labels stop discriminating: `money` fires on 17% of sections in a supplier agreement, which
is exactly what makes it a useful filter, but it would fire on essentially every part of an
invoice — while `shall`/`must` would fire on almost none. A label that fires on everything
carries no information.

The deeper point is that an invoice is not a document you summarise. It is a set of fields
you extract — invoice number, PO, due date, line items, tax, total — and the right output is
a filled form, not a paragraph. That is a different pipeline, not new rules on this one.

**Format is a separate variable from type.** Invoices usually arrive as PDFs or scans rather
than clean text, and parser quality drops measurably on the same document going from
Markdown to PDF. Test document type and document format separately.

## What is not in this repository, and why

**No documents, and no output derived from them.** Everything this was measured on is real
customer agreements containing party names, fee schedules, named signatories and working
email addresses. The derived JSON is worse than the source in this respect, not better — it
reassembles the document section by section. None of it is here.

`examples/sample_sow.md` is **entirely fictional**, written for this repository to exercise
the rules. Point `run.py` at a document of your own to reproduce the pipeline end to end.

Nothing in this pipeline calls an LLM or sends text anywhere. All four models run locally
from downloaded weights, single-threaded (`torch.set_num_threads(1)`). The only network
traffic is the initial weight download, inbound.
