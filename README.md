# TransformerProcessing

Builds a **navigable index** of a document: splits it into sections, then writes a short
summary of each, so an agent or a person can find the right section without reading the whole
file.

Input is Markdown converted from a PDF or DOCX. Output is an outline and a per-section summary.

## Quick start

    pip install -r requirements.txt

    mkdir -p inputs/contracts
    cp examples/fixture_amendment.md inputs/contracts/

    python outline_only.py       # outlines for everything under inputs/ -- no model, ~1s
    DEVICE=cuda python run_all.py    # outlines AND summaries -- needs the model
    python group_outlines.py out     # collect the outlines by folder

`outline_only.py` uses no model at all: an outline is the section rules and nothing else.
Only the summaries need a GPU.

## Outline format

A section the document gives no heading for is written as `no header` rather than dropped --
if a block of text has no heading, that is a fact about the document. Subsections are only
ever those the document itself numbers; none are inferred. An execution block becomes a
single `signature block` entry rather than one section per `By:` line.

    no header
    First party Integration with Customer
    Business User Flexibility, Audience Segmentation and Activation
    Build and Activate Audiences
    Acceptance Period Timing
    signature block

## How sections are found

The mistake this replaced was applying every heading rule to every document. A lease has
hundreds of numbered clauses; a bill and a quarterly report have almost none. So
`pipeline/survey.py` first counts what structure signals are present and names a **regime**,
and only the rules that fit that regime run:

| regime     | chosen when                                    | rules enabled                        |
|------------|------------------------------------------------|--------------------------------------|
| `clauses`  | 20+ numbered clauses, or 5+ articles           | markdown, bold, clause, decimal      |
| `headings` | 5+ markdown headings at 0.16+ per KB           | markdown, bold                       |
| `sparse`   | markdown headings present but too sparse to be the backbone | markdown, bold, label, block-lead |
| `fields`   | 10+ label/value lines and no numbering         | markdown only                        |
| `labels`   | 3+ colon-terminated topic labels, nothing else | markdown, bold, label                |
| `flat`     | no signal at all                               | everything, including shape inference |

Signals that matter beyond the counts:

- **Heading density, not count.** Seven headings in a 4 KB memo are its skeleton; seven in a
  90 KB slide deck are incidental, because the deck's real titles are set in large type and
  arrive as a bare line.
- **Numbering restarts.** A contract numbers its clauses once and runs up. A lease's exhibits
  each restart at 1. Counting the restarts tells an untitled clause from a list item.
- **Signature keys are not structure.** `By:` / `Name:` / `Title:` appear in every signed
  contract and say nothing about how the body is organised.
- **Block boundaries.** Each `<a id=...>` marks a visual block lifted off the page, so "first
  line of a block, with a body under it" is where an unmarked title lives.

## Summaries

`facebook/bart-large-cnn`, deterministic (`do_sample=False`, `num_beams=4`), fp16 on GPU.
Long sections exceed its 1,024-token limit and are split into parts, each summarised, then
the part summaries summarised together. Nothing is truncated.

**Tables are not summarised by the model.** BART invents figures -- it returned `$353.00` for
`$353,377` -- so `pipeline/tables.py` computes table summaries arithmetically from the cells.
Prose goes to the model; numbers do not.

## Layout

    outline_only.py       outlines only, no model
    run_all.py            outlines + summaries, one output per input file
    group_outlines.py     collect finished outlines by folder
    compare_run.py        sectionise -> summarise -> check; the pipeline entry point
    outline.py            render an outline in nested form

    pipeline/survey.py     count structure signals, pick the regime
    pipeline/sections.py   the section rules
    pipeline/tables.py     arithmetic table summaries
    pipeline/figures.py    read signatures, dates and labels out of figure blocks
    pipeline/summarise.py  how the model is called
    pipeline/chunking.py   sentence splitting and part splitting
    pipeline/categories.py section category labels
    pipeline/verify.py     faithfulness checks

## Notes

`examples/fixture_amendment.md` is a synthetic document -- invented parties and dates -- with
no headings, no numbering, colon-terminated topic labels and a two-party execution block. That
shape used to yield zero sections, so it exercises the label regime, the `no header` entry and
signature detection in one pass. No customer documents are in this repository.
