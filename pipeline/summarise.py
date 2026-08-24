"""Stage 4 -- how each transformer is called.

Four models, chosen because they represent genuinely different strategies rather than four
points on one curve:

  BART-large-CNN     writes a new sentence. 1,024-token limit, so long sections are split.
  DistilBART-CNN     the same, distilled: ~1.8x faster for about a point less usefulness.
  LongT5-16384       reads 16,384 tokens, so it never splits. An alternative to splitting,
                     not a cheaper version of it.
  MiniLM extractive  writes nothing -- quotes the real sentence closest to the section's
                     average. The floor: it cannot invent, and it is rarely informative.

Settings are identical across all four apart from the reading limit and the beam count, and
nothing is sampled, so a re-run reproduces the same summaries word for word.

  do_sample=False          makes the run reproducible
  min_new_tokens=12        stops a model answering in three words
  no_repeat_ngram_size=3   stops the repetition loops these models fall into on tables
  num_beams=4 (2 for LongT5, where 4 was not worth the time)

MEASURED RESULTS on the corpus this was built for, weighting usefulness 50%, faithfulness
30%, speed 20%:

  BART-large-CNN        7.2/10   5.64 s per section   recommended
  DistilBART-CNN        7.1/10   3.15 s               0.1 behind and 1.8x faster
  MiniLM extractive     6.5/10   0.07 s               scores 3/10 on usefulness
  LongT5-16384          4.0/10   2.37 s               the only one that invents content

Two findings worth carrying forward. Splitting long sections BEAT reading them whole:
LongT5 read a 3,089-word section in one pass and invented a detail that was not in the
document, while BART read the same section in five parts and every part summary was
accurate. And quoting is safe but useless: the extractive model never invents, because
every word is quoted, but "most typical sentence" in a formal document is the boilerplate.
"""
import os
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from .chunking import SENT, split_parts

# Run on the GPU when there is one. A T4 is sm_75, which does fp16 but not bf16, so fp16 is
# the only half precision available -- and it is the point of going to the GPU at all: BART's
# 400M parameters at fp32 on one CPU thread is what made the full 33-file run take hours.
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    torch.set_num_threads(1)

MODELS = [("BART-large-CNN", "facebook/bart-large-cnn", 1024, "abstractive"),
          ("DistilBART-CNN", "sshleifer/distilbart-cnn-12-6", 1024, "abstractive"),
          ("LongT5-16384", "pszemraj/long-t5-tglobal-base-16384-book-summary", 16384,
           "abstractive"),
          ("MiniLM-L6 extractive", "sentence-transformers/all-MiniLM-L6-v2", 512,
           "extractive")]


def load(repo, kind):
    tok = AutoTokenizer.from_pretrained(repo)
    if kind != "abstractive":
        return tok, AutoModel.from_pretrained(repo).to(DEVICE).eval()
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=DTYPE).to(DEVICE).eval()
    if "long-t5" in repo:
        # This checkpoint stores one embedding table under `shared.weight` but declares
        # tie_word_embeddings=False. transformers therefore randomly initialises the
        # encoder and decoder embeddings and throws the real ones away, and the model
        # emits word salad on every input. Measured: the real table has standard
        # deviation 10.09, the randomly initialised one 1.0002.
        #
        # Worth knowing because the failure is indistinguishable from "this model is bad".
        # It still came last after the fix -- but for inventing content, which is a real
        # finding, rather than for a loading bug, which was ours.
        mdl.encoder.embed_tokens.weight = mdl.shared.weight
        mdl.decoder.embed_tokens.weight = mdl.shared.weight
    return tok, mdl


# How long a run of the INPUT's own words the model is forbidden to reproduce. This is the whole
# fix for "the summaries are just the first two sentences": BART-large-CNN was trained on
# CNN/DailyMail, where the opening lines ARE the summary, so on a short formal clause it copies.
# Measured over 35 sections of five document types, mean longest-verbatim-run as a fraction of the
# summary:
#
#     unconstrained   0.696      what shipped -- and 61.6% of summaries introduced no new word at all
#     n = 6           0.264      copying down 62%, 6 of 35 sections newly flagged
#     n = 5           0.239      text starts to mangle: ("*Ag Agreement*")
#     n = 4           0.168      11 of 35 flagged
#     n = 3           0.117      6.6 novel words per summary, and drifts
#
# Six is the setting, and the reason is the mangling rather than the flag count: at five and below
# the model is forced off the source's wording so hard that it fuses and splits words to satisfy
# the constraint. length_penalty was swept alongside and had NO effect at these token limits.
#
# This is a real trade. Copying is safe and uninformative; abstraction is informative and has to be
# checked. Nothing here decides that -- pipeline/verify.py still runs on every summary, and it is
# known to over-flag paraphrase, so a flag is a prompt to read rather than a verdict.
# DEFAULT OFF. Three independent blind judges scored the forced-abstractive variants at 8, 20 and
# 15 inventions out of 35 against 2, 7 and 6 for the unconstrained model, and all three found the
# gradient monotone: no item had a more abstractive candidate that was both cleaner and more
# informative. Kept switchable because the mechanism is sound and a future model may survive it.
ENC_NO_REPEAT = int(os.environ.get("ENC_NO_REPEAT", "0"))
# The token-exemption experiment. Measured byte-identical to the blunt ban, so OFF:
# an n-gram ban blocks the CONTINUATION of a run, so exempting "$" and "19" while
# "billion" stays banned still forces the model off the span. See abstractive.py.
PROSE_ABSTRACT = int(os.environ.get("PROSE_ABSTRACT", "0"))


# How long a summary should be, as a share of the section it summarises, and the floor and
# ceiling on that. Measured on this corpus: with a FIXED 70-token budget the summary came out at
# 21 words for a 50-word section and 34 words for a 4,000-word one, so the share of the section
# actually covered fell from 67% to 10% and compression ran to 24x. A summary that says a tenth
# of a long clause is not a summary of it.
#
#   share 0.30 gives   100 words -> ~40   400 -> ~120   1000 -> ~195 (at the ceiling)
#   i.e. roughly 3-5x compression across the range instead of 1.6x to 24x.
SUMMARY_SHARE = float(os.environ.get("SUMMARY_SHARE", "0.30"))
SUMMARY_MIN_TOK = int(os.environ.get("SUMMARY_MIN_TOK", "48"))
SUMMARY_MAX_TOK = int(os.environ.get("SUMMARY_MAX_TOK", "260"))
TOK_PER_WORD = 1.35            # BART's tokeniser on legal English, measured on this corpus


def budget(section_words, longer=False, input_words=None):
    """Token budget for a summary of a section this long.

    Bounded by the INPUT as well as by the section. A summary can never be longer than the text it
    summarises, and asking for one is not merely wasteful -- it is how a summariser is made to
    hallucinate. A 14-part split of one Tesla clause handed BART a fragment reading "(ii) multiplied
    by", the floor demanded 29 tokens from it, and the model padded to length with invented figures:
    "($1,000,000) (i.e. $1,500,000 is $1 million) ($1 million is $100,000)". Two amounts that appear
    nowhere in the document, produced purely by a minimum length.
    """
    if not section_words:
        cap = 80 if longer else 70
    else:
        want = int(section_words * SUMMARY_SHARE * TOK_PER_WORD)
        cap = max(SUMMARY_MIN_TOK, min(SUMMARY_MAX_TOK, want))
    if input_words:
        cap = min(cap, max(24, int(input_words * TOK_PER_WORD * 0.85)))
    return cap


def summarise(tok, mdl, repo, limit, text, longer=False, section_words=None):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=limit).to(DEVICE)
    iw = len(text.split())
    cap = budget(section_words, longer, input_words=iw)
    # the floor tracks the INPUT, never the ceiling. A fixed or ceiling-derived floor is what forced
    # a one-line fragment to produce thirty tokens of invented money.
    floor = max(8, min(12, iw // 2))
    kw = dict(max_new_tokens=cap, min_new_tokens=min(floor, max(8, cap - 4)),
              num_beams=4 if "long-t5" not in repo else 2,
              do_sample=False, no_repeat_ngram_size=3)
    # PROSE_ABSTRACT=n pushes the model off the source's sentence shapes while leaving every
    # fact-bearing token free to be copied exactly -- see pipeline/abstractive.py for why the
    # blunt encoder_no_repeat_ngram_size was abandoned after three blind judges measured it.
    if PROSE_ABSTRACT:
        from .abstractive import processors
        kw["logits_processor"] = processors(tok, enc["input_ids"], PROSE_ABSTRACT)
    elif ENC_NO_REPEAT:
        kw["encoder_no_repeat_ngram_size"] = ENC_NO_REPEAT
    g = mdl.generate(**enc, **kw)
    return tok.decode(g[0], skip_special_tokens=True).strip()


def extract(tok, mdl, text):
    """Quote the sentence nearest the section's mean embedding. Invents nothing."""
    sents = [x.strip() for x in SENT.split(text) if len(x.strip()) > 30]
    if not sents:
        return text[:180]
    vs = []
    with torch.inference_mode():
        for x in sents:
            e = tok(x, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vs.append(torch.nn.functional.normalize(v, dim=-1)[0])
    E = torch.stack(vs)
    c = torch.nn.functional.normalize(E.mean(0, keepdim=True), dim=-1)
    return sents[int(torch.argmax(E @ c.T))]


def run_model(name, repo, limit, kind, sections):
    """Summarise every section, splitting the ones over the model's limit."""
    tok, mdl = load(repo, kind)
    rows, splits = [], 0
    for s in sections:
        if s.get("container_only") or not s["text"].strip():
            rows.append({"summary": None, "parts": [], "n_parts": 0})
            continue
        if kind == "extractive":
            rows.append({"summary": extract(tok, mdl, s["text"]), "parts": [], "n_parts": 1,
                         "quoted": True})
            continue
        parts = split_parts(tok, s["text"], limit)
        if len(parts) == 1:
            rows.append({"summary": summarise(tok, mdl, repo, limit, s["text"]),
                         "parts": [], "n_parts": 1})
            continue
        splits += 1
        ps = [{"label": "part %d of %d" % (n + 1, len(parts)), "words": len(p.split()),
               "summary": summarise(tok, mdl, repo, limit, p)}
              for n, p in enumerate(parts)]
        # the section summary is built from the part summaries: one step further from the
        # source than everything else here, which is why the parts are kept and shown
        stitched = summarise(tok, mdl, repo, limit,
                             " ".join(p["summary"] for p in ps), longer=True)
        rows.append({"summary": stitched, "parts": ps, "n_parts": len(ps)})
    return {"repo": repo, "kind": kind, "input_limit": limit,
            "sections_needing_split": splits, "rows": rows}
