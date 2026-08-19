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
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from .chunking import SENT, split_parts

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
        return tok, AutoModel.from_pretrained(repo).eval()
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.float32).eval()
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


def summarise(tok, mdl, repo, limit, text, longer=False):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=limit)
    g = mdl.generate(**enc, max_new_tokens=80 if longer else 70, min_new_tokens=12,
                     num_beams=4 if "long-t5" not in repo else 2,
                     do_sample=False, no_repeat_ngram_size=3)
    return tok.decode(g[0], skip_special_tokens=True).strip()


def extract(tok, mdl, text):
    """Quote the sentence nearest the section's mean embedding. Invents nothing."""
    sents = [x.strip() for x in SENT.split(text) if len(x.strip()) > 30]
    if not sents:
        return text[:180]
    vs = []
    with torch.inference_mode():
        for x in sents:
            e = tok(x, return_tensors="pt", truncation=True, max_length=512)
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
