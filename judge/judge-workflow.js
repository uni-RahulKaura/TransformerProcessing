export const meta = {
  name: 'grade-33-blind',
  description: 'Five specialised blind judges per document over all 33, adversarial verification of every finding, then synthesis',
  phases: [
    { title: 'Judge',     detail: 'five specialised remits per document, each reading only the raw markdown' },
    { title: 'Verify',    detail: 'independent refuters per finding, majority rules' },
    { title: 'Synthesis', detail: 'one verdict per instrument across all documents' },
  ],
}

const ROOT = '/Users/rahulkaura/Documents/gap-format-prototype'

const BAN = `
HARD RULES -- breaking any of these invalidates your work:
* You may read ONLY two things: the raw Markdown file named in your packet, and the packet itself.
* You must NOT read, list, grep or open any of: ${ROOT}/indexer*, ${ROOT}/gpu-bundle*/pipeline,
  ${ROOT}/gpu-bundle*/*.py, ${ROOT}/corpus, ${ROOT}/judge/*.json, ${ROOT}/*.txt, ${ROOT}/BBC-DEMO,
  ${ROOT}/RUN-GPU*, any file matching *ground_truth*, *.truth.txt, *verdict*, or any other judge's output.
  Those contain the code that produced the summaries and other judges' opinions. Seeing either biases you.
* Judge ONLY against the raw Markdown. It is the sole authority. If the raw Markdown and the
  summary disagree, the raw Markdown is right by definition.
* Do not assume a summary is wrong because it is short, reworded, or omits detail. Rewording is
  intended. Omission is not a defect unless the omitted thing reverses or distorts what remains.
* Quote exactly. Every finding must carry the literal source text and the literal summary text.
  A finding you cannot quote from both sides is not a finding -- drop it.
* Report nothing rather than something you are unsure of. An empty findings list is a valid,
  useful answer and is preferred over a speculative one.
`

const FINDINGS = {
  type: 'object', additionalProperties: false,
  properties: {
    sections_checked: { type: 'integer' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          section_title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          claim: { type: 'string', description: 'one sentence: what is wrong' },
          quote_from_source: { type: 'string' },
          quote_from_summary: { type: 'string' },
          why_wrong: { type: 'string' },
        },
        required: ['section_title', 'severity', 'claim', 'quote_from_source', 'quote_from_summary', 'why_wrong'],
      },
    },
  },
  required: ['sections_checked', 'findings'],
}

const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['refuted', 'reason', 'confidence'],
}

const ABSTRACT = {
  type: 'object', additionalProperties: false,
  properties: {
    sections_checked: { type: 'integer' },
    verbatim_or_near_verbatim: { type: 'integer' },
    genuinely_reworded: { type: 'integer' },
    examples_of_copying: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { section_title: { type: 'string' }, copied_run: { type: 'string' } },
        required: ['section_title', 'copied_run'],
      },
    },
  },
  required: ['sections_checked', 'verbatim_or_near_verbatim', 'genuinely_reworded'],
}

const TOPICS = {
  type: 'object', additionalProperties: false,
  properties: {
    labels_checked: { type: 'integer' },
    wrong_labels: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          section_title: { type: 'string' }, topic: { type: 'string' },
          confidence: { type: 'number' }, why_wrong: { type: 'string' },
        },
        required: ['section_title', 'topic', 'why_wrong'],
      },
    },
    missing_labels: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { section_title: { type: 'string' }, should_be: { type: 'string' } },
        required: ['section_title', 'should_be'],
      },
    },
  },
  required: ['labels_checked', 'wrong_labels'],
}

const REMITS = [
  { key: 'figures', schema: FINDINGS, brief: `You check NUMBERS ONLY. For every figure in every published summary --
money, percentages, dates, durations, counts, section cross-references, addresses, quantities --
find it in the raw Markdown for that section and confirm it matches exactly. Report a finding when a
figure in the summary is absent from the section, altered, or attached to the wrong thing (right
number, wrong subject). Ignore wording entirely. Ignore missing figures unless their absence makes a
remaining figure misleading.` },

  { key: 'obligations', schema: FINDINGS, brief: `You check the DIRECTION OF DUTIES ONLY. For each summary, decide
whether it preserves exactly who must do what, who may do what, and who is forbidden from doing
what. Report a finding when the summary reverses a duty (must -> must not, may -> may not, and the
reverse), moves a duty from one party to the other, turns a conditional duty into an absolute one or
the reverse, drops an exception that changes what is permitted, or converts a prohibition into a
permission. Read exception language carefully: "may not be amended except in writing" is a
requirement to use writing, not a ban on amendment. This is the highest-consequence remit -- a
reversed obligation is worse than any other error, so be thorough, but quote both sides or drop it.` },

  { key: 'entities', schema: FINDINGS, brief: `You check WHO ONLY. For each summary, confirm every party name, defined
term and role is one the section actually uses, and is attached to the right side of the transaction.
Report a finding when the summary names a party the section does not contain, uses a defined term the
document never defines, swaps two parties' roles, or attributes an act to the wrong party. Treat
singular/plural and possessive forms of the same defined term as the same term -- that is not a
finding. A shortened form of a name the section contains is not a finding.` },

  { key: 'coverage', schema: FINDINGS, brief: `You check whether the summary MISREPRESENTS THE SECTION AS A WHOLE.
Not whether it is complete -- a summary is allowed to omit. Report a finding only when what was kept,
without what was dropped, leaves a reader with a materially wrong impression of the section: the
summary states a rule the section qualifies, describes a minor point as if it were the section's
subject, or reads as being about a different clause than the one it belongs to. Also report a
summary that is incoherent or garbled to the point of being unusable, quoting it.` },
]

// ---------------------------------------------------------------------------
const packets = (args && args.packets) || []
if (!packets.length) {
  log('no packets passed in args.packets -- nothing to judge')
}
log(`grading ${packets.length} documents with ${REMITS.length} specialised judges + abstractiveness + topics`)

function judgePrompt(remit, pk) {
  return `You are one of several independent judges. You have exactly one remit and you must stay inside it.

YOUR REMIT
${remit.brief}

${BAN}

WHAT TO DO
1. Read the packet: ${pk}
   It lists the document, the path to its raw Markdown, and for each section the section title plus
   the summary that was published for it, plus any topic labels.
2. Read the raw Markdown file the packet names. That file is the only authority.
3. For each section in the packet, locate that section in the raw Markdown yourself and apply your
   remit to its published summary.
4. Return your findings. Set sections_checked to the number you actually examined.

Return the structured object. Your text output is discarded -- only the structured result is used.`
}

const perDoc = await pipeline(
  packets,
  // ---- stage 1: five specialised judges on one document, concurrently
  (pk, _orig, i) => parallel([
    ...REMITS.map(r => () =>
      agent(judgePrompt(r, pk), {
        label: `${r.key}:${i + 1}`, phase: 'Judge', schema: r.schema,
      }).then(v => ({ remit: r.key, packet: pk, result: v }))),
    () => agent(`You judge ONE thing: is each published summary a genuine REWRITE of its section, or is it
the section's own sentences copied out?

This matters because the requirement is an abstractive summary -- reworded -- not an extractive one.
A summary that lifts a sentence unchanged fails the requirement even if every fact in it is correct.

Count a section as verbatim_or_near_verbatim when a run of 12 or more consecutive words in the
summary appears identically in the section, or when the whole summary is one of the section's own
sentences with only the opening words trimmed. Count it as genuinely_reworded when the sentence
structure or vocabulary differs even though the meaning is preserved. Give up to 8 concrete examples
of copying, quoting the copied run.

Do NOT judge accuracy here. A reworded summary that is factually wrong still counts as reworded.

${BAN}

Packet: ${pk}
Return the structured object.`, { label: `copying:${i + 1}`, phase: 'Judge', schema: ABSTRACT })
      .then(v => ({ remit: 'copying', packet: pk, result: v })),
    () => agent(`You judge TOPIC LABELS ONLY.

Each section in the packet may carry up to three topic labels with a confidence between 0 and 1. For
each label, decide from the raw Markdown whether that label is a fair description of what the section
is about. Report the ones that are wrong, with the reason. Separately, list sections that carry NO
label but obviously should, saying what the label should have been -- describe it in your own words,
you have no list to choose from.

Judge the label, not the number. A correct label with a low confidence is not an error. Do not
penalise a label for being broad if it is right.

${BAN}

Packet: ${pk}
Return the structured object.`, { label: `topics:${i + 1}`, phase: 'Judge', schema: TOPICS })
      .then(v => ({ remit: 'topics', packet: pk, result: v })),
  ]),

  // ---- stage 2: adversarially verify this document's critical/high findings as soon as they land
  (judged, orig, i) => {
    const found = (judged || []).filter(Boolean)
    const claims = []
    for (const j of found) {
      if (!j.result || !j.result.findings) continue
      for (const f of j.result.findings) {
        if (f.severity === 'critical' || f.severity === 'high') {
          claims.push({ ...f, remit: j.remit, packet: j.packet })
        }
      }
    }
    if (!claims.length) return { packet: orig, judged: found, confirmed: [], claims: 0 }
    // Bounded by the harness ceiling, not by taste. 33 documents x (6 judges + 3 refuters per
    // verified claim) must stay under 1000 agents, which puts the cap at 7:
    // 33 * (6 + 7*3) + 1 = 892, leaving room for retries. The cap is REPORTED rather than applied
    // quietly, because a silent truncation reads afterwards as "everything was verified".
    const CAP = 7
    const picked = claims.slice(0, CAP)
    if (claims.length > CAP) {
      log(`${orig.split('/').pop()}: ${claims.length} high/critical claims, verifying the first ${CAP} -- ${claims.length - CAP} NOT verified`)
    }
    return parallel(picked.map(c => () =>
      parallel([0, 1, 2].map(k => () =>
        agent(`You are a SKEPTIC. Another judge claims a published summary misstates its source. Your job is to
REFUTE that claim if it can be refuted. Default to refuted=true unless the claim clearly holds.

A claim is REFUTED if any of these is true:
* the source text actually does support the summary, read fairly and in full context
* the difference is wording, emphasis, or omission that does not change meaning
* the claimed quote is not in the source, or is quoted out of a context that reverses its sense
* the summary is a fair compression even though it is not word-for-word
* an exception or proviso elsewhere in the same section resolves the apparent conflict

A claim STANDS only if a careful reader of the summary would be actively misled about the source.

${BAN}

THE CLAIM
  document section : ${c.section_title}
  claim            : ${c.claim}
  quoted source    : ${c.quote_from_source}
  quoted summary   : ${c.quote_from_summary}
  reasoning given  : ${c.why_wrong}

Read the raw Markdown named in ${c.packet}, find that section yourself, and decide. Angle ${k + 1} of 3:
${['read the section in full, including anything before and after the quoted line',
   'test whether the difference changes the legal effect for either party',
   'test whether the quoted source text is real and fairly represented'][k]}.

Return the structured verdict.`, { label: `refute${k + 1}:${i + 1}`, phase: 'Verify', schema: VERDICT })))
        .then(vs => {
          const v = vs.filter(Boolean)
          const stands = v.filter(x => !x.refuted).length >= 2
          return { ...c, stands, votes: v.map(x => ({ refuted: x.refuted, why: x.reason.slice(0, 200) })) }
        })
    )).then(rs => ({
      packet: orig, judged: found,
      confirmed: rs.filter(Boolean).filter(r => r.stands),
      claims: claims.length, verified: picked.length,
    }))
  }
)

// ---------------------------------------------------------------------------
phase('Synthesis')
const docs = perDoc.filter(Boolean)

function gather(remit, field) {
  const out = []
  for (const d of docs) for (const j of (d.judged || [])) {
    if (j.remit === remit && j.result) out.push({ packet: d.packet, ...j.result })
  }
  return out
}

const copying = gather('copying')
const topicRes = gather('topics')
const totals = {
  documents: docs.length,
  high_or_critical_claims: docs.reduce((a, d) => a + (d.claims || 0), 0),
  claims_actually_verified: docs.reduce((a, d) => a + (d.verified || 0), 0),
  confirmed_after_refutation: docs.reduce((a, d) => a + (d.confirmed || []).length, 0),
  copying: {
    sections: copying.reduce((a, c) => a + (c.sections_checked || 0), 0),
    verbatim: copying.reduce((a, c) => a + (c.verbatim_or_near_verbatim || 0), 0),
    reworded: copying.reduce((a, c) => a + (c.genuinely_reworded || 0), 0),
  },
  topics: {
    labels: topicRes.reduce((a, t) => a + (t.labels_checked || 0), 0),
    wrong: topicRes.reduce((a, t) => a + (t.wrong_labels || []).length, 0),
    missing: topicRes.reduce((a, t) => a + (t.missing_labels || []).length, 0),
  },
}
log(`confirmed defects: ${totals.confirmed_after_refutation} of ${totals.high_or_critical_claims} claims survived 3 refuters`)

const confirmed = docs.flatMap(d => (d.confirmed || []).map(c => ({
  document: d.packet.split('/').pop(), section: c.section_title, remit: c.remit,
  severity: c.severity, claim: c.claim,
  source: c.quote_from_source.slice(0, 300), summary: c.quote_from_summary.slice(0, 300),
})))

const report = await agent(`Write the verdict on a summarisation system, for an engineer who will act on it.

You are given every defect that survived three independent skeptics, plus aggregate measurements.
Nothing here is your own opinion to revisit -- report what the numbers and the confirmed defects say.

MEASUREMENTS
${JSON.stringify(totals, null, 1)}

CONFIRMED DEFECTS (survived 3 refuters, majority rule)
${JSON.stringify(confirmed, null, 1)}

Write, in plain prose with no marketing language:
1. Whether the summaries are fit to publish, stated plainly, with the number that decides it.
2. The defect classes that survived, ordered by how much damage each does, each with one concrete
   example quoted from the data above. Group repeats -- if six documents show the same class, say so
   once and give the count.
3. Whether the abstractive requirement is met, using the copying counts. State the fraction reworded.
   If any sections are still near-verbatim, say how many and in which documents.
4. Topic labels: how many were judged wrong out of how many checked, and the pattern in the wrong
   ones if there is one.
5. What to fix next, in priority order, each tied to a defect class above. Be specific about the
   change, not the goal.
6. What this exercise did NOT establish -- coverage gaps, remits nobody held, anything a judge could
   not see. Be honest and specific.

Do not congratulate anyone. Do not hedge a clear result. If something is broken, say it is broken.`,
  { label: 'verdict', phase: 'Synthesis' })

return { totals, confirmed, report }
