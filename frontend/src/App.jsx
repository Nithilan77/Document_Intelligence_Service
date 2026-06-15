import { useState, useRef, useCallback } from 'react'

const EXAMPLES = [
  'What capital ratios must American Express maintain?',
  'What interchange cap did the Federal Reserve set for debit?',
  'Does Mastercard issue cards or extend credit?',
]

const REFUSAL = 'do not contain enough information'

// Split answer text on [n] citation markers, returning React nodes where each
// marker is a live chip that highlights its source card.
function renderAnswer(text, onCite) {
  const parts = []
  const re = /\[(\d+)\]/g
  let last = 0, m
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    const n = parseInt(m[1], 10)
    parts.push(
      <span
        key={`${m.index}-${n}`}
        className="cite"
        onClick={() => onCite(n)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onCite(n)}
      >[{n}]</span>
    )
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

export default function App() {
  const [q, setQ] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lit, setLit] = useState(null)
  const sourceRefs = useRef({})

  const ask = useCallback(async (question) => {
    const text = (question ?? q).trim()
    if (!text) return
    setLoading(true); setError(null); setData(null); setLit(null)
    try {
      const url = `/api/ask?q=${encodeURIComponent(text)}&mode=${mode}&k=5`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      setData(await res.json())
    } catch (e) {
      setError(`Could not reach the service. Is the backend running on :8000? (${e.message})`)
    } finally {
      setLoading(false)
    }
  }, [q, mode])

  const onCite = useCallback((n) => {
    setLit(n)
    const el = sourceRefs.current[n]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => setLit((cur) => (cur === n ? null : cur)), 2200)
  }, [])

  const refused = data && data.answer.toLowerCase().includes(REFUSAL)

  return (
    <div className="wrap">
      <header className="masthead">
        <p className="eyebrow">Retrieval-grounded · SEC 10-K filings</p>
        <h1 className="title">Filing Intelligence</h1>
        <p className="sub">
          Ask a question about three payments-industry annual reports. Every answer is
          built only from retrieved passages, with citations you can trace back to the source.
        </p>
        <div className="corpus">
          <span className="tag">American Express 10-K</span>
          <span className="tag">Mastercard 10-K</span>
          <span className="tag">Visa 10-K</span>
        </div>
      </header>

      <div className="modes">
        {['hybrid', 'dense', 'sparse'].map((mname) => (
          <label key={mname}>
            <input
              type="radio" name="mode" value={mname}
              checked={mode === mname} onChange={() => setMode(mname)}
            />
            {mname}
          </label>
        ))}
      </div>

      <div className="ask">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && ask()}
          placeholder="Ask about capital ratios, interchange, competitors…"
          aria-label="Question"
        />
        <button onClick={() => ask()} disabled={loading || !q.trim()}>
          {loading ? 'Asking…' : 'Ask'}
        </button>
      </div>

      <div className="examples">
        Try:{' '}
        {EXAMPLES.map((ex, i) => (
          <span key={i}>
            <button onClick={() => { setQ(ex); ask(ex) }}>{ex}</button>
            {i < EXAMPLES.length - 1 ? ' · ' : ''}
          </span>
        ))}
      </div>

      {loading && <p className="loading">Retrieving passages and composing a grounded answer<span>…</span></p>}
      {error && <p className="error">{error}</p>}

      {data && (
        <div className="answer-block">
          <div className={`status ${refused ? 'refused' : 'grounded'}`}>
            {refused ? 'Insufficient evidence in corpus' : `Grounded · ${mode} retrieval`}
          </div>
          <div className={`answer ${refused ? 'refused' : ''}`}>
            {renderAnswer(data.answer, onCite)}
          </div>

          {!refused && data.sources?.length > 0 && (
            <section className="sources">
              <h2>Sources</h2>
              {data.sources.map((s) => (
                <div
                  key={s.n}
                  ref={(el) => (sourceRefs.current[s.n] = el)}
                  className={`source ${lit === s.n ? 'lit' : ''}`}
                >
                  <div className="source-head">
                    <span className="source-n">{s.n}</span>
                    <span className="source-doc">{s.doc_id}</span>
                    <span className="source-sec">{s.section}</span>
                    <span className="source-score">rrf {s.score.toFixed(4)}</span>
                  </div>
                  <p className="source-text">{s.text ? s.text.slice(0, 260) + '…' : ''}</p>
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  )
}