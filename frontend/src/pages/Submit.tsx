import { useState, useEffect, useCallback } from 'react'
import {
  Loader2,
  Plus,
  FlaskConical,
  Image,
  Lock,
  Unlock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronRight,
  Shield,
  Copy,
} from 'lucide-react'
import type { Task, Miner, SubmissionResult } from '../api/client'
import {
  getPendingTasks,
  getMiners,
  generateProbe,
  commitHash,
  revealAnswer,
} from '../api/client'

const STEPS = [
  { label: 'Select Task', num: 1 },
  { label: 'Analyze', num: 2 },
  { label: 'Commit', num: 3 },
  { label: 'Reveal', num: 4 },
  { label: 'Result', num: 5 },
]

const METHODS = [
  { value: '', label: 'None' },
  { value: 'copy_move', label: 'Copy-Move' },
  { value: 'splicing', label: 'Splicing' },
  { value: 'compression', label: 'Compression' },
  { value: 'inpainting', label: 'Inpainting' },
  { value: 'metadata', label: 'Metadata' },
]

async function computeHash(
  verdict: string,
  confidence: number,
  method: string,
  nonce: string
): Promise<string> {
  const payload = `${verdict}|${confidence}|${method}|${nonce}`
  const encoder = new TextEncoder()
  const data = encoder.encode(payload)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')
}

function generateNonce(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

function Stepper({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-1 mb-8">
      {STEPS.map(({ label, num }, i) => (
        <div key={num} className="flex items-center">
          <div className="flex items-center gap-2">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                num < current
                  ? 'bg-red-600 text-white'
                  : num === current
                    ? 'bg-red-600 text-white ring-2 ring-red-600/50 ring-offset-2 ring-offset-[#0a0a0a]'
                    : 'bg-white/10 text-[#a1a1a1]'
              }`}
            >
              {num < current ? <CheckCircle2 className="w-4 h-4" /> : num}
            </div>
            <span
              className={`text-sm font-medium hidden sm:block ${
                num <= current ? 'text-white' : 'text-[#a1a1a1]'
              }`}
            >
              {label}
            </span>
          </div>
          {i < STEPS.length - 1 && (
            <ChevronRight className="w-4 h-4 text-[#a1a1a1] mx-2" />
          )}
        </div>
      ))}
    </div>
  )
}

function MinerSelector({
  miners,
  selectedId,
  onSelect,
}: {
  miners: Miner[]
  selectedId: string
  onSelect: (id: string) => void
}) {
  return (
    <div className="flex items-center gap-3 mb-6">
      <label className="text-sm text-[#a1a1a1]">Miner:</label>
      <select
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-red-600/50"
      >
        <option value="">Select a miner</option>
        {miners.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name} ({m.id.slice(0, 8)})
          </option>
        ))}
      </select>
    </div>
  )
}

function StepSelectTask({
  tasks,
  loading,
  generating,
  onAccept,
  onGenerate,
}: {
  tasks: Task[]
  loading: boolean
  generating: boolean
  onAccept: (task: Task) => void
  onGenerate: () => void
}) {
  return (
    <div className="animate-slide-up space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Available Tasks</h2>
        <button
          onClick={onGenerate}
          disabled={generating}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
        >
          {generating ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Plus className="w-4 h-4" />
          )}
          Get New Task
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 text-red-600 animate-spin" />
        </div>
      ) : tasks.length === 0 ? (
        <div className="text-center py-16 text-[#a1a1a1]">
          <Image className="w-10 h-10 mx-auto mb-3 opacity-40" />
          <p className="text-sm mb-4">No pending tasks available</p>
          <button
            onClick={onGenerate}
            disabled={generating}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
          >
            Generate a Probe Task
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {tasks.map((task) => (
            <div
              key={task.id}
              className="bg-[#141414] border border-white/10 rounded-xl overflow-hidden hover:border-red-600/40 transition-colors group"
            >
              {task.image_url && (
                <img
                  src={task.image_url}
                  alt=""
                  className="w-full h-40 object-cover bg-white/5"
                />
              )}
              <div className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-[#a1a1a1]">
                    {task.id.slice(0, 12)}
                  </span>
                  {task.type === 'probe' ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-purple-600/20 text-purple-400">
                      <FlaskConical className="w-3 h-3" />
                      PROBE
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-blue-600/20 text-blue-400">
                      <Image className="w-3 h-3" />
                      REAL
                    </span>
                  )}
                </div>
                <button
                  onClick={() => onAccept(task)}
                  className="w-full py-2 bg-white/5 hover:bg-red-600/20 hover:text-red-400 border border-white/10 hover:border-red-600/30 rounded-lg text-sm font-medium transition-all"
                >
                  Accept Task
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function StepAnalyze({
  task,
  verdict,
  confidence,
  method,
  onVerdictChange,
  onConfidenceChange,
  onMethodChange,
  onNext,
}: {
  task: Task
  verdict: string
  confidence: number
  method: string
  onVerdictChange: (v: string) => void
  onConfidenceChange: (c: number) => void
  onMethodChange: (m: string) => void
  onNext: () => void
}) {
  const ready = verdict !== ''

  return (
    <div className="animate-slide-up space-y-6">
      <h2 className="text-lg font-semibold">Analyze Image</h2>

      {task.image_url && (
        <div className="rounded-xl overflow-hidden border border-white/10">
          <img
            src={task.image_url}
            alt="Task image"
            className="w-full max-h-[400px] object-contain bg-black"
          />
        </div>
      )}

      <div className="space-y-5">
        <div>
          <label className="block text-sm text-[#a1a1a1] mb-3">Verdict</label>
          <div className="flex gap-3">
            <button
              onClick={() => onVerdictChange('authentic')}
              className={`flex-1 py-4 rounded-xl text-base font-bold border-2 transition-all ${
                verdict === 'authentic'
                  ? 'border-green-500 bg-green-600/20 text-green-400'
                  : 'border-white/10 bg-white/5 text-[#a1a1a1] hover:border-green-600/30 hover:text-green-400'
              }`}
            >
              AUTHENTIC
            </button>
            <button
              onClick={() => onVerdictChange('tampered')}
              className={`flex-1 py-4 rounded-xl text-base font-bold border-2 transition-all ${
                verdict === 'tampered'
                  ? 'border-red-500 bg-red-600/20 text-red-400'
                  : 'border-white/10 bg-white/5 text-[#a1a1a1] hover:border-red-600/30 hover:text-red-400'
              }`}
            >
              TAMPERED
            </button>
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-[#a1a1a1]">Confidence</span>
            <span className="font-mono font-semibold">{confidence}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={confidence}
            onChange={(e) => onConfidenceChange(Number(e.target.value))}
            className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-red-600"
          />
        </div>

        <div>
          <label className="block text-sm text-[#a1a1a1] mb-2">
            Method (optional)
          </label>
          <select
            value={method}
            onChange={(e) => onMethodChange(e.target.value)}
            className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-red-600/50"
          >
            {METHODS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>

        {ready && (
          <div className="animate-fade-in p-4 bg-white/5 border border-white/10 rounded-xl">
            <p className="text-xs text-[#a1a1a1] mb-2 uppercase tracking-wider">
              Response Preview
            </p>
            <div className="flex items-center gap-4 text-sm">
              <span
                className={`font-bold ${verdict === 'tampered' ? 'text-red-400' : 'text-green-400'}`}
              >
                {verdict.toUpperCase()}
              </span>
              <span className="text-[#a1a1a1]">|</span>
              <span className="font-mono">{confidence}%</span>
              {method && (
                <>
                  <span className="text-[#a1a1a1]">|</span>
                  <span className="px-2 py-0.5 bg-white/10 rounded text-xs font-mono">
                    {method}
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        <button
          onClick={onNext}
          disabled={!ready}
          className="w-full py-3 bg-red-600 hover:bg-red-700 disabled:opacity-30 disabled:cursor-not-allowed rounded-xl text-sm font-bold transition-colors"
        >
          Continue to Commit
        </button>
      </div>
    </div>
  )
}

function StepCommit({
  hash,
  nonce,
  computing,
  committing,
  committed,
  onCommit,
}: {
  hash: string
  nonce: string
  computing: boolean
  committing: boolean
  committed: boolean
  onCommit: () => void
}) {
  const [copied, setCopied] = useState<'hash' | 'nonce' | null>(null)

  const copyToClipboard = (text: string, type: 'hash' | 'nonce') => {
    navigator.clipboard.writeText(text)
    setCopied(type)
    setTimeout(() => setCopied(null), 2000)
  }

  if (computing) {
    return (
      <div className="animate-slide-up flex flex-col items-center justify-center py-16 space-y-4">
        <div className="relative">
          <Shield className="w-16 h-16 text-red-600 animate-pulse" />
          <Lock className="w-6 h-6 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
        </div>
        <p className="text-lg font-medium">Computing SHA-256 hash...</p>
        <p className="text-sm text-[#a1a1a1]">
          Sealing your answer cryptographically
        </p>
      </div>
    )
  }

  return (
    <div className="animate-slide-up space-y-6">
      <div className="text-center mb-6">
        <div className="animate-seal-stamp inline-block mb-4">
          <div className="p-4 rounded-full bg-red-600/20 border-2 border-red-600/40">
            <Lock className="w-10 h-10 text-red-500" />
          </div>
        </div>
        <h2 className="text-lg font-semibold">Commitment Hash</h2>
        <p className="text-sm text-[#a1a1a1] mt-1">
          Your answer has been cryptographically sealed
        </p>
      </div>

      <div className="space-y-4">
        <div className="animate-hash-reveal p-4 bg-[#0d0d0d] border border-red-600/30 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-[#a1a1a1] uppercase tracking-wider">
              SHA-256 Hash
            </span>
            <button
              onClick={() => copyToClipboard(hash, 'hash')}
              className="flex items-center gap-1 text-xs text-[#a1a1a1] hover:text-white transition-colors"
            >
              <Copy className="w-3 h-3" />
              {copied === 'hash' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <p className="font-mono text-sm text-red-400 break-all leading-relaxed">
            {hash}
          </p>
        </div>

        <div className="p-4 bg-[#0d0d0d] border border-white/10 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-[#a1a1a1] uppercase tracking-wider">
              Nonce (save this!)
            </span>
            <button
              onClick={() => copyToClipboard(nonce, 'nonce')}
              className="flex items-center gap-1 text-xs text-[#a1a1a1] hover:text-white transition-colors"
            >
              <Copy className="w-3 h-3" />
              {copied === 'nonce' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <p className="font-mono text-sm text-yellow-400 break-all">
            {nonce}
          </p>
        </div>
      </div>

      {!committed && (
        <div className="p-4 bg-yellow-600/10 border border-yellow-600/20 rounded-xl">
          <p className="text-sm text-yellow-400">
            <AlertTriangle className="w-4 h-4 inline mr-1.5 -mt-0.5" />
            Save your nonce before submitting. You will need it to reveal your
            answer.
          </p>
        </div>
      )}

      <button
        onClick={onCommit}
        disabled={committing || committed}
        className={`w-full py-3 rounded-xl text-sm font-bold transition-all ${
          committed
            ? 'bg-green-600/20 border border-green-600/30 text-green-400 cursor-default'
            : 'bg-red-600 hover:bg-red-700 disabled:opacity-50'
        }`}
      >
        {committing ? (
          <span className="flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Submitting...
          </span>
        ) : committed ? (
          <span className="flex items-center justify-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            Commitment Recorded
          </span>
        ) : (
          'Submit Commitment'
        )}
      </button>

      {committed && (
        <div className="animate-fade-in p-4 bg-white/5 border border-white/10 rounded-xl text-center">
          <Lock className="w-5 h-5 text-red-500 mx-auto mb-2" />
          <p className="text-sm text-[#a1a1a1]">
            Your answer is now locked. Other miners cannot see it.
          </p>
        </div>
      )}
    </div>
  )
}

function StepReveal({
  verdict,
  confidence,
  method,
  nonce,
  revealing,
  onReveal,
}: {
  verdict: string
  confidence: number
  method: string
  nonce: string
  revealing: boolean
  onReveal: () => void
}) {
  return (
    <div className="animate-slide-up space-y-6">
      <div className="text-center mb-4">
        <Unlock className="w-10 h-10 text-yellow-400 mx-auto mb-3" />
        <h2 className="text-lg font-semibold">Ready to Reveal</h2>
        <p className="text-sm text-[#a1a1a1] mt-1">
          All commitments collected. Reveal your answer for verification.
        </p>
      </div>

      <div className="p-5 bg-[#141414] border border-white/10 rounded-xl space-y-3">
        <p className="text-xs text-[#a1a1a1] uppercase tracking-wider mb-3">
          Your Response
        </p>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-[#a1a1a1] text-xs">Verdict</span>
            <p
              className={`font-bold ${verdict === 'tampered' ? 'text-red-400' : 'text-green-400'}`}
            >
              {verdict.toUpperCase()}
            </p>
          </div>
          <div>
            <span className="text-[#a1a1a1] text-xs">Confidence</span>
            <p className="font-mono font-semibold">{confidence}%</p>
          </div>
          {method && (
            <div>
              <span className="text-[#a1a1a1] text-xs">Method</span>
              <p className="font-mono text-sm">{method}</p>
            </div>
          )}
          <div className={method ? '' : 'col-span-2'}>
            <span className="text-[#a1a1a1] text-xs">Nonce</span>
            <p className="font-mono text-xs text-yellow-400 break-all">
              {nonce}
            </p>
          </div>
        </div>
      </div>

      <button
        onClick={onReveal}
        disabled={revealing}
        className="w-full py-3 bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-xl text-sm font-bold transition-colors"
      >
        {revealing ? (
          <span className="flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Revealing...
          </span>
        ) : (
          'Reveal Answer'
        )}
      </button>
    </div>
  )
}

function StepResult({ result }: { result: SubmissionResult }) {
  const scoreColor =
    result.score >= 0.7
      ? 'text-green-400'
      : result.score >= 0.4
        ? 'text-yellow-400'
        : 'text-red-400'

  const strikeConfig: Record<string, { color: string; label: string }> = {
    normal: { color: 'text-green-400', label: 'Normal' },
    yellow_card: { color: 'text-yellow-400', label: 'Yellow Card' },
    red_card: { color: 'text-red-400', label: 'Red Card' },
    banned: { color: 'text-red-500', label: 'BANNED' },
  }
  const strike = strikeConfig[result.strike_status] ?? strikeConfig.normal

  return (
    <div className="animate-slide-up space-y-6">
      <div className="text-center">
        {result.hash_valid ? (
          <CheckCircle2 className="w-12 h-12 text-green-500 mx-auto mb-3" />
        ) : (
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
        )}
        <h2 className="text-lg font-semibold">
          {result.hash_valid ? 'Submission Scored' : 'Hash Mismatch!'}
        </h2>
        {!result.hash_valid && (
          <p className="text-sm text-red-400 mt-1">
            Your revealed answer does not match your commitment. Score: 0
          </p>
        )}
      </div>

      {result.hash_valid && (
        <div className="p-6 bg-[#141414] border border-white/10 rounded-xl text-center">
          <p className="text-xs text-[#a1a1a1] uppercase tracking-wider mb-2">
            Score
          </p>
          <p className={`text-5xl font-bold font-mono ${scoreColor}`}>
            {(result.score * 100).toFixed(1)}
          </p>
        </div>
      )}

      {result.is_probe && result.ground_truth && (
        <div className="p-5 bg-[#141414] border border-white/10 rounded-xl space-y-3">
          <p className="text-xs text-[#a1a1a1] uppercase tracking-wider">
            Ground Truth (Probe)
          </p>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-[#a1a1a1] text-xs">Verdict</span>
              <p
                className={`font-bold ${
                  result.ground_truth.verdict === 'tampered'
                    ? 'text-red-400'
                    : 'text-green-400'
                }`}
              >
                {result.ground_truth.verdict?.toUpperCase()}
              </p>
            </div>
            {result.ground_truth.method && (
              <div>
                <span className="text-[#a1a1a1] text-xs">Method</span>
                <p className="font-mono">{result.ground_truth.method}</p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex items-center gap-4 p-4 bg-[#141414] border border-white/10 rounded-xl">
        <div className="flex-1">
          <p className="text-xs text-[#a1a1a1] mb-1">Strike Status</p>
          <p className={`text-sm font-bold ${strike.color}`}>{strike.label}</p>
        </div>
        {result.strike_status !== 'normal' && (
          <AlertTriangle
            className={`w-5 h-5 ${
              result.strike_status === 'banned'
                ? 'text-red-500'
                : 'text-yellow-400'
            }`}
          />
        )}
      </div>

      {result.probe_history.length > 0 && (
        <div className="p-4 bg-[#141414] border border-white/10 rounded-xl">
          <p className="text-xs text-[#a1a1a1] uppercase tracking-wider mb-3">
            Recent Probe History
          </p>
          <div className="flex items-center gap-1.5">
            {result.probe_history.map((correct, i) => (
              <div
                key={i}
                className={`w-5 h-5 rounded-full ${
                  correct ? 'bg-green-500' : 'bg-red-500'
                }`}
                title={correct ? 'Correct' : 'Incorrect'}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Submit() {
  const [step, setStep] = useState(1)
  const [miners, setMiners] = useState<Miner[]>([])
  const [selectedMinerId, setSelectedMinerId] = useState('')
  const [tasks, setTasks] = useState<Task[]>([])
  const [loadingTasks, setLoadingTasks] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [verdict, setVerdict] = useState('')
  const [confidence, setConfidence] = useState(75)
  const [method, setMethod] = useState('')
  const [nonce, setNonce] = useState('')
  const [hash, setHash] = useState('')
  const [computing, setComputing] = useState(false)
  const [committing, setCommitting] = useState(false)
  const [committed, setCommitted] = useState(false)
  const [revealing, setRevealing] = useState(false)
  const [result, setResult] = useState<SubmissionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchTasks = useCallback(async () => {
    setLoadingTasks(true)
    try {
      const data = await getPendingTasks()
      setTasks(data)
    } catch {
      setTasks([])
    } finally {
      setLoadingTasks(false)
    }
  }, [])

  useEffect(() => {
    fetchTasks()
    getMiners()
      .then((m) => {
        setMiners(m)
        if (m.length > 0) setSelectedMinerId(m[0].id)
      })
      .catch(() => {})
  }, [fetchTasks])

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      await generateProbe()
      await fetchTasks()
    } catch {
      setError('Failed to generate probe task')
    } finally {
      setGenerating(false)
    }
  }

  const handleAcceptTask = (task: Task) => {
    if (!selectedMinerId) {
      setError('Please select a miner first')
      return
    }
    setSelectedTask(task)
    setError(null)
    setStep(2)
  }

  const handleAnalyzeNext = async () => {
    setComputing(true)
    setStep(3)

    const n = generateNonce()
    setNonce(n)

    await new Promise((r) => setTimeout(r, 800))

    const confidenceDecimal = confidence / 100
    const h = await computeHash(verdict, confidenceDecimal, method, n)
    setHash(h)
    setComputing(false)
  }

  const handleCommit = async () => {
    if (!selectedTask) return
    setCommitting(true)
    setError(null)

    try {
      await commitHash(selectedTask.id, selectedMinerId, hash)
      setCommitted(true)
      setTimeout(() => setStep(4), 1200)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to submit commitment'
      )
    } finally {
      setCommitting(false)
    }
  }

  const handleReveal = async () => {
    if (!selectedTask) return
    setRevealing(true)
    setError(null)

    try {
      const confidenceDecimal = confidence / 100
      const res = await revealAnswer(
        selectedTask.id,
        selectedMinerId,
        verdict,
        confidenceDecimal,
        method || null,
        nonce
      )
      setResult(res)
      setStep(5)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reveal answer')
    } finally {
      setRevealing(false)
    }
  }

  const handleReset = () => {
    setStep(1)
    setSelectedTask(null)
    setVerdict('')
    setConfidence(75)
    setMethod('')
    setNonce('')
    setHash('')
    setComputing(false)
    setCommitting(false)
    setCommitted(false)
    setRevealing(false)
    setResult(null)
    setError(null)
    fetchTasks()
  }

  return (
    <div className="animate-fade-in max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-1">Miner Submission</h1>
        <p className="text-[#a1a1a1] text-sm">
          Commit-Reveal protocol: submit your analysis securely
        </p>
      </div>

      <Stepper current={step} />

      <MinerSelector
        miners={miners}
        selectedId={selectedMinerId}
        onSelect={setSelectedMinerId}
      />

      {error && (
        <div className="flex items-center gap-2 p-4 rounded-xl bg-red-600/10 border border-red-600/30 text-red-400 mb-6">
          <XCircle className="w-4 h-4 shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {step === 1 && (
        <StepSelectTask
          tasks={tasks}
          loading={loadingTasks}
          generating={generating}
          onAccept={handleAcceptTask}
          onGenerate={handleGenerate}
        />
      )}

      {step === 2 && selectedTask && (
        <StepAnalyze
          task={selectedTask}
          verdict={verdict}
          confidence={confidence}
          method={method}
          onVerdictChange={setVerdict}
          onConfidenceChange={setConfidence}
          onMethodChange={setMethod}
          onNext={handleAnalyzeNext}
        />
      )}

      {step === 3 && (
        <StepCommit
          hash={hash}
          nonce={nonce}
          computing={computing}
          committing={committing}
          committed={committed}
          onCommit={handleCommit}
        />
      )}

      {step === 4 && (
        <StepReveal
          verdict={verdict}
          confidence={confidence}
          method={method}
          nonce={nonce}
          revealing={revealing}
          onReveal={handleReveal}
        />
      )}

      {step === 5 && result && <StepResult result={result} />}

      {step === 5 && (
        <button
          onClick={handleReset}
          className="mt-6 w-full py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm font-medium transition-colors"
        >
          Start New Submission
        </button>
      )}
    </div>
  )
}
