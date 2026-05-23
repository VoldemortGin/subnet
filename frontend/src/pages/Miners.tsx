import { useEffect, useState } from 'react'
import { Users, ChevronDown, ChevronRight } from 'lucide-react'
import type { Miner } from '../api/client'
import { getMiners } from '../api/client'

function StrikeBadge({ status }: { status: Miner['strike_status'] }) {
  const config = {
    normal: { label: 'Normal', className: 'bg-green-600/20 text-green-400' },
    yellow_card: {
      label: 'Warning',
      className: 'bg-yellow-600/20 text-yellow-400',
    },
    red_card: { label: 'Red Card', className: 'bg-red-600/20 text-red-400' },
    banned: { label: 'Banned', className: 'bg-red-800/30 text-red-500' },
  }
  const { label, className } = config[status]
  return (
    <span
      className={`inline-flex px-2 py-0.5 rounded text-xs font-semibold ${className}`}
    >
      {label}
    </span>
  )
}

function ProbeHistory({ history }: { history: boolean[] }) {
  const last10 = history.slice(-10)
  return (
    <div className="flex items-center gap-1">
      {last10.map((passed, i) => (
        <div
          key={i}
          className={`w-2.5 h-2.5 rounded-full ${
            passed ? 'bg-green-500' : 'bg-red-500'
          }`}
          title={passed ? 'Passed' : 'Failed'}
        />
      ))}
      {last10.length === 0 && (
        <span className="text-xs text-[#a1a1a1]">No probes</span>
      )}
    </div>
  )
}

function ScoreBreakdown({
  miner,
}: {
  miner: Miner
}) {
  const scores = [
    {
      label: 'Probe Score',
      value: miner.probe_accuracy,
      weight: '60%',
      color: 'bg-red-600',
    },
    {
      label: 'Total Score',
      value: miner.total_score / 100,
      weight: '100%',
      color: 'bg-blue-600',
    },
  ]

  return (
    <div className="pt-3 pb-1 px-3 space-y-3 border-t border-white/5">
      {scores.map((s) => (
        <div key={s.label}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-[#a1a1a1]">
              {s.label}{' '}
              <span className="text-white/30">({s.weight})</span>
            </span>
            <span className="font-mono">
              {(s.value * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${s.color}`}
              style={{ width: `${Math.min(s.value * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Miners() {
  const [miners, setMiners] = useState<Miner[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  useEffect(() => {
    getMiners()
      .then(setMiners)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-1">Miners</h1>
        <p className="text-[#a1a1a1] text-sm">
          Network participants and their performance
        </p>
      </div>

      <div className="bg-[#141414] border border-white/10 rounded-xl overflow-hidden">
        <div className="grid grid-cols-[2rem_1fr_8rem_7rem_6rem_7rem_10rem] gap-3 px-5 py-3 text-xs font-medium text-[#a1a1a1] border-b border-white/5 uppercase tracking-wider">
          <span />
          <span>Miner</span>
          <span>Backend</span>
          <span>Accuracy</span>
          <span>Score</span>
          <span>Status</span>
          <span>Last 10 Probes</span>
        </div>

        {miners.length === 0 ? (
          <div className="py-12 text-center text-[#a1a1a1]">
            <Users className="w-8 h-8 mx-auto mb-2 opacity-40" />
            <p className="text-sm">No miners registered</p>
          </div>
        ) : (
          miners.map((miner) => {
            const expanded = expandedId === miner.id
            return (
              <div key={miner.id} className="border-b border-white/5 last:border-0">
                <button
                  onClick={() =>
                    setExpandedId(expanded ? null : miner.id)
                  }
                  className="w-full grid grid-cols-[2rem_1fr_8rem_7rem_6rem_7rem_10rem] gap-3 px-5 py-3 items-center hover:bg-white/[0.02] transition-colors text-left"
                >
                  <span className="text-[#a1a1a1]">
                    {expanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </span>
                  <span className="text-sm font-medium truncate">
                    {miner.name}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded bg-white/5 text-[#a1a1a1] w-fit">
                    {miner.backend}
                  </span>
                  <span className="text-sm font-mono">
                    {(miner.probe_accuracy * 100).toFixed(1)}%
                  </span>
                  <span className="text-sm font-mono font-semibold text-red-400">
                    {miner.total_score.toFixed(1)}
                  </span>
                  <StrikeBadge status={miner.strike_status} />
                  <ProbeHistory history={miner.probe_history} />
                </button>
                {expanded && <ScoreBreakdown miner={miner} />}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
