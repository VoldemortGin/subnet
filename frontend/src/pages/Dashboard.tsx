import { useEffect, useState } from 'react'
import {
  ScanSearch,
  ShieldAlert,
  Users,
  Target,
  TrendingUp,
  Clock,
} from 'lucide-react'
import type { DashboardStats, ImageResult, Miner } from '../api/client'
import { getDashboardStats, getImages, getLeaderboard } from '../api/client'

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType
  label: string
  value: string | number
  color: string
}) {
  return (
    <div className="bg-[#141414] border border-white/10 rounded-xl p-5 hover:border-red-600/30 transition-colors">
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg ${color}`}>
          <Icon className="w-4 h-4" />
        </div>
        <span className="text-sm text-[#a1a1a1]">{label}</span>
      </div>
      <p className="text-2xl font-bold font-mono">{value}</p>
    </div>
  )
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const isTampered = verdict === 'tampered'
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${
        isTampered
          ? 'bg-red-600/20 text-red-400'
          : 'bg-green-600/20 text-green-400'
      }`}
    >
      {isTampered ? (
        <ShieldAlert className="w-3 h-3" />
      ) : (
        <TrendingUp className="w-3 h-3" />
      )}
      {verdict.toUpperCase()}
    </span>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [images, setImages] = useState<ImageResult[]>([])
  const [leaderboard, setLeaderboard] = useState<Miner[]>([])
  const [loading, setLoading] = useState(true)

  const fetchAll = () => {
    Promise.allSettled([
      getDashboardStats(),
      getImages(),
      getLeaderboard(),
    ]).then(([s, i, l]) => {
      if (s.status === 'fulfilled') setStats(s.value)
      if (i.status === 'fulfilled') setImages(i.value.slice(0, 10))
      if (l.status === 'fulfilled') setLeaderboard(l.value.slice(0, 5))
      setLoading(false)
    })
  }

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 10_000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="animate-fade-in space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
        <p className="text-[#a1a1a1] text-sm">
          Real-time overview of the HARM network
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={ScanSearch}
          label="Total Analyzed"
          value={stats?.total_images ?? 0}
          color="bg-red-600/15 text-red-400"
        />
        <StatCard
          icon={ShieldAlert}
          label="Tampered Detected"
          value={stats?.tampered_count ?? 0}
          color="bg-orange-600/15 text-orange-400"
        />
        <StatCard
          icon={Users}
          label="Active Miners"
          value={stats?.active_miners ?? 0}
          color="bg-blue-600/15 text-blue-400"
        />
        <StatCard
          icon={Target}
          label="Avg Accuracy"
          value={`${((stats?.avg_accuracy ?? 0) * 100).toFixed(1)}%`}
          color="bg-green-600/15 text-green-400"
        />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-[#141414] border border-white/10 rounded-xl p-5">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#a1a1a1]" />
            Recent Analyses
          </h2>
          {images.length === 0 ? (
            <p className="text-[#a1a1a1] text-sm py-8 text-center">
              No images analyzed yet
            </p>
          ) : (
            <div className="space-y-2">
              {images.map((img) => (
                <div
                  key={img.id}
                  className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <img
                      src={img.image_url}
                      alt=""
                      className="w-8 h-8 rounded object-cover bg-white/5"
                    />
                    <span className="text-sm truncate">{img.filename}</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-xs text-[#a1a1a1] font-mono">
                      {(img.confidence * 100).toFixed(0)}%
                    </span>
                    <VerdictBadge verdict={img.verdict} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-[#141414] border border-white/10 rounded-xl p-5">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[#a1a1a1]" />
            Miner Leaderboard
          </h2>
          {leaderboard.length === 0 ? (
            <p className="text-[#a1a1a1] text-sm py-8 text-center">
              No miners registered yet
            </p>
          ) : (
            <div className="space-y-2">
              {leaderboard.map((miner, i) => (
                <div
                  key={miner.id}
                  className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono text-[#a1a1a1] w-5">
                      #{i + 1}
                    </span>
                    <span className="text-sm font-medium">{miner.name}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-white/5 text-[#a1a1a1]">
                      {miner.backend}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs text-[#a1a1a1]">
                      Accuracy:{' '}
                      <span className="text-white font-mono">
                        {(miner.probe_accuracy * 100).toFixed(0)}%
                      </span>
                    </span>
                    <span className="text-sm font-mono font-semibold text-red-400">
                      {miner.total_score.toFixed(1)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
