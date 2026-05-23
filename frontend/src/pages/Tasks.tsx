import { useEffect, useState, useCallback } from 'react'
import {
  ListTodo,
  FlaskConical,
  Image,
  Loader2,
  Plus,
  Clock,
  CheckCircle2,
} from 'lucide-react'
import type { Task } from '../api/client'
import { getTasks, generateProbe } from '../api/client'

function TypeBadge({ type }: { type: Task['type'] }) {
  return type === 'probe' ? (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-purple-600/20 text-purple-400">
      <FlaskConical className="w-3 h-3" />
      PROBE
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-blue-600/20 text-blue-400">
      <Image className="w-3 h-3" />
      REAL
    </span>
  )
}

function StatusBadge({ status }: { status: Task['status'] }) {
  const config = {
    pending: {
      icon: Clock,
      label: 'Pending',
      className: 'bg-yellow-600/20 text-yellow-400',
    },
    assigned: {
      icon: Loader2,
      label: 'Assigned',
      className: 'bg-blue-600/20 text-blue-400',
    },
    complete: {
      icon: CheckCircle2,
      label: 'Complete',
      className: 'bg-green-600/20 text-green-400',
    },
  }
  const { icon: Icon, label, className } = config[status]
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${className}`}
    >
      <Icon className="w-3 h-3" />
      {label}
    </span>
  )
}

type FilterType = 'all' | 'probe' | 'real'
type FilterStatus = 'all' | 'pending' | 'assigned' | 'complete'

export default function Tasks() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [filterType, setFilterType] = useState<FilterType>('all')
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all')

  const fetchTasks = useCallback(() => {
    setLoading(true)
    getTasks({
      type: filterType === 'all' ? undefined : filterType,
      status: filterStatus === 'all' ? undefined : filterStatus,
    })
      .then(setTasks)
      .catch(() => setTasks([]))
      .finally(() => setLoading(false))
  }, [filterType, filterStatus])

  useEffect(() => {
    fetchTasks()
  }, [fetchTasks])

  const handleGenerateProbe = async () => {
    setGenerating(true)
    try {
      await generateProbe()
      fetchTasks()
    } catch {
      // silently fail
    } finally {
      setGenerating(false)
    }
  }

  const formatTime = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold mb-1">Tasks</h1>
          <p className="text-[#a1a1a1] text-sm">
            Probe and real analysis tasks
          </p>
        </div>
        <button
          onClick={handleGenerateProbe}
          disabled={generating}
          className="flex items-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
        >
          {generating ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Plus className="w-4 h-4" />
          )}
          Generate Probe
        </button>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 bg-[#141414] border border-white/10 rounded-lg p-1">
          {(['all', 'probe', 'real'] as FilterType[]).map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                filterType === t
                  ? 'bg-white/10 text-white'
                  : 'text-[#a1a1a1] hover:text-white'
              }`}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1 bg-[#141414] border border-white/10 rounded-lg p-1">
          {(['all', 'pending', 'assigned', 'complete'] as FilterStatus[]).map(
            (s) => (
              <button
                key={s}
                onClick={() => setFilterStatus(s)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  filterStatus === s
                    ? 'bg-white/10 text-white'
                    : 'text-[#a1a1a1] hover:text-white'
                }`}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            )
          )}
        </div>
      </div>

      <div className="bg-[#141414] border border-white/10 rounded-xl overflow-hidden">
        <div className="grid grid-cols-[1fr_6rem_7rem_6rem_5rem_10rem] gap-3 px-5 py-3 text-xs font-medium text-[#a1a1a1] border-b border-white/5 uppercase tracking-wider">
          <span>Task ID</span>
          <span>Type</span>
          <span>Status</span>
          <span>Verdict</span>
          <span>Subs</span>
          <span>Created</span>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 text-red-600 animate-spin" />
          </div>
        ) : tasks.length === 0 ? (
          <div className="py-12 text-center text-[#a1a1a1]">
            <ListTodo className="w-8 h-8 mx-auto mb-2 opacity-40" />
            <p className="text-sm">No tasks found</p>
          </div>
        ) : (
          tasks.map((task) => (
            <div
              key={task.id}
              className="grid grid-cols-[1fr_6rem_7rem_6rem_5rem_10rem] gap-3 px-5 py-3 items-center border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                {task.image_url && (
                  <img
                    src={task.image_url}
                    alt=""
                    className="w-7 h-7 rounded object-cover bg-white/5 shrink-0"
                  />
                )}
                <span className="text-sm font-mono truncate">
                  {task.id.slice(0, 12)}...
                </span>
              </div>
              <TypeBadge type={task.type} />
              <StatusBadge status={task.status} />
              <span className="text-sm">
                {task.verdict ? (
                  <span
                    className={`font-semibold ${
                      task.verdict === 'tampered'
                        ? 'text-red-400'
                        : 'text-green-400'
                    }`}
                  >
                    {task.verdict}
                  </span>
                ) : (
                  <span className="text-[#a1a1a1]">-</span>
                )}
              </span>
              <span className="text-sm font-mono text-[#a1a1a1]">
                {task.submissions_count}
              </span>
              <span className="text-xs text-[#a1a1a1]">
                {formatTime(task.created_at)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
