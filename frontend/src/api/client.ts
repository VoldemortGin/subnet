export interface ImageResult {
  id: string
  filename: string
  verdict: 'authentic' | 'tampered'
  confidence: number
  method: string | null
  image_url: string
  visualization_url: string | null
  created_at: string
}

export interface Miner {
  id: string
  name: string
  backend: string
  probe_accuracy: number
  total_score: number
  strike_status: 'normal' | 'yellow_card' | 'red_card' | 'banned'
  probe_history: boolean[]
}

export interface MinerDetail extends Miner {
  submissions: Submission[]
  score_breakdown: {
    probe_score: number
    consensus_score: number
    latency_score: number
  }
}

export interface DashboardStats {
  total_images: number
  tampered_count: number
  authentic_count: number
  total_miners: number
  active_miners: number
  total_probes: number
  avg_accuracy: number
}

export interface Task {
  id: string
  type: 'probe' | 'real'
  status: 'pending' | 'assigned' | 'complete'
  image_url: string | null
  verdict: string | null
  submissions_count: number
  created_at: string
}

export interface Submission {
  id: string
  task_id: string
  miner_id: string
  verdict: string
  confidence: number
  score: number | null
  created_at: string
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

export function uploadImage(file: File): Promise<ImageResult> {
  const form = new FormData()
  form.append('file', file)
  return request<ImageResult>('/api/images/upload', {
    method: 'POST',
    body: form,
  })
}

export function getImages(): Promise<ImageResult[]> {
  return request<ImageResult[]>('/api/images')
}

export function getImage(id: string): Promise<ImageResult> {
  return request<ImageResult>(`/api/images/${id}`)
}

export function getMiners(): Promise<Miner[]> {
  return request<Miner[]>('/api/miners')
}

export function getMinerDetail(id: string): Promise<MinerDetail> {
  return request<MinerDetail>(`/api/miners/${id}`)
}

export function getDashboardStats(): Promise<DashboardStats> {
  return request<DashboardStats>('/api/dashboard/stats')
}

export function getLeaderboard(): Promise<Miner[]> {
  return request<Miner[]>('/api/dashboard/leaderboard')
}

export function getTasks(filters?: { type?: string; status?: string }): Promise<Task[]> {
  const params = new URLSearchParams()
  if (filters?.type && filters.type !== 'all') params.set('type', filters.type)
  if (filters?.status && filters.status !== 'all') params.set('status', filters.status)
  const qs = params.toString()
  return request<Task[]>(`/api/tasks${qs ? `?${qs}` : ''}`)
}

export function generateProbe(): Promise<Task> {
  return request<Task>('/api/tasks/probe', { method: 'POST' })
}

export function submitResult(
  taskId: string,
  minerId: string,
  verdict: string,
  confidence: number
): Promise<Submission> {
  return request<Submission>(`/api/miners/${minerId}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId, verdict, confidence }),
  })
}

export function registerMiner(name: string, backend: string): Promise<Miner> {
  return request<Miner>('/api/miners/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, backend_name: backend }),
  })
}

export function getPendingTasks(): Promise<Task[]> {
  return request<Task[]>('/api/tasks?status=pending')
}

export interface CommitResponse {
  ok: boolean
  message: string
}

export function commitHash(
  taskId: string,
  minerId: string,
  hash: string
): Promise<CommitResponse> {
  return request<CommitResponse>(`/api/tasks/${taskId}/commit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ miner_id: minerId, hash }),
  })
}

export interface SubmissionResult {
  hash_valid: boolean
  score: number
  is_probe: boolean
  ground_truth: { verdict: string; method: string | null } | null
  strike_status: string
  probe_history: boolean[]
}

export function revealAnswer(
  taskId: string,
  minerId: string,
  verdict: string,
  confidence: number,
  method: string | null,
  nonce: string
): Promise<SubmissionResult> {
  return request<SubmissionResult>(`/api/tasks/${taskId}/reveal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      miner_id: minerId,
      verdict,
      confidence,
      method,
      nonce,
    }),
  })
}
