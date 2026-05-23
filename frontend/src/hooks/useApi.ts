import { useState, useEffect, useCallback, useRef } from 'react'

interface UseApiResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
}

export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const refetch = useCallback(() => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setError(null)

    fetcher()
      .then((result) => {
        if (!controller.signal.aborted) {
          setData(result)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setLoading(false)
        }
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    refetch()
    return () => abortRef.current?.abort()
  }, [refetch])

  return { data, loading, error, refetch }
}
