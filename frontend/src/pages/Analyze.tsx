import { useState, useCallback, useRef } from 'react'
import {
  Upload,
  ShieldAlert,
  ShieldCheck,
  Loader2,
  ImageIcon,
  X,
} from 'lucide-react'
import type { ImageResult } from '../api/client'
import { uploadImage } from '../api/client'

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full animate-fill-bar ${color}`}
        style={{ width: `${value * 100}%` }}
      />
    </div>
  )
}

function ResultDisplay({ result }: { result: ImageResult }) {
  const isTampered = result.verdict === 'tampered'

  return (
    <div
      className={`animate-fade-in rounded-2xl border p-6 ${
        isTampered
          ? 'border-red-600/40 animate-pulse-red'
          : 'border-green-600/40 animate-pulse-green'
      }`}
    >
      <div className="flex items-center gap-3 mb-6">
        {isTampered ? (
          <div className="p-3 rounded-xl bg-red-600/20">
            <ShieldAlert className="w-8 h-8 text-red-500" />
          </div>
        ) : (
          <div className="p-3 rounded-xl bg-green-600/20">
            <ShieldCheck className="w-8 h-8 text-green-500" />
          </div>
        )}
        <div>
          <h3
            className={`text-2xl font-bold ${
              isTampered ? 'text-red-500' : 'text-green-500'
            }`}
          >
            {isTampered ? 'TAMPERED' : 'AUTHENTIC'}
          </h3>
          <p className="text-sm text-[#a1a1a1]">{result.filename}</p>
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1.5">
          <span className="text-[#a1a1a1]">Confidence</span>
          <span className="font-mono font-semibold">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <ConfidenceBar
          value={result.confidence}
          color={isTampered ? 'bg-red-600' : 'bg-green-600'}
        />
      </div>

      {result.method && (
        <div className="mb-6">
          <span className="text-xs text-[#a1a1a1]">Detection method</span>
          <span className="ml-2 text-xs px-2 py-1 rounded bg-white/10 font-mono">
            {result.method}
          </span>
        </div>
      )}

      <div
        className={`grid gap-4 ${
          result.visualization_url ? 'grid-cols-2' : 'grid-cols-1'
        }`}
      >
        <div>
          <p className="text-xs text-[#a1a1a1] mb-2">Original</p>
          <img
            src={result.image_url}
            alt="Original"
            className="w-full rounded-lg border border-white/10"
          />
        </div>
        {result.visualization_url && (
          <div>
            <p className="text-xs text-[#a1a1a1] mb-2">Detection Result</p>
            <img
              src={result.visualization_url}
              alt="Detection visualization"
              className="w-full rounded-lg border border-white/10"
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default function Analyze() {
  const [results, setResults] = useState<ImageResult[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const result = await uploadImage(file)
      setResults((prev) => [result, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
      e.target.value = ''
    },
    [handleFile]
  )

  return (
    <div className="animate-fade-in space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold mb-1">Analyze Image</h1>
        <p className="text-[#a1a1a1] text-sm">
          Upload an image to detect potential forgery
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative cursor-pointer border-2 border-dashed rounded-2xl p-16 text-center transition-all ${
          dragOver
            ? 'border-red-600 bg-red-600/10'
            : 'border-white/20 hover:border-red-600/50 hover:bg-white/[0.02]'
        } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          className="hidden"
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 text-red-600 animate-spin" />
            <p className="text-lg font-medium">Analyzing image...</p>
            <p className="text-sm text-[#a1a1a1]">
              Running forgery detection algorithms
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 rounded-2xl bg-red-600/10">
              <Upload className="w-10 h-10 text-red-600" />
            </div>
            <div>
              <p className="text-lg font-medium mb-1">
                Drop an image here or click to upload
              </p>
              <p className="text-sm text-[#a1a1a1]">
                Supports JPEG, PNG, WebP, and BMP
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 rounded-xl bg-red-600/10 border border-red-600/30 text-red-400">
          <X className="w-4 h-4 shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-[#a1a1a1]" />
            Results ({results.length})
          </h2>
          {results.map((result) => (
            <ResultDisplay key={result.id} result={result} />
          ))}
        </div>
      )}
    </div>
  )
}
