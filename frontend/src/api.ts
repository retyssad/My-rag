// 后端 API 客户端：REST + SSE

import { fetchEventSource } from '@microsoft/fetch-event-source'

export interface Citation {
  index: number
  chunk_id: string
  doc_id: number
  doc_title: string
  doc_type: string
  product_category: string
  content: string
  score: number | null
}

export interface RetrievedChunk {
  chunk_id: string
  doc_title: string
  doc_type: string
  chunk_type: string
  content: string
  score: number | null
  parent_id?: string | null
  matched_child_ids?: string[]
}

export interface DocumentItem {
  id: number
  title: string
  doc_type: string
  product_category: string
  source_path: string
  status: string
  chunk_count: number
  error_msg: string | null
  created_at: string
  updated_at: string
}

export interface ChatOptions {
  mode?: 'dense' | 'sparse' | 'hybrid'
  use_rerank?: boolean
  use_query_rewrite?: boolean
  use_intent?: boolean
  use_parent_retrieval?: boolean
  product_category?: string
  doc_type?: string
}

const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const ADMIN_TOKEN = import.meta.env.VITE_ADMIN_TOKEN || 'change-me-admin-token'

export async function listDocuments(params: Record<string, string> = {}): Promise<{ total: number; items: DocumentItem[] }> {
  const qs = new URLSearchParams(params).toString()
  const r = await fetch(`${BASE}/api/documents${qs ? '?' + qs : ''}`)
  if (!r.ok) throw new Error(`加载文档失败: ${r.status}`)
  return r.json()
}

export async function uploadDocument(form: FormData): Promise<{ doc_id: number; status: string }> {
  const r = await fetch(`${BASE}/api/documents/upload`, {
    method: 'POST',
    headers: { 'X-Admin-Token': ADMIN_TOKEN },
    body: form,
  })
  if (!r.ok) throw new Error(`上传失败: ${r.status} ${await r.text()}`)
  return r.json()
}

export async function deleteDocument(id: number): Promise<void> {
  const r = await fetch(`${BASE}/api/documents/${id}`, {
    method: 'DELETE',
    headers: { 'X-Admin-Token': ADMIN_TOKEN },
  })
  if (!r.ok) throw new Error(`删除失败: ${r.status} ${await r.text()}`)
}

export async function retryDocument(id: number): Promise<void> {
  const r = await fetch(`${BASE}/api/documents/${id}/retry`, {
    method: 'POST',
    headers: { 'X-Admin-Token': ADMIN_TOKEN },
  })
  if (!r.ok) throw new Error(`重试失败: ${r.status}`)
}

export interface ChatHandlers {
  onRetrieval?: (data: { conversation_id: number; query_rewritten: string | null; mode: string; intent?: string | null; chunks: RetrievedChunk[] }) => void
  onDelta?: (text: string) => void
  onCitations?: (citations: Citation[]) => void
  onDone?: (data: { conversation_id: number }) => void
  onError?: (msg: string) => void
}

export function streamChat(
  query: string,
  conversationId: number | null,
  options: ChatOptions,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return fetchEventSource(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, conversation_id: conversationId, ...options }),
    signal,
    openWhenHidden: true,
    async onopen(response) {
      if (response.ok) return
      throw new Error(`HTTP ${response.status}`)
    },
    onmessage(ev) {
      const data = ev.data ? JSON.parse(ev.data) : {}
      switch (ev.event) {
        case 'retrieval': handlers.onRetrieval?.(data); break
        case 'delta': handlers.onDelta?.(data.text); break
        case 'citations': handlers.onCitations?.(data.citations); break
        case 'done': handlers.onDone?.(data); break
        case 'error': handlers.onError?.(data.message); break
      }
    },
    onerror(err) {
      handlers.onError?.(String(err))
      throw err
    },
  })
}
