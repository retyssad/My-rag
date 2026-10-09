<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { streamChat, type Citation, type RetrievedChunk, type ChatOptions } from '../api'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  retrieved?: RetrievedChunk[]
  rewritten?: string | null
  intent?: string | null
  streaming?: boolean
}

const messages = ref<ChatMessage[]>([])
const input = ref('')
const conversationId = ref<number | null>(null)
const sending = ref(false)
const scrollBox = ref<HTMLElement | null>(null)

// 消融/演示参数面板
const opts = ref({
  mode: 'hybrid' as ChatOptions['mode'],
  use_rerank: true,
  use_query_rewrite: true,
  use_intent: true,
  use_parent_retrieval: true,
  product_category: '' as string,
  doc_type: '' as string,
})

const examples = [
  '星耀 X30 Pro 的处理器是什么？',
  '耳机的退换货政策是什么？',
  '降噪最好的耳机是哪款？',
  '对比一下 X30 Pro 和 X30 的处理器差异',
]

async function scrollToBottom() {
  await nextTick()
  scrollBox.value?.scrollTo({ top: scrollBox.value.scrollHeight, behavior: 'smooth' })
}

function buildOptions(): ChatOptions {
  const o: ChatOptions = {
    mode: opts.value.mode,
    use_rerank: opts.value.use_rerank,
    use_query_rewrite: opts.value.use_query_rewrite,
    use_intent: opts.value.use_intent,
    use_parent_retrieval: opts.value.use_parent_retrieval,
  }
  if (opts.value.product_category) o.product_category = opts.value.product_category
  if (opts.value.doc_type) o.doc_type = opts.value.doc_type
  return o
}

async function send(text?: string) {
  const q = (text ?? input.value).trim()
  if (!q || sending.value) return
  input.value = ''
  sending.value = true
  messages.value.push({ role: 'user', content: q })
  const assistant: ChatMessage = { role: 'assistant', content: '', streaming: true }
  messages.value.push(assistant)
  scrollToBottom()

  try {
    await streamChat(q, conversationId.value, buildOptions(), {
      onRetrieval(data) {
        conversationId.value = data.conversation_id
        assistant.retrieved = data.chunks
        assistant.rewritten = data.query_rewritten
        assistant.intent = data.intent
        scrollToBottom()
      },
      onDelta(t) {
        assistant.content += t
        scrollToBottom()
      },
      onCitations(cites) {
        assistant.citations = cites
      },
      onDone(data) {
        conversationId.value = data.conversation_id
        assistant.streaming = false
      },
      onError(msg) {
        assistant.content += `\n[出错] ${msg}`
        assistant.streaming = false
      },
    })
  } catch (e) {
    assistant.content += `\n[出错] ${e}`
  } finally {
    assistant.streaming = false
    sending.value = false
    scrollToBottom()
  }
}

function newChat() {
  messages.value = []
  conversationId.value = null
}

const typeLabel: Record<string, string> = {
  product_detail: '商品详情',
  comparison: '选购对比',
  faq: 'FAQ',
  policy: '售后政策',
}
const catLabel: Record<string, string> = { phone: '手机', laptop: '笔记本', earbuds: '耳机', all: '全站' }

const intentLabel: Record<string, string> = {
  product_qa: '商品咨询', policy: '售后政策', compare: '商品对比', chitchat: '闲聊',
}
const intentType: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
  product_qa: 'primary', policy: 'success', compare: 'warning', chitchat: 'info',
}
</script>

<template>
  <div class="chat-layout">
    <div class="config-bar">
      <el-select v-model="opts.mode" style="width: 120px" size="small">
        <el-option label="混合检索" value="hybrid" />
        <el-option label="仅向量" value="dense" />
        <el-option label="仅关键词" value="sparse" />
      </el-select>
      <el-checkbox v-model="opts.use_rerank" size="small" label="重排序" />
      <el-checkbox v-model="opts.use_query_rewrite" size="small" label="查询改写" />
      <el-checkbox v-model="opts.use_intent" size="small" label="意图识别" />
      <el-checkbox v-model="opts.use_parent_retrieval" size="small" label="扩展上下文" />
      <el-select v-model="opts.product_category" placeholder="全部类目" clearable style="width: 120px" size="small">
        <el-option label="手机" value="phone" />
        <el-option label="笔记本" value="laptop" />
        <el-option label="耳机" value="earbuds" />
      </el-select>
      <el-select v-model="opts.doc_type" placeholder="全部类型" clearable style="width: 120px" size="small">
        <el-option label="商品详情" value="product_detail" />
        <el-option label="选购对比" value="comparison" />
        <el-option label="FAQ" value="faq" />
        <el-option label="售后政策" value="policy" />
      </el-select>
      <el-button size="small" @click="newChat">新会话</el-button>
    </div>

    <div ref="scrollBox" class="messages">
      <el-empty v-if="!messages.length" description="输入问题开始对话，上方可切换检索策略做对比">
        <div class="examples">
          <el-tag v-for="e in examples" :key="e" class="example" @click="send(e)">{{ e }}</el-tag>
        </div>
      </el-empty>

      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <div class="bubble">
          <template v-if="m.role === 'user'">{{ m.content }}</template>
          <template v-else>
            <div v-if="m.intent || m.rewritten" class="meta-line">
              <el-tag v-if="m.intent" size="small" :type="intentType[m.intent] || 'info'">
                意图：{{ intentLabel[m.intent] || m.intent }}
              </el-tag>
              <span v-if="m.rewritten" class="rewritten">改写后：{{ m.rewritten }}</span>
            </div>

            <el-alert v-if="m.intent === 'chitchat'" type="info" :closable="false" show-icon
              title="识别为闲聊，未检索知识库" class="chitchat-note" />

            <el-collapse v-if="m.retrieved?.length" class="retrieved">
              <el-collapse-item :title="`检索到 ${m.retrieved.length} 个片段`">
                <div v-for="(c, ci) in m.retrieved" :key="ci" class="retrieved-item">
                  <div class="retrieved-head">
                    <el-tag size="small" type="info">{{ typeLabel[c.doc_type] || c.doc_type }}</el-tag>
                    <span class="doc-title">{{ c.doc_title }}</span>
                    <el-tag v-if="c.parent_id" size="small" type="success">完整上下文</el-tag>
                    <el-tag v-if="c.chunk_type === 'table'" size="small" type="warning">表格</el-tag>
                    <span v-if="c.score != null" class="score">score={{ c.score.toFixed(3) }}</span>
                  </div>
                  <pre class="chunk-content">{{ c.content }}</pre>
                </div>
              </el-collapse-item>
            </el-collapse>

            <div class="answer">{{ m.content }}<span v-if="m.streaming" class="cursor">▍</span></div>

            <div v-if="m.citations?.length" class="citations">
              <div class="citations-title">引用来源</div>
              <el-collapse>
                <el-collapse-item v-for="c in m.citations" :key="c.index"
                  :title="`[${c.index}] 《${c.doc_title}》 · ${catLabel[c.product_category] || c.product_category}`">
                  <pre class="chunk-content">{{ c.content }}</pre>
                </el-collapse-item>
              </el-collapse>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="input-bar">
      <el-input v-model="input" placeholder="输入问题，回车发送" :disabled="sending"
        @keyup.enter="send()" />
      <el-button type="primary" :loading="sending" @click="send()">发送</el-button>
    </div>
  </div>
</template>

<style scoped>
.chat-layout { display: flex; flex-direction: column; height: 100%; }
.config-bar { display: flex; gap: 12px; align-items: center; padding: 10px 4px; flex-wrap: wrap; }
.messages { flex: 1; overflow-y: auto; padding: 8px 4px; }
.examples { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.example { cursor: pointer; }
.msg { display: flex; margin: 10px 0; }
.msg.user { justify-content: flex-end; }
.msg.assistant { justify-content: flex-start; }
.bubble { max-width: 78%; padding: 10px 14px; border-radius: 10px; background: #f4f4f5; line-height: 1.6; }
.msg.user .bubble { background: #d9ecff; }
.meta-line { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
.rewritten { color: #909399; font-size: 12px; }
.chitchat-note { margin-bottom: 8px; }
.retrieved { margin-bottom: 8px; }
.retrieved-item { margin-bottom: 8px; }
.retrieved-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.doc-title { font-weight: 600; font-size: 13px; }
.score { color: #909399; font-size: 12px; }
.chunk-content { white-space: pre-wrap; word-break: break-word; background: #fff; padding: 8px; border-radius: 6px;
  font-size: 12px; max-height: 240px; overflow: auto; font-family: inherit; margin: 0; }
.answer { white-space: pre-wrap; }
.cursor { animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0; } }
.citations { margin-top: 10px; }
.citations-title { font-size: 12px; color: #909399; margin-bottom: 4px; }
.input-bar { display: flex; gap: 10px; padding: 10px 4px; border-top: 1px solid #eee; }
</style>
