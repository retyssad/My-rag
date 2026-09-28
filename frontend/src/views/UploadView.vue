<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listDocuments, uploadDocument, deleteDocument, retryDocument, type DocumentItem } from '../api'

const docs = ref<DocumentItem[]>([])
const total = ref(0)
const loading = ref(false)
const uploading = ref(false)
let timer: number | undefined

const form = ref({
  doc_type: 'product_detail',
  product_category: 'phone',
  title: '',
})
const file = ref<File | null>(null)

const typeLabel: Record<string, string> = {
  product_detail: '商品详情', comparison: '选购对比', faq: 'FAQ', policy: '售后政策',
}
const catLabel: Record<string, string> = { phone: '手机', laptop: '笔记本', earbuds: '耳机', all: '全站' }
const statusType: Record<string, string> = {
  pending: 'info', processing: 'warning', done: 'success', failed: 'danger',
}
const statusLabel: Record<string, string> = {
  pending: '待处理', processing: '处理中', done: '完成', failed: '失败',
}

async function load() {
  loading.value = true
  try {
    const r = await listDocuments({ limit: '100' })
    docs.value = r.items
    total.value = r.total
  } catch (e) {
    ElMessage.error(String(e))
  } finally {
    loading.value = false
  }
}

function onFileChange(f: { raw?: File }) {
  file.value = f.raw ?? null
}

async function submit() {
  if (!file.value) {
    ElMessage.warning('请先选择文件')
    return
  }
  uploading.value = true
  try {
    const fd = new FormData()
    fd.append('file', file.value)
    fd.append('doc_type', form.value.doc_type)
    fd.append('product_category', form.value.product_category)
    if (form.value.title) fd.append('title', form.value.title)
    await uploadDocument(fd)
    ElMessage.success('上传成功，后台处理中')
    file.value = null
    form.value.title = ''
    await load()
  } catch (e) {
    ElMessage.error(String(e))
  } finally {
    uploading.value = false
  }
}

async function onDelete(doc: DocumentItem) {
  try {
    await ElMessageBox.confirm(`确认删除《${doc.title}》？`, '删除确认', { type: 'warning' })
  } catch {
    return
  }
  try {
    await deleteDocument(doc.id)
    ElMessage.success('已删除')
    await load()
  } catch (e) {
    ElMessage.error(String(e))
  }
}

async function onRetry(doc: DocumentItem) {
  try {
    await retryDocument(doc.id)
    ElMessage.success('已重新入队')
    await load()
  } catch (e) {
    ElMessage.error(String(e))
  }
}

onMounted(() => {
  load()
  // 处理中状态轮询
  timer = window.setInterval(() => {
    if (docs.value.some((d) => d.status === 'pending' || d.status === 'processing')) load()
  }, 3000)
})
onUnmounted(() => timer && clearInterval(timer))
</script>

<template>
  <div class="upload-page">
    <el-card class="upload-card">
      <template #header>上传文档</template>
      <el-form label-width="90px">
        <el-form-item label="文件">
          <el-upload :auto-upload="false" :limit="1" :on-change="onFileChange" :show-file-list="true">
            <el-button>选择文件</el-button>
            <template #tip>
              <div class="tip">支持 PDF / DOCX / HTML / Markdown</div>
            </template>
          </el-upload>
        </el-form-item>
        <el-form-item label="文档类型">
          <el-select v-model="form.doc_type" style="width: 180px">
            <el-option v-for="(v, k) in typeLabel" :key="k" :label="v" :value="k" />
          </el-select>
        </el-form-item>
        <el-form-item label="商品类目">
          <el-select v-model="form.product_category" style="width: 180px">
            <el-option label="手机" value="phone" />
            <el-option label="笔记本" value="laptop" />
            <el-option label="耳机" value="earbuds" />
            <el-option label="全站" value="all" />
          </el-select>
        </el-form-item>
        <el-form-item label="标题">
          <el-input v-model="form.title" placeholder="留空则用文件名" style="width: 320px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="uploading" @click="submit">上传并入库</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="list-card">
      <template #header>
        知识库文档（共 {{ total }} 篇）
        <el-button style="float: right" size="small" @click="load">刷新</el-button>
      </template>
      <el-table :data="docs" v-loading="loading" size="small" style="width: 100%">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="title" label="标题" min-width="180" show-overflow-tooltip />
        <el-table-column label="类型" width="90">
          <template #default="{ row }">{{ typeLabel[row.doc_type] || row.doc_type }}</template>
        </el-table-column>
        <el-table-column label="类目" width="80">
          <template #default="{ row }">{{ catLabel[row.product_category] || row.product_category }}</template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="Chunks" width="80" />
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tooltip :disabled="!row.error_msg" :content="row.error_msg || ''">
              <el-tag :type="statusType[row.status]">{{ statusLabel[row.status] || row.status }}</el-tag>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="130">
          <template #default="{ row }">
            <el-button v-if="row.status === 'failed'" size="small" @click="onRetry(row)">重试</el-button>
            <el-button size="small" type="danger" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.upload-page { display: flex; flex-direction: column; gap: 16px; padding: 4px; }
.tip { color: #909399; font-size: 12px; }
</style>
