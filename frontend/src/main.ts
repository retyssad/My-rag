import './style.css'
import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import ChatView from './views/ChatView.vue'
import UploadView from './views/UploadView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', component: ChatView, meta: { title: '智能问答' } },
    { path: '/upload', component: UploadView, meta: { title: '知识库管理' } },
  ],
})

createApp(App).use(router).use(ElementPlus).mount('#app')
