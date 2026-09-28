<script setup lang="ts">
import { useRoute } from 'vue-router'
const route = useRoute()
</script>

<template>
  <el-container class="app">
    <el-header class="header">
      <div class="brand">企业电商客服知识库问答</div>
      <el-menu mode="horizontal" :default-active="route.path" router class="menu" :ellipsis="false">
        <el-menu-item index="/chat">智能问答</el-menu-item>
        <el-menu-item index="/upload">知识库管理</el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<style>
html, body, #app { height: 100%; margin: 0; }
.app { height: 100%; }
.header { display: flex; align-items: center; border-bottom: 1px solid #eee; }
.brand { font-weight: 700; font-size: 16px; margin-right: 24px; white-space: nowrap; }
.menu { border-bottom: none; flex: 1; }
.main { height: calc(100vh - 61px); overflow: hidden; padding: 12px 20px; }
</style>
