# Milvus 版本锁定在 2.6 线（v2.6.24）

## 背景

Milvus 在选型时的可用版本有三条线：2.5.x（最新 v2.5.27）、2.6.x（最新 v2.6.24）、3.0.x（最新 v3.0.2，GitHub 标记 Latest）。版本选择直接影响安全性与 SDK 兼容性。

- **2.5.x**：存在认证绕过漏洞 **CVE-2026-26190**，官方强烈建议 2.5.x 用户立即升级。
- **3.0.2**：最新 stable，但作为大版本，SDK 兼容性声明尚不完整，生态（pymilvus、第三方示例）未必全部跟进。
- **2.6.24**：2.6 线成熟版本，官方 docker-compose 文件现行可用，且 2.6+ 默认使用嵌入式 Woodpecker（以 MinIO 为 WAL），部署更简单。

## 决定

Milvus 锁定 **v2.6.24**，通过官方 standalone docker-compose 部署。

## 理由

1. **规避已知 CVE**：2.5 线的认证绕过漏洞使其不适合新项目起点。
2. **成熟度优先于最新**：作品集项目需要的是「一次跑通、少踩坑」，而非追最新大版本；3.0 的 SDK 生态风险不值得承担。
3. **部署简化**：2.6+ 内嵌消息队列，standalone 仅依赖 etcd + MinIO，共三个容器，无独立 MQ。

## 后果

- `docker-compose.yml` 中的镜像 tag 与 compose 文件均固定到 `v2.6.24`，不使用 `latest`。
- 若后续需要升级到 3.0，需先核实 pymilvus 与所用 API（`hybrid_search` / `RRFRanker`）的兼容性。
