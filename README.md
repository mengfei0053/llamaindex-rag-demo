# LlamaIndex RAG 示例

一个最小可运行的检索增强生成示例。服务读取 `data/` 下的 Markdown 和纯文本，用 LlamaIndex 建成向量索引并写入 Chroma，再通过 FastAPI 回答问题。示例文档是虚构公司「北辰软件 / Northstar Software」的常见问题、产品说明和支持政策，中文和英文问题都可以问。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务是否存活，以及索引里有多少片段 |
| `POST` | `/query` | 请求体 `{"question": "..."}`，返回答案和来源片段 |
| `POST` | `/ingest` | 清空现有索引，重新读取 `data/` 并写入 Chroma |

设置了 `OPENAI_API_KEY` 且 `storage/` 里还没有索引时，服务会在后台读取 `data/` 并建立索引。`/health` 会立刻返回；入库完成后 `index_ready` 变为 `true`。没有密钥时进程也可以启动，此时 `index_ready` 为 `false`，`/query` 和 `/ingest` 返回 503。

## 环境变量

复制 `.env.example` 为 `.env` 后填写密钥。不要把真实密钥提交到仓库。

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 查询和入库时必需 | OpenAI 或兼容代理的密钥 |
| `OPENAI_BASE_URL` | 可选 | 兼容接口地址。留空则使用 OpenAI 官方地址。本地代理一般要带 `/v1` |
| `OPENAI_MODEL` | 可选 | 对话模型，默认 `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL` | 可选 | 向量模型，默认 `text-embedding-3-small` |
| `DATA_DIR` | 可选 | 文档目录，默认 `./data` |
| `STORAGE_DIR` | 可选 | Chroma 持久化目录，默认 `./storage` |

## 本地运行

需要 Python 3.10 及以上（镜像使用 3.12）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker 运行

在仓库根目录执行：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
docker compose up --build
```

容器监听 `8000`。`./data` 以只读方式挂进容器，索引写到 `./storage`（已在 `.gitignore` 中忽略）。更换文档后调用一次 `POST /ingest`。

只构建镜像、不启动：

```bash
docker compose build
```

不用 Compose 时：

```bash
docker build -t llamaindex-rag-demo .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e DATA_DIR=/app/data \
  -e STORAGE_DIR=/app/storage \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/storage:/app/storage" \
  llamaindex-rag-demo
```

## 调用示例

```bash
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"专业版多少钱？"}'

curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund window?"}'

curl -s -X POST http://127.0.0.1:8000/ingest
```

`/query` 的响应形如：

```json
{
  "answer": "专业版每人每月 99 元。",
  "sources": [
    {
      "file_name": "product.md",
      "score": 0.82,
      "text": "专业版：最多 50 名成员，每人每月 99 元，100 GB 存储空间。"
    }
  ]
}
```

答案只依据检索到的片段生成。问英文时用英文回答，问中文时用中文回答。文档里没有的内容会说明不知道。

## 示例文档

- `data/faq.md`：公司、工作时间、试用和发票
- `data/product.md`：岚图协作的版本、价格和数据驻留
- `data/support.md`：支持时效、可用性目标和退款规则

可以继续往 `data/` 添加 `.md` 或 `.txt`，然后调用 `POST /ingest`。
