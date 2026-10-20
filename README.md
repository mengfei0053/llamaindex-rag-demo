# LlamaIndex RAG 示例

一个最小可运行的检索增强生成示例。服务读取 `data/` 下的 Markdown 和纯文本，用 LlamaIndex 建成向量索引并写入 Chroma，再通过 FastAPI 回答问题。示例文档是虚构公司「北辰软件 / Northstar Software」的常见问题、产品说明和支持政策，中文和英文问题都可以问。

向量有两条路径。默认 `EMBEDDING_PROVIDER=openai`，入库和检索调用 `OPENAI_BASE_URL` 的 `/v1/embeddings`。如果代理只有对话、没有 embeddings（常见是 `/v1/embeddings` 返回 404），把 `EMBEDDING_PROVIDER` 设为 `local`：向量改在容器里用 FastEmbed 的 ONNX 模型计算，对话仍然走 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

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
| `OPENAI_EMBED_MODEL` | 可选 | `openai` 模式下的向量模型，默认 `text-embedding-3-small`。`local` 模式不会调用它 |
| `EMBEDDING_PROVIDER` | 可选 | `openai`（默认）或 `local` |
| `LOCAL_EMBED_MODEL` | 可选 | `local` 模式的 ONNX 模型，默认 `BAAI/bge-small-zh-v1.5` |
| `FASTEMBED_CACHE_PATH` | 可选 | ONNX 模型缓存目录。本机默认 `~/.cache/fastembed`。Compose 固定为容器内 `/app/fastembed-cache` |
| `DATA_DIR` | 可选 | 文档目录，默认 `./data` |
| `STORAGE_DIR` | 可选 | Chroma 持久化目录，默认 `./storage` |

## 本地向量（代理没有 `/v1/embeddings`）

`local` 使用 [fastembed](https://github.com/qdrant/fastembed)（ONNX Runtime），不安装 PyTorch。默认模型 `BAAI/bge-small-zh-v1.5` 约 90MB、512 维，面向中文，示例里的英文短句也可以检索。Linux aarch64 用的是 `onnxruntime` 的官方 aarch64 轮子，不需要下载多 GB 的 torch。

想换模型时设置 `LOCAL_EMBED_MODEL`。若更在意多语言而不是中文专项，可以用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（约 220MB）。模型列表见 [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)。

第一次以 `local` 启动时，进程会把 ONNX 文件下载到 `FASTEMBED_CACHE_PATH`。镜像里没有模型，所以这次启动需要能访问 Hugging Face（`Qdrant/bge-small-zh-v1.5`）或 `storage.googleapis.com`。下载完成后保留该目录，之后就不会再下。

对话密钥仍然必填：`/query` 和 `/ingest` 都会检查 `OPENAI_API_KEY`，但 `local` 模式下这个密钥只用来构造对话客户端，不会去请求 `/v1/embeddings`。

索引里会记下当时的 provider 和模型名。之后如果这两者变了，下次访问会自动重建。更早建立、没有这段标记的索引会继续沿用；要换模型就再调一次 `POST /ingest`。

在只有对话接口的机器上，`.env` 写成：

```bash
EMBEDDING_PROVIDER=local
OPENAI_API_KEY=代理密钥
OPENAI_BASE_URL=http://代理地址/v1
OPENAI_MODEL=代理实际提供的对话模型
# OPENAI_EMBED_MODEL 不会被调用
# 可选，默认就是下面这个：
# LOCAL_EMBED_MODEL=BAAI/bge-small-zh-v1.5
```

## 本地运行

需要 Python 3.10 到 3.12（镜像使用 3.12）。`llama-index-embeddings-fastembed` 暂不支持 3.13。

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

容器监听 `8000`。`./data` 以只读方式挂进容器，索引写到 `./storage`（已在 `.gitignore` 中忽略）。`EMBEDDING_PROVIDER=local` 时，ONNX 模型写到宿主机 `./fastembed-cache`（容器内 `/app/fastembed-cache`）。第一次启动会下载约 90MB 的模型，请保留这个目录。更换文档后调用一次 `POST /ingest`。

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
  -v "$(pwd)/fastembed-cache:/app/fastembed-cache" \
  llamaindex-rag-demo
```

`local` 模式再加上 `-e EMBEDDING_PROVIDER=local`。缓存目录由镜像里的 `FASTEMBED_CACHE_PATH=/app/fastembed-cache` 指定。

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
