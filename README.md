<!-- BlackRoad SEO Enhanced -->

# roadai

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad OS](https://img.shields.io/badge/Org-BlackRoad-OS-2979ff?style=for-the-badge)](https://github.com/BlackRoad-OS)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**roadai** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> BlackRoad AI Platform Core - Unified AI orchestration and inference

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Unified API** - Single interface for OpenAI, Anthropic, and Ollama models
- **Smart Routing** - Automatic provider selection based on model
- **Fallback Support** - Graceful degradation when providers are unavailable
- **Cost Tracking** - Real-time token counting and cost estimation
- **Streaming** - Server-sent events for real-time responses
- **OpenAI Compatible** - Drop-in replacement for OpenAI's API
- **Local Models** - First-class Ollama support for sovereignty

## Quick Start

### Installation

```bash
pip install roadai
```

### Start the Server

```bash
roadai serve --port 8000
```

### Use the API

```python
import httpx

response = httpx.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
})

print(response.json()["choices"][0]["message"]["content"])
```

### CLI Usage

```bash
# Chat with AI
echo "What is the capital of France?" | roadai chat

# Specify model
roadai chat "Explain quantum computing" --model claude-3.5-sonnet

# Check health
roadai health
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o1-mini |
| Anthropic | claude-3.5-sonnet, claude-3.5-haiku, claude-3-opus |
| Ollama | llama3.2, mistral, phi3, codellama, deepseek-coder |

## Configuration

Set API keys via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OLLAMA_HOST=http://localhost:11434
```

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

Response includes usage statistics:

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "choices": [...],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 45,
    "total_tokens": 57,
    "estimated_cost_usd": 0.000034
  }
}
```

### GET /health

Check provider availability:

```json
{
  "status": "healthy",
  "providers": {
    "openai": true,
    "anthropic": true,
    "ollama": false
  }
}
```

## Docker

```bash
docker build -t roadai .
docker run -p 8000:8000 -e OPENAI_API_KEY=... roadai
```

## License

MIT License - see [LICENSE](LICENSE)

---

Built with ❤️ by [BlackRoad OS](https://blackroad.io)
