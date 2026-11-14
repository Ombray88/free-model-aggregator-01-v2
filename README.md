# Free Model Aggregator

A single TypeScript server that aggregates multiple free-tier LLM providers into unified OpenAI-compatible and Anthropic-compatible API interfaces, picking the best free model for each incoming request.

## Overview

The server acts as a unified proxy in front of various free-tier AI model providers. For every inbound completion or message request, it dynamically selects and routes to the highest-rated available free model.

## Prerequisites & Execution

- **Runtime:** Node.js 22 (no external dependencies required)
- **Environment Variables:**
  - `MAKEFUN_ENV_URL`: Base URL for trial registration and work polling.
  - `MAKEFUN_PROMPT_FILE`: Path to the file containing provider definitions and credentials.
- **Execution Options:**
  - `node main.ts`: Starts the HTTP server and registers with the environment.
  - `node main.ts --list`: Probes providers and outputs the model catalog as JSON, then exits.

## Supported Routes

- `POST /v1/chat/completions`: OpenAI-compatible chat completions (JSON & SSE streaming).
- `POST /v1/messages`: Anthropic-compatible messages endpoint (JSON & SSE streaming).
- `GET /v1/models`: OpenAI and Anthropic format model listings.

## Intelligent Routing

1. **Workload Planning:** Analyzes fixed workload demand upfront to allocate per-trial provider budgets.
2. **Model Ranking:** Ranks eligible free models using public rating benchmarks.
3. **Failover Execution:** Maintains a per-request failover list to seamlessly fall back to secondary providers.
4. **Demotion Policy:** Demotes a provider endpoint only after repeated retryable failures.

## Wire Translation & Protocols

- **Upstream Wires:** Native wire implementations for OpenAI, Anthropic, and Google Gemini provider APIs.
- **Payload Translation:** Translates request and response payloads for both standard JSON responses and SSE streaming.
- **Usage Pass-Through:** Preserves and forwards upstream token usage metrics directly to clients.

## Codebase Layout

- `main.ts`: Entrypoint managing HTTP server startup, routing, registration, and work polling.
- `lib/gemini.ts`: Dedicated wire client module for Google Gemini upstream API.
- `lib/upstream.ts`: Wire client module handling OpenAI and Anthropic upstream communication and retry logic.
- `lib/translate.ts`: Translation layer mapping requests and responses across supported formats.
- `lib/router.ts`: Budget-aware request workload planner, model ranker, and failover router.
- `lib/providers.ts`: Provider catalog probing, health checks, and capacity management.
- `lib/wire.ts`: Wire protocol message types and error response formatting.

SPDX-License-Identifier: MIT
