# Free Model Aggregator v2

Free Model Aggregator v2 is a standalone TypeScript server built for Node 22 with zero external dependencies. It aggregates multiple free-tier LLM providers into unified, OpenAI-compatible and Anthropic-compatible API endpoints.

## Overview & Prerequisites

The server runs natively on Node 22. Required environment variables:
- `MAKEFUN_ENV_URL`: Base URL for trial registration and work polling.
- `MAKEFUN_PROMPT_FILE`: Path to the brief file containing provider definitions and credentials.

### Execution
- `node main.ts`: Start the HTTP server, register the trial, and poll for work.
- `node main.ts --list`: Probe providers, print the model catalog as JSON, and exit.

## Supported Routes
- `POST /v1/chat/completions`: OpenAI-compatible chat completions (supports JSON and SSE streaming).
- `POST /v1/messages`: Anthropic-compatible messages (supports JSON and SSE streaming).
- `GET /v1/models`: Unified model catalog listings in both formats.

## Risk-Aware Routing
Version 2 implements a risk-aware routing strategy rather than rating-only selection:
- Maintains a refusal-risk table estimating refusal probability per provider call number from generator draws (first calls are always safe).
- Builds a static execution plan ordered by scarcity, assigning the lowest-risk eligible provider-model within cumulative quality loss budgets.
- At runtime, attempts the planned pick first, falling back through fresh choices ordered by risk and quality loss.
- Records outcomes and classifies upstream failure text: spent budgets or outages mark providers dead for the trial; refused calls increment specific call number risks.

## Wire Translation
Supports native OpenAI, Anthropic, and Google Gemini upstream wires with comprehensive request and response translation for both JSON and SSE streaming, preserving token usage, stop reasons, and echo.

## Codebase Layout
- `main.ts`: Entrypoint for server startup, routing, registration, and work polling.
- `lib/planner.ts`: Refusal-risk table, static plan builder, failover chooser, and outcome recording.
- `lib/router.ts`: Model pool management and plan execution coordinator.
- `lib/pool.ts`: Model pool builder from brief directory and catalog probes.
- `lib/providers.ts`: Brief file reader, catalog probing, and capability checks.
- `lib/translate.ts`: Request and response wire format translation.
- `lib/upstream.ts`: OpenAI and Anthropic upstream invocation and retry logic.
- `lib/gemini.ts`: Google Gemini upstream wire handling.
- `lib/wire.ts`: Wire message types and error response structures.

SPDX-License-Identifier: MIT
