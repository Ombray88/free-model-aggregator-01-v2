import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { Router, type PlannedModel } from './lib/router.ts';
import {
  buildUpstreamBody,
  buildOpenAIJsonResponse,
  buildAnthropicJsonResponse,
  buildOpenAISseResponse,
  buildAnthropicSseResponse,
} from './lib/translate.ts';
import {
  callOpenAI,
  callAnthropic,
  isRetryable,
  type UpstreamResult,
} from './lib/upstream.ts';
import { callGemini } from './lib/gemini.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router;
let arrivalCounter = 0;

const syntheticTried = new Map<string, Set<string>>();
const syntheticCandidatesCache = new Map<string, PlannedModel[]>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(brief, probes);
  router.plan(brief.workload?.manifest ?? []);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

function getSyntheticCandidate(requestId: string, requires: Requirement, clientWire: 'openai' | 'anthropic'): PlannedModel | null {
  let candidates = syntheticCandidatesCache.get(requestId);
  if (!candidates) {
    candidates = [];
    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        const caps = capabilitiesOf(model, row);
        if (!meetsRequirements(requires, caps)) continue;
        if (clientWire === 'openai' && requires.json === true) {
          if (probe.provider.wire !== 'openai' && probe.provider.wire !== 'gemini') continue;
        }
        if (requires.min_context !== undefined) {
          if (caps.context_length === null || caps.context_length < requires.min_context) continue;
        }
        candidates.push({
          providerId: probe.provider.id,
          modelId: model.id,
          wire: probe.provider.wire,
          baseUrl: probe.provider.base,
          key: probe.provider.key,
          auth: probe.provider.auth,
          rating: row?.rating_milli ?? 0,
        });
      }
    }
    candidates.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
      return a.modelId < b.modelId ? -1 : 1;
    });
    syntheticCandidatesCache.set(requestId, candidates);
  }

  let tried = syntheticTried.get(requestId);
  if (!tried) {
    tried = new Set();
    syntheticTried.set(requestId, tried);
  }

  const untried = candidates.filter((m) => !tried!.has(`${m.providerId}:${m.modelId}`));
  if (untried.length === 0) return null;

  const selected = untried[0];
  tried.add(`${selected.providerId}:${selected.modelId}`);
  return selected;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handleChatOrMessages(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    if (clientWire === 'anthropic') {
      return sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not one JSON object'));
    }
    return sendJson(res, 400, openAIError('the request body is not one JSON object', 'invalid_request_error', null));
  }

  const arrivalIndex = arrivalCounter++;
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[arrivalIndex];

  let requestId: string;
  let requires: Requirement;

  if (entry !== undefined) {
    requestId = entry.id;
    requires = entry.requires;
  } else {
    requestId = `synth_${arrivalIndex}`;
    requires = requirementsFromBody(body);
  }

  const stream = body.stream === true;
  const requestedModelId = typeof body.model === 'string' && body.model !== '' ? body.model : undefined;

  let lastError: UpstreamResult | null = null;
  let triedAny = false;

  for (;;) {
    let candidate: PlannedModel | null = null;
    if (entry !== undefined) {
      candidate = router.nextCandidate(requestId);
    } else {
      candidate = getSyntheticCandidate(requestId, requires, clientWire);
    }

    if (candidate === null) {
      break;
    }

    triedAny = true;
    const modelIdForUpstream = requestedModelId || candidate.modelId;
    const upstreamBody = buildUpstreamBody(clientWire, candidate.wire, body, modelIdForUpstream, stream);
    const baseUrl = joinBase(ENV_URL, candidate.baseUrl);

    let result: UpstreamResult;
    try {
      if (candidate.wire === 'openai') {
        result = await callOpenAI(baseUrl, candidate.key, candidate.modelId, upstreamBody, stream, candidate.auth);
      } else if (candidate.wire === 'anthropic') {
        result = await callAnthropic(baseUrl, candidate.key, candidate.modelId, upstreamBody, stream, candidate.auth);
      } else if (candidate.wire === 'gemini') {
        result = await callGemini(baseUrl, candidate.key, candidate.modelId, upstreamBody, stream, candidate.auth);
      } else {
        result = { ok: false, status: 501, code: 'not_implemented', message: `unsupported wire ${candidate.wire}` };
      }
    } catch (err) {
      result = { ok: false, status: 0, code: 'network', message: err instanceof Error ? err.message : String(err) };
    }

    if (result.ok) {
      router.noteSuccess(candidate.providerId);
      const modelForResponse = requestedModelId || candidate.modelId;
      if (clientWire === 'openai') {
        if (stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(buildOpenAISseResponse(result, modelForResponse));
        } else {
          sendJson(res, 200, buildOpenAIJsonResponse(result, modelForResponse));
        }
      } else {
        if (stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(buildAnthropicSseResponse(result, modelForResponse));
        } else {
          sendJson(res, 200, buildAnthropicJsonResponse(result, modelForResponse));
        }
      }
      return;
    } else {
      lastError = result;
      const retryable = result.status === 0 || isRetryable(result.status);
      if (retryable) {
        router.noteFailure(candidate.providerId);
      }
    }
  }

  if (!triedAny) {
    const msg = 'no free model on the directory meets this request';
    if (clientWire === 'openai') {
      return sendJson(res, 503, openAIError(msg, 'no_free_model', null));
    } else {
      return sendJson(res, 503, anthropicError('overloaded_error', msg));
    }
  }

  const errStatus = lastError && lastError.status > 0 ? lastError.status : 502;
  const errMessage = lastError ? lastError.message : 'upstream request failed';
  const errCode = lastError ? lastError.code : 'upstream_error';

  if (clientWire === 'openai') {
    sendJson(res, errStatus, openAIError(errMessage, errCode, null));
  } else {
    sendJson(res, errStatus, anthropicError('api_error', errMessage));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChatOrMessages(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleChatOrMessages(req, res, 'anthropic');
    return;
  }
  const isAnthropic = url.pathname === '/v1/messages' || typeof req.headers['x-api-key'] === 'string';
  if (isAnthropic) {
    sendJson(res, 404, anthropicError('not_found', `no route ${req.method ?? ''} ${url.pathname}`));
  } else {
    sendJson(res, 404, openAIError(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found', null));
  }
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Try again on next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const isAnthropic = req.url?.startsWith('/v1/messages') === true || typeof req.headers['x-api-key'] === 'string';
        if (isAnthropic) {
          sendJson(res, 500, anthropicError('api_error', 'the server failed to answer'));
        } else {
          sendJson(res, 500, openAIError('the server failed to answer', 'internal_error', null));
        }
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
