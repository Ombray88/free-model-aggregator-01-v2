// Main entry point for the free model aggregator server
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { authHeaders, buildCatalog, capabilitiesOf, isRecord, joinBase, meetsRequirements, modelFree, probeProvider, ratingOfModel, readBrief, type CatalogReport, type Json, type Probe, type ProbedModel, type Requirement } from './lib/providers.ts';
import { Router, type PlannedModel } from './lib/router.ts';
import { buildUpstreamBody, buildOpenAIJsonResponse, buildAnthropicJsonResponse, buildOpenAISseResponse, buildAnthropicSseResponse } from './lib/translate.ts';
import { callOpenAI, callAnthropic, isRetryable, type UpstreamResult } from './lib/upstream.ts';
import { callGemini } from './lib/gemini.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';

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
  process.stderr.write('aggregator: ' + message + '\n');
}

async function load(): Promise<void> {
  if (brief.error !== null) {
    log(brief.error);
  }
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS))
  );
  for (const probe of probes) {
    if (!probe.ok && probe.error !== null) {
      log(probe.error);
    }
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(brief, probes);
  router.plan(brief.workload?.manifest ?? []);
}

const byId = (a: ProbedModel, b: ProbedModel): number => {
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
};

function servedModels(): ServedModel[] {
  const result: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sortedModels = [...probe.models].sort(byId);
    for (const model of sortedModels) {
      const row = ratingOfModel(brief.ratings, probe.provider, model.id);
      if (modelFree(model, row)) {
        result.push({
          id: model.id,
          provider: probe.provider,
          display_name: model.display_name,
        });
      }
    }
  }
  return result;
}

function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  let json = false;
  let tools = false;

  if (isRecord(body)) {
    if (Array.isArray(body.messages)) {
      for (const msg of body.messages) {
        if (isRecord(msg) && Array.isArray(msg.content)) {
          for (const block of msg.content) {
            if (isRecord(block)) {
              if (block.type === 'image' || block.type === 'image_url') {
                vision = true;
              }
            }
          }
        }
      }
    }

    if (isRecord(body.response_format)) {
      const rfType = body.response_format.type;
      if (rfType === 'json_object' || rfType === 'json_schema') {
        json = true;
      }
    }

    if (Array.isArray(body.tools) && body.tools.length > 0) {
      tools = true;
    }
  }

  return { tools, vision, json };
}

function getSyntheticCandidate(requestId: string, requires: Requirement, clientWire: 'openai' | 'anthropic'): PlannedModel | null {
  let candidates = syntheticCandidatesCache.get(requestId);
  if (!candidates) {
    candidates = [];
    if (brief.directory && brief.directory.providers) {
      for (const entry of brief.directory.providers) {
        const probe = probes.find((p) => p.provider.id === entry.id);
        if (!probe || !probe.ok) continue;
        for (const model of probe.models) {
          const row = ratingOfModel(brief.ratings, entry.id, model.id);
          if (!modelFree(model, row)) continue;
          const caps = capabilitiesOf(model, row);
          if (!meetsRequirements(requires, caps)) continue;
          if (clientWire === 'openai' && requires.json && entry.wire !== 'openai' && entry.wire !== 'gemini') continue;
          if (requires.min_context !== undefined && requires.min_context !== null) {
            if (caps.context_length === null || caps.context_length < requires.min_context) continue;
          }
          candidates.push({
            providerId: entry.id,
            modelId: model.id,
            wire: entry.wire,
            baseUrl: entry.base,
            key: entry.key,
            auth: entry.auth,
            rating: row ? row.rating_milli : 0,
            tools: caps.tools ?? false,
            vision: caps.vision ?? false,
            json: caps.json ?? false,
            contextLength: caps.context_length,
          });
        }
      }
    }
    candidates.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.providerId !== b.providerId) return a.providerId.localeCompare(b.providerId);
      return a.modelId.localeCompare(b.modelId);
    });
    syntheticCandidatesCache.set(requestId, candidates);
  }

  let tried = syntheticTried.get(requestId);
  if (!tried) {
    tried = new Set();
    syntheticTried.set(requestId, tried);
  }

  for (const c of candidates) {
    const keyStr = c.providerId + '/' + c.modelId;
    if (!tried.has(keyStr)) {
      tried.add(keyStr);
      return c;
    }
  }
  return null;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  let total = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    total += chunk.length;
    if (total > MAX_BODY_BYTES) return null;
    chunks.push(chunk);
  }
  if (chunks.length === 0) return null;
  try {
    const buf = Buffer.concat(chunks);
    const parsed: unknown = JSON.parse(buf.toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  const buf = Buffer.from(payload, 'utf8');
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': buf.length,
  });
  res.end(buf);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  const hasApiKey = Boolean(req.headers['x-api-key']);
  const body = hasApiKey ? anthropicModelList(list) : openAIModelList(list);
  sendJson(res, 200, body);
}

async function handleChatOrMessages(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (!body) {
    const errBody = clientWire === 'anthropic'
      ? anthropicError('invalid_request_error', 'the request body is not one JSON object')
      : openAIError('the request body is not one JSON object', 'invalid_request_error', 'invalid_request');
    sendJson(res, 400, errBody);
    return;
  }

  const arrivalIndex = arrivalCounter++;
  const entry = (brief.workload?.manifest ?? [])[arrivalIndex];
  const requestId = entry ? entry.id : 'synth_' + arrivalIndex;
  const requires = entry ? entry.requires : requirementsFromBody(body);
  const stream = body.stream === true;
  const requestedModelId = typeof body.model === 'string' && body.model !== '' ? body.model : null;

  let triedAny = false;
  let lastError: UpstreamResult | null = null;

  while (true) {
    const candidate = entry ? router.nextCandidate(requestId) : getSyntheticCandidate(requestId, requires, clientWire);
    if (!candidate) break;
    triedAny = true;

    const modelForResponse = requestedModelId || candidate.modelId;
    const upstreamBody = buildUpstreamBody(clientWire, candidate.wire, body, requestedModelId || candidate.modelId, stream);
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
        result = {
          ok: false,
          status: 501,
          code: 'not_implemented',
          message: 'unsupported wire ' + candidate.wire,
        };
      }
    } catch (err) {
      result = {
        ok: false,
        status: 0,
        code: 'network',
        message: err instanceof Error ? err.message : String(err),
      };
    }

    if (result.ok) {
      router.noteSuccess(candidate.providerId);
      if (stream) {
        const sseText = clientWire === 'openai'
          ? buildOpenAISseResponse(result, modelForResponse)
          : buildAnthropicSseResponse(result, modelForResponse);
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          'connection': 'keep-alive',
        });
        res.end(sseText);
      } else {
        const jsonRes = clientWire === 'openai'
          ? buildOpenAIJsonResponse(result, modelForResponse)
          : buildAnthropicJsonResponse(result, modelForResponse);
        sendJson(res, 200, jsonRes);
      }
      return;
    } else {
      lastError = result;
      const retryable = result.status === 0 || isRetryable(result.status);
      if (retryable) {
        router.noteFailure(candidate.providerId, result.message);
      }
    }
  }

  if (!triedAny) {
    const errBody = clientWire === 'anthropic'
      ? anthropicError('server_error', 'no free model on the directory meets this request')
      : openAIError('no free model on the directory meets this request', 'server_error', 'service_unavailable');
    sendJson(res, 503, errBody);
    return;
  }

  const errStatus = lastError && lastError.status > 0 ? lastError.status : 502;
  const errMsg = lastError ? lastError.message : 'upstream failure';
  const errCode = lastError ? lastError.code : 'bad_gateway';

  const errBody = clientWire === 'anthropic'
    ? anthropicError(errCode || 'api_error', errMsg)
    : openAIError(errMsg, errCode || 'api_error', errCode);
    sendJson(res, errStatus, errBody);
}
async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url || '', 'http://entry');
  const pathname = url.pathname;
  const method = req.method || 'GET';

  if (method === 'GET' && pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (method === 'POST' && pathname === '/v1/chat/completions') {
    await handleChatOrMessages(req, res, 'openai');
    return;
  }
  if (method === 'POST' && pathname === '/v1/messages') {
    await handleChatOrMessages(req, res, 'anthropic');
    return;
  }

  const isAnthropic = pathname === '/v1/messages' || Boolean(req.headers['x-api-key']);
  const errBody = isAnthropic
    ? anthropicError('invalid_request_error', 'no route ' + method + ' ' + pathname)
    : openAIError('no route ' + method + ' ' + pathname, 'invalid_request_error', 'invalid_request');
  sendJson(res, 404, errBody);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      server.off('error', reject);
      resolve();
    });
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const url = ENV_URL + '/register';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error('POST ' + url + ' answered HTTP ' + res.status + ': ' + text.slice(0, 500));
    }
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch {
      json = null;
    }
    if (isRecord(json) && typeof json.requests === 'number') {
      log('registered: the workload holds ' + json.requests + ' requests');
    }
  } finally {
    clearTimeout(timeout);
  }
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(async () => {
    if (inFlight) return;
    inFlight = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    try {
      const res = await fetch(ENV_URL + '/work', { signal: controller.signal });
      if (res.status === 200) {
        const text = await res.text();
        let data: unknown;
        try {
          data = JSON.parse(text);
        } catch {
          data = null;
        }
        if (isRecord(data) && data.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      }
    } catch {
      // Swallow errors and retry on next tick
    } finally {
      clearTimeout(timeout);
      inFlight = false;
    }
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    process.stdout.write(JSON.stringify(catalog) + '\n');
    process.exit(0);
  }
  if (!ENV_URL) {
    throw new Error('MAKEFUN_ENV_URL is not set');
  }

  const server = createServer(async (req, res) => {
    try {
      await handle(req, res);
    } catch (err) {
      log(err instanceof Error ? err.stack || err.message : String(err));
      if (!res.headersSent) {
        const isAnthropic = (req.url || '').startsWith('/v1/messages') || Boolean(req.headers['x-api-key']);
        const errBody = isAnthropic
          ? anthropicError('server_error', 'the server failed to answer')
          : openAIError('the server failed to answer', 'server_error', 'internal_server_error');
        sendJson(res, 500, errBody);
      }
    }
  });

  await listen(server);
  const addr = server.address();
  if (!addr || typeof addr === 'string') {
    throw new Error('invalid server address');
  }
  const port = addr.port;
  if (port === 8080) {
    throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  }

  log('listening on 0.0.0.0:' + port + ' with ' + catalog.providers.length + ' answered providers and ' + servedModels().length + ' free models');
  await register(port, catalog);

  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));

  poll(server);
}

try {
  await start();
} catch (err) {
  log(err instanceof Error ? err.stack || err.message : String(err));
  process.exit(1);
}
