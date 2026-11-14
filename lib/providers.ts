/**
 * Reads a trial brief (three JSON documents found by their schema member) and probes each provider's catalog route.
 */
import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export function joinBase(envUrl: string, base: string): string {
  const trimmedBase = base.trim();
  const httpPrefixRegex = /^https?:\/\//i;
  if (httpPrefixRegex.test(trimmedBase)) {
    return trimmedBase.replace(/\/+$/, '');
  }
  const strippedEnv = envUrl.replace(/\/+$/, '');
  if (trimmedBase === '') {
    return strippedEnv;
  }
  const prefix = trimmedBase.startsWith('/') ? '' : '/';
  return strippedEnv + prefix + trimmedBase.replace(/\/+$/, '');
}

export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') {
    return {};
  }
  switch (auth) {
    case 'bearer':
      return { authorization: 'Bearer ' + key };
    case 'x-api-key':
      return { 'x-api-key': key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': key };
    default:
      return {};
  }
}

function objectEnd(text: string, start: number): number {
  if (text[start] !== '{') {
    return -1;
  }
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
    } else {
      if (ch === '"') {
        inString = true;
      } else if (ch === '{') {
        depth++;
      } else if (ch === '}') {
        depth--;
        if (depth === 0) {
          return i;
        }
      }
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const docs: Json[] = [];
  let pos = 0;
  while (pos < text.length) {
    const idx = text.indexOf('{', pos);
    if (idx === -1) {
      break;
    }
    const end = objectEnd(text, idx);
    if (end === -1) {
      pos = idx + 1;
      continue;
    }
    const slice = text.slice(idx, end + 1);
    try {
      const parsed = JSON.parse(slice);
      if (isRecord(parsed) && typeof parsed.schema === 'string') {
        docs.push(parsed);
      }
    } catch {
      // ignore parse errors
    }
    pos = idx + 1;
  }
  return docs;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (!id || !wire) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = stringOf(v.path) || stringOf(v.base_path) || stringOf(v.base_url) || stringOf(v.base);
  const name = stringOf(v.name) || id;
  return { id, name, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (!model || !provider) return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (!id) return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const min_context = typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? requires.min_context : undefined;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(min_context !== undefined ? { min_context } : {})
    }
  };
}

export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (!path) {
    return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err: any) {
    return { directory, ratings: null, workload: null, error: 'cannot read ' + path + ': ' + (err?.message ?? String(err)) };
  }
  const docs = documentsIn(text);
  let dirDoc: any = null;
  let ratDoc: any = null;
  let workDoc: any = null;
  for (const doc of docs) {
    if (isRecord(doc)) {
      if (doc.schema === PROVIDERS_SCHEMA && Array.isArray(doc.providers)) {
        dirDoc = doc;
      } else if (doc.schema === RATINGS_SCHEMA && Array.isArray(doc.ratings)) {
        ratDoc = doc;
      } else if (doc.schema === WORKLOAD_SCHEMA && Array.isArray(doc.manifest)) {
        workDoc = doc;
      }
    }
  }
  if (!dirDoc) {
    return { directory, ratings: null, workload: null, error: 'the brief at ' + path + ' holds no ' + PROVIDERS_SCHEMA + ' document' };
  }
  const providers = dirDoc.providers.map(providerOf).filter((x: ProviderDirectoryEntry | null): x is ProviderDirectoryEntry => x !== null);

  let ratings: RatingsFile | null = null;
  if (ratDoc) {
    const ratRows = ratDoc.ratings.map(ratingOf).filter((x: RatingRow | null): x is RatingRow => x !== null);
    const bestEntries: { id: string; rating_milli: number }[] = [];
    if (Array.isArray(ratDoc.best)) {
      for (const b of ratDoc.best) {
        if (isRecord(b)) {
          const id = stringOf(b.id);
          if (id) {
            bestEntries.push({ id, rating_milli: numberOrNull(b.rating_milli) ?? 0 });
          }
        }
      }
    }
    ratings = {
      schema: RATINGS_SCHEMA,
      ratings: ratRows,
      best: bestEntries
    };
  }

  let workload: WorkloadFile | null = null;
  if (workDoc) {
    const manifest = workDoc.manifest.map(workloadRequestOf).filter((x: WorkloadRequest | null): x is WorkloadRequest => x !== null);
    workload = {
      schema: WORKLOAD_SCHEMA,
      manifest
    };
  }

  return {
    directory: { schema: PROVIDERS_SCHEMA, providers },
    ratings,
    workload,
    error: null
  };
}
export function ratingOfModel(
  ratings: RatingsFile | null,
  providerId: string,
  modelId: string
): RatingRow | null {
  if (!ratings) return null;
  for (const row of ratings.ratings) {
    if (row.provider === providerId && row.model === modelId) {
      return row;
    }
  }
  return null;
}

export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) {
    return row.free;
  }
  if (model.free !== null) {
    return model.free;
  }
  return false;
}

export function capabilitiesOf(
  model: ProbedModel,
  row: RatingRow | null
): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

export function meetsRequirements(
  requires: Requirement,
  caps: Capabilities
): boolean {
  if (requires.tools === true && caps.tools !== true) {
    return false;
  }
  if (requires.vision === true && caps.vision !== true) {
    return false;
  }
  if (requires.json === true && caps.json !== true) {
    return false;
  }
  if (requires.min_context !== undefined && requires.min_context !== null) {
    if (
      caps.context_length === null ||
      caps.context_length < requires.min_context
    ) {
      return false;
    }
  }
  return true;
}

function zeroPriced(v: Json): boolean | null {
  const pricing = typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Record<string, Json>).pricing : null;
  const pricingObj = typeof pricing === 'object' && pricing !== null && !Array.isArray(pricing) ? (pricing as Record<string, Json>) : null;
  if (!pricingObj) return null;
  const values: number[] = [];
  for (const val of Object.values(pricingObj)) {
    if (typeof val === 'number') {
      values.push(val);
    } else if (typeof val === 'string' && val.trim() !== '') {
      const num = Number(val);
      if (!isNaN(num)) {
        values.push(num);
      }
    }
  }
  if (values.length === 0) return null;
  for (const val of values) {
    if (!Number.isFinite(val) || val !== 0) {
      return false;
    }
  }
  return true;
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) return null;
  const rec = v as Record<string, unknown>;
  let id = '';
  if (wire === 'gemini') {
    const rawName = stringOf(rec.name);
    id = rawName.startsWith('models/') ? rawName.slice('models/'.length) : rawName;
  } else {
    id = stringOf(rec.id);
  }
  if (!id) return null;

  const context =
    numberOrNull(rec.context_length) ??
    numberOrNull(rec.context_window) ??
    numberOrNull(rec.inputTokenLimit);

  const displayNameRaw =
    stringOf(rec.display_name) || stringOf(rec.displayName);
  const display_name = displayNameRaw !== '' ? displayNameRaw : id;

  const free = boolOrNull(rec.free) ?? zeroPriced(rec as Json);

  return {
    id,
    display_name,
    free,
    context_length: context,
    tools: boolOrNull(rec.tools),
    vision: boolOrNull(rec.vision),
    json: boolOrNull(rec.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) return [];
  const rec = body as Record<string, unknown>;
  const list = wire === 'gemini' ? rec.models : rec.data;
  if (!Array.isArray(list)) return [];
  const results: ProbedModel[] = [];
  for (const entry of list) {
    const parsed = modelIn(wire, entry);
    if (parsed !== null) {
      results.push(parsed);
    }
  }
  return results;
}

export async function probeProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  timeoutMs = 5000
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url =
    provider.wire === 'gemini'
      ? base + '/v1beta/models'
      : base + '/v1/models';
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
  }
  try {
    const res = await fetch(url, {
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) {
      return {
        provider,
        ok: false,
        status: res.status,
        models: [],
        error: 'GET ' + url + ' answered HTTP ' + res.status,
      };
    }
    const body = await res.json();
    return {
      provider,
      ok: true,
      status: res.status,
      models: modelsIn(provider.wire, body),
      error: null,
    };
  } catch (err: any) {
    return {
      provider,
      ok: false,
      status: 0,
      models: [],
      error: 'GET ' + url + ' failed: ' + (err?.message ?? String(err)),
    };
  }
}

export function buildCatalog(
  _envUrl: string,
  probes: readonly Probe[],
  ratings: RatingsFile | null
): CatalogReport {
  const providers: ProviderCatalogEntry[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => a.id.localeCompare(b.id));
    const seen = new Set<string>();
    const uniqueModels: ProbedModel[] = [];
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      uniqueModels.push(model);
    }

    const models: CatalogModel[] = [];
    for (const model of uniqueModels) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }

    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
