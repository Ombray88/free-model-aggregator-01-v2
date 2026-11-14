import type { Brief, Probe, RatingRow, WorkloadRequest, Capabilities } from './providers.ts';
import { meetsRequirements, capabilitiesOf, ratingOfModel, modelFree } from './providers.ts';

/**
 * Safe planning caps for providers. These represent the minimum of their documented 
 * budget ranges and are safe to use for planning requests in any trial.
 */
export const DEFAULT_BUDGET_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12
};

/**
 * A specific model from a provider that can be routed to.
 */
export type PlannedModel = {
  providerId: string;
  modelId: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  baseUrl: string;
  key: string;
  auth: string;
  rating: number;
};

/**
 * The planned routing for a single request, including a primary candidate and fallbacks.
 */
export type PlannedRoute = {
  requestId: string;
  candidates: PlannedModel[];
};

/**
 * Internal pool entry combining routing info with capability details for planning.
 */
type PoolEntry = PlannedModel & { capabilities: Capabilities };

/**
 * Router determines how to distribute requests across available free model providers.
 * It manages provider health, budget constraints, and failover logic.
 */
export class Router {
  public calls = 0;
  private pool: PoolEntry[] = [];
  private budgetCaps: Record<string, number>;
  
  // Runtime state
  private routes: Map<string, PlannedRoute> = new Map();
  private tried: Map<string, Set<string>> = new Map(); // requestId -> Set of providerId:modelId
  private failures: Map<string, number> = new Map(); // providerId -> consecutive failure count
  private demoted: Set<string> = new Set(); // providerId set

  constructor(brief: Brief, probes: readonly Probe[], budgetCaps?: Record<string, number>) {
    this.budgetCaps = { ...DEFAULT_BUDGET_CAPS, ...budgetCaps };
    this.buildPool(brief, probes);
  }

  /**
   * Initializes the pool of available models based on catalog probes and the ratings table.
   */
  private buildPool(brief: Brief, probes: readonly Probe[]): void {
    const providers = brief.directory.providers;
    const ratings = brief.ratings;

    for (const entry of providers) {
      const probe = probes.find(p => p.provider.id === entry.id);
      let modelsFound = false;

      if (probe && probe.ok && probe.models.length > 0) {
        for (const model of probe.models) {
          const row = ratingOfModel(ratings, entry.id, model.id);
          if (modelFree(model, row)) {
            const caps = capabilitiesOf(model, row);
            this.pool.push({
              providerId: entry.id,
              modelId: model.id,
              wire: entry.wire,
              baseUrl: entry.base,
              key: entry.key,
              auth: entry.auth,
              rating: row?.rating_milli ?? 0,
              capabilities: caps
            });
            modelsFound = true;
          }
        }
      }

      // Fallback to ratings table if probe failed or returned nothing
      if (!modelsFound && ratings) {
        const fallbackRows = ratings.ratings.filter(r => r.provider === entry.id && r.free);
        for (const row of fallbackRows) {
          this.pool.push({
            providerId: entry.id,
            modelId: row.model,
            wire: entry.wire,
            baseUrl: entry.base,
            key: entry.key,
            auth: entry.auth,
            rating: row.rating_milli,
            capabilities: {
              tools: row.tools,
              vision: row.vision,
              json: row.json,
              context_length: row.context_length
            }
          });
        }
      }
    }
  }

  /**
   * Creates a deterministic routing plan for the given workload manifest.
   */
  plan(manifest: readonly WorkloadRequest[]): PlannedRoute[] {
    this.reset();
    
    // 1. Identify eligible models for each request and sort them
    const requestCandidates = new Map<string, PoolEntry[]>();
    for (const req of manifest) {
      const eligible = this.pool.filter(m => {
        // Basic requirement matching
        if (!meetsRequirements(req.requires, m.capabilities)) return false;
        
        // OpenAI-client JSON mode rule
        if (req.api === 'openai' && req.requires.json === true) {
          if (m.wire !== 'openai' && m.wire !== 'gemini') return false;
        }

        // Context length requirement
        if (req.requires.min_context !== undefined) {
          if (m.capabilities.context_length === null || m.capabilities.context_length < req.requires.min_context) {
            return false;
          }
        }

        return true;
      });

      // Sort: rating desc, providerId asc, modelId asc
      eligible.sort((a, b) => {
        if (b.rating !== a.rating) return b.rating - a.rating;
        if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
        return a.modelId < b.modelId ? -1 : 1;
      });

      requestCandidates.set(req.id, eligible);
    }

    // 2. Plan requests in priority order: weight desc, then manifest order
    const planOrder = [...manifest].sort((a, b) => {
      if (b.weight !== a.weight) return b.weight - a.weight;
      // Stable sort by index in manifest
      return manifest.indexOf(a) - manifest.indexOf(b);
    });

    const plannedAssignments = new Map<string, number>(); // providerId -> count
    const finalRoutes = new Map<string, PlannedRoute>();

    for (const req of planOrder) {
      const candidates = requestCandidates.get(req.id) || [];
      if (candidates.length === 0) {
        finalRoutes.set(req.id, { requestId: req.id, candidates: [] });
        continue;
      }

      // Pick first candidate under budget cap, or just the first if all at cap
      let pickedIdx = candidates.findIndex(m => {
        const current = plannedAssignments.get(m.providerId) || 0;
        const cap = this.budgetCaps[m.providerId] ?? 5;
        return current < cap;
      });

      if (pickedIdx === -1) pickedIdx = 0;

      const picked = candidates[pickedIdx];
      plannedAssignments.set(picked.providerId, (plannedAssignments.get(picked.providerId) || 0) + 1);

      // Candidate list: picked first, then others in sorted order
      const orderedCandidates = [picked, ...candidates.filter((_, i) => i !== pickedIdx)].map(m => {
        // Strip capabilities for the public PlannedModel type
        const { capabilities: _, ...rest } = m;
        return rest;
      });

      finalRoutes.set(req.id, { requestId: req.id, candidates: orderedCandidates });
    }

    // Store the plan for runtime
    this.routes = finalRoutes;

    // Return routes in original manifest order
    return manifest.map(req => finalRoutes.get(req.id)!);
  }

  /**
   * Provides the next available model candidate for a request, handling failover and demotions.
   */
  nextCandidate(requestId: string): PlannedModel | null {
    const route = this.routes.get(requestId);
    if (!route || route.candidates.length === 0) return null;

    let triedSet = this.tried.get(requestId);
    if (!triedSet) {
      triedSet = new Set();
      this.tried.set(requestId, triedSet);
    }

    const untried = route.candidates.filter(m => !triedSet!.has(`${m.providerId}:${m.modelId}`));
    if (untried.length === 0) return null;

    // Try to find a non-demoted candidate first
    let selected = untried.find(m => !this.demoted.has(m.providerId)) || null;

    // If all remaining untried are demoted, take the first one anyway
    if (!selected) {
      selected = untried[0];
    }

    triedSet.add(`${selected.providerId}:${selected.modelId}`);
    this.calls++;
    return selected;
  }

  /**
   * Records a successful call to a provider, clearing its failure state.
   */
  noteSuccess(providerId: string): void {
    this.failures.set(providerId, 0);
    this.demoted.delete(providerId);
  }

  /**
   * Records a failed call, demoting the provider if failures persist.
   */
  noteFailure(providerId: string): void {
    const count = (this.failures.get(providerId) || 0) + 1;
    this.failures.set(providerId, count);
    if (count >= 2) {
      this.demoted.add(providerId);
    }
  }

  /**
   * Resets all runtime state, including the current plan and health tracking.
   */
  reset(): void {
    this.routes.clear();
    this.tried.clear();
    this.failures.clear();
    this.demoted.clear();
    this.calls = 0;
  }
}
