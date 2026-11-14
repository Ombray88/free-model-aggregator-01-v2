// Router implementation managing model selection and failover based on planner and ratings
import { buildPlan, chooseNext, createPlannerState, recordCall, recordOutcome } from './planner.ts';
import type { PlannerCandidate, PlannerRequest, PlannerState } from './planner.ts';
import { buildPool } from './pool.ts';
import type { Brief, Probe, WorkloadRequest } from './providers.ts';

export type PlannedModel = PlannerCandidate;

export class Router {
  private pool: PlannerCandidate[] = [];
  private state: PlannerState;
  private planMap: Map<string, PlannerCandidate> = new Map();
  private manifestById: Map<string, WorkloadRequest> = new Map();
  private tried: Map<string, Set<string>> = new Map();
  private lastCall: Map<string, number> = new Map();
  private dead: Set<string> = new Set();
  private readonly planBudget = 500;
  private readonly failoverBudget = 700;
  public calls = 0;

  private brief: Brief;

  constructor(brief: Brief, probes: readonly Probe[]) {
    this.brief = brief;
    this.pool = buildPool(brief, probes);
    this.state = createPlannerState();
  }

  plan(manifest: readonly WorkloadRequest[]): void {
    const reqs: PlannerRequest[] = manifest.map((m) => ({
      id: m.id,
      api: m.api,
      weight: m.weight,
      best: (this.briefRatingsBest(m.id) ?? 1475),
      requires: m.requires ? {
        tools: m.requires.tools,
        vision: m.requires.vision,
        json: m.requires.json,
        minContext: m.requires.min_context
      } : undefined
    }));
    this.planMap = buildPlan(reqs, this.pool, this.planBudget);
    this.manifestById = new Map(manifest.map((m) => [m.id, m]));
    this.tried.clear();
    this.lastCall.clear();
    this.dead.clear();
    this.calls = 0;
  }

  nextCandidate(requestId: string): PlannedModel | null {
    const planned = this.planMap.get(requestId);
    if (planned && !this.triedFor(requestId).has(this.key(planned)) && !this.dead.has(planned.providerId)) {
      return this.handOut(planned, requestId);
    }
    const entry = this.manifestById.get(requestId);
    if (!entry) return null;
    const req: PlannerRequest = {
      id: entry.id,
      api: entry.api,
      weight: entry.weight,
      best: (this.briefRatingsBest(entry.id) ?? 1475),
      requires: entry.requires ? {
        tools: entry.requires.tools,
        vision: entry.requires.vision,
        json: entry.requires.json,
        minContext: entry.requires.min_context
      } : undefined
    };
    const cand = chooseNext(this.state, req, this.pool, this.triedFor(requestId), this.failoverBudget);
    return cand ? this.handOut(cand, requestId) : null;
  }

  noteSuccess(_providerId: string): void {}

  noteFailure(providerId: string, failureText: string): void {
    const n = this.lastCall.get(providerId);
    if (n === undefined) return;
    recordOutcome(this.state, providerId, n, false, failureText);
    if (/budget is spent/i.test(failureText) || /is out/i.test(failureText) || failureText.includes('does not exist') || failureText.includes('not available')) {
      this.dead.add(providerId);
    }
  }

  private triedFor(id: string): Set<string> {
    let s = this.tried.get(id);
    if (!s) {
      s = new Set<string>();
      this.tried.set(id, s);
    }
    return s;
  }

  private key(c: PlannerCandidate): string {
    return c.providerId + ':' + c.modelId;
  }

  private handOut(c: PlannerCandidate, id: string): PlannerCandidate {
    const n = recordCall(this.state, c.providerId);
    this.lastCall.set(c.providerId, n);
    this.triedFor(id).add(this.key(c));
    this.calls++;
    return c;
  }

  private briefRatingsBest(id: string): number | undefined {
    const bestEntry = this.brief.ratings?.best?.find((b) => b.id === id);
    return bestEntry?.rating_milli;
  }
}
