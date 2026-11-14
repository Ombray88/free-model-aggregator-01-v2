// Constructs the planner model pool from directory, probe, and rating data.
import type { PlannerCandidate } from './planner.ts';
import { capabilitiesOf, modelFree, ratingOfModel } from './providers.ts';
import type { Brief, Probe } from './providers.ts';

export function buildPool(brief: Brief, probes: readonly Probe[]): PlannerCandidate[] {
  const pool: PlannerCandidate[] = [];
  for (const entry of brief.directory.providers) {
    const probe = probes.find((p) => p.provider.id === entry.id);
    let added = 0;
    if (probe && probe.ok && probe.models.length > 0) {
      for (const model of probe.models) {
        const row = ratingOfModel(brief.ratings, entry.id, model.id);
        if (!modelFree(model, row)) continue;
        const caps = capabilitiesOf(model, row);
        pool.push({ providerId: entry.id, modelId: model.id, wire: entry.wire, baseUrl: entry.base, key: entry.key, auth: entry.auth, rating: row ? row.rating_milli : 0, tools: caps.tools, vision: caps.vision, json: caps.json, contextLength: caps.context_length });
        added++;
      }
    }
    if (added === 0) {
      for (const row of brief.ratings.ratings) {
        if (row.provider !== entry.id || !row.free) continue;
        pool.push({ providerId: entry.id, modelId: row.model, wire: entry.wire, baseUrl: entry.base, key: entry.key, auth: entry.auth, rating: row.rating_milli, tools: row.tools, vision: row.vision, json: row.json, contextLength: row.context_length });
      }
    }
  }
  return pool;
}
