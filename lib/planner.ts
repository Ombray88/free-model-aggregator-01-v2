export type WireType = 'openai' | 'anthropic' | 'gemini';

export interface PlannerCandidate {
  providerId: string;
  modelId: string;
  wire: WireType;
  baseUrl: string;
  key: string;
  auth: string;
  rating: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  contextLength: number | null;
}

export type ApiType = 'openai' | 'anthropic';

export interface PlannerRequestRequires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface PlannerRequest {
  id: string;
  api: ApiType;
  weight: number;
  best: number;
  requires: PlannerRequestRequires;
}

export interface PlannerState {
  calls: Record<string, number>;
  failedNumbers: Record<string, number[]>;
  dead: Record<string, boolean>;
  loss: number;
}

const RISK_TABLE: Record<string, number[]> = {
  groq: [0, 0.27, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.20, 0.39, 0.58, 0.78, 1, 1, 1, 1],
  cerebras: [0, 0.51, 0.45, 0.47, 0.26, 0.55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  gemini: [0, 0.39, 0.32, 0.33, 0.35, 0.25, 0.47, 0.70, 1, 1, 1, 1, 1, 1, 1, 1],
  zai: [0, 0.68, 0.63, 0.31, 0.55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  openrouter: [0, 0.26, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.23, 0.19, 0.44, 0.70, 1, 1, 1, 1],
  sambanova: [0, 0.68, 0.63, 0.31, 0.55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  pollinations: [0, 0.25, 0.16, 0.17, 0.18, 0.19, 0.20, 0.20, 0.21, 0.19, 0.35, 0.50, 0.66, 0.82, 1, 1],
  cloudflare: [0, 0.36, 0.29, 0.30, 0.31, 0.33, 0.21, 0.55, 1, 1, 1, 1, 1, 1, 1, 1],
  ovh: [0, 0.51, 0.45, 0.47, 0.26, 0.55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  kilol: [0, 0.22, 0.13, 0.14, 0.15, 0.16, 0.16, 0.17, 0.18, 0.19, 0.19, 0.17, 0.34, 0.50, 0.66, 0.82]
};

const PROVIDER_IDS = [
  'groq',
  'cerebras',
  'gemini',
  'zai',
  'openrouter',
  'sambanova',
  'pollinations',
  'cloudflare',
  'ovh',
  'kilol'
];

export function createPlannerState(): PlannerState {
  const calls: Record<string, number> = {};
  const failedNumbers: Record<string, number[]> = {};
  const dead: Record<string, boolean> = {};

  for (const id of PROVIDER_IDS) {
    calls[id] = 0;
    failedNumbers[id] = [];
    dead[id] = false;
  }

  return {
    calls,
    failedNumbers,
    dead,
    loss: 0
  };
}

export function riskFor(state: PlannerState, providerId: string): number {
  if (state.dead[providerId]) {
    return 1;
  }
  const calls = state.calls[providerId] || 0;
  const table = RISK_TABLE[providerId];
  if (!table) {
    return 1;
  }
  const callNum = calls + 1;
  if (callNum > table.length) {
    return 1;
  }
  const baseRisk = table[callNum - 1];

  const failedCount = state.failedNumbers[providerId]?.length || 0;
  if (failedCount >= 2) {
    return Math.min(baseRisk, 0.08);
  } else if (failedCount === 1) {
    return baseRisk / 2;
  }
  return baseRisk;
}

export function lossFor(request: PlannerRequest, candidate: PlannerCandidate): number {
  const diff = request.best - Math.min(candidate.rating, request.best);
  return request.weight * diff;
}

export function isEligible(request: PlannerRequest, candidate: PlannerCandidate): boolean {
  if (request.requires.tools && !candidate.tools) {
    return false;
  }
  if (request.requires.vision && !candidate.vision) {
    return false;
  }
  if (request.requires.json && !candidate.json) {
    return false;
  }
  if (request.requires.minContext !== undefined && request.requires.minContext !== null) {
    if (candidate.contextLength === null || candidate.contextLength < request.requires.minContext) {
      return false;
    }
  }
  if (request.api === 'openai' && request.requires.json) {
    if (candidate.wire !== 'openai' && candidate.wire !== 'gemini') {
      return false;
    }
  }
  return true;
}

export function buildPlan(
  requests: PlannerRequest[],
  pool: PlannerCandidate[],
  qualityBudget: number
): Map<string, PlannerCandidate> {
  const plan = new Map<string, PlannerCandidate>();

  // Order requests by scarcity:
  // 1. fewest eligible models in pool first
  // 2. ties by higher weight first
  // 3. then by the order the requests appear in the input list
  const sortedRequests = requests.map((req, index) => {
    const eligibleCount = pool.filter(c => isEligible(req, c)).length;
    return { req, index, eligibleCount };
  }).sort((a, b) => {
    if (a.eligibleCount !== b.eligibleCount) {
      return a.eligibleCount - b.eligibleCount;
    }
    if (b.req.weight !== a.req.weight) {
      return b.req.weight - a.req.weight;
    }
    return a.index - b.index;
  }).map(item => item.req);

  const pretendCalls: Record<string, number> = {};
  for (const id of PROVIDER_IDS) {
    pretendCalls[id] = 0;
  }
  let pretendLoss = 0;

  for (const request of sortedRequests) {
    const eligibleCandidates = pool.filter(c => isEligible(request, c));
    if (eligibleCandidates.length === 0) {
      continue;
    }

    // Evaluate each eligible candidate for pretend risk and loss
    const evaluated = eligibleCandidates.map(candidate => {
      const pCalls = pretendCalls[candidate.providerId] || 0;
      const table = RISK_TABLE[candidate.providerId];
      let risk = 1;
      const callNum = pCalls + 1;
      if (table && callNum <= table.length) {
        risk = table[callNum - 1];
      }
      const loss = lossFor(request, candidate);
      const fits = (pretendLoss + loss) <= qualityBudget;
      return { candidate, risk, loss, fits };
    });

    const fitting = evaluated.filter(e => e.fits);
    const poolToChoose = fitting.length > 0 ? fitting : evaluated;

    poolToChoose.sort((a, b) => {
      if (a.risk !== b.risk) {
        return a.risk - b.risk;
      }
      if (a.loss !== b.loss) {
        return a.loss - b.loss;
      }
      const pCompare = a.candidate.providerId.localeCompare(b.candidate.providerId);
      if (pCompare !== 0) {
        return pCompare;
      }
      return a.candidate.modelId.localeCompare(b.candidate.modelId);
    });

    const chosen = poolToChoose[0];
    plan.set(request.id, chosen.candidate);

    pretendCalls[chosen.candidate.providerId] = (pretendCalls[chosen.candidate.providerId] || 0) + 1;
    pretendLoss += chosen.loss;
  }

  return plan;
}

export function chooseNext(
  state: PlannerState,
  request: PlannerRequest,
  pool: PlannerCandidate[],
  tried: Set<string>,
  qualityBudget: number
): PlannerCandidate | null {
  const survivors: Array<{
    candidate: PlannerCandidate;
    risk: number;
    loss: number;
    fits: boolean;
  }> = [];

  for (const candidate of pool) {
    const key = `${candidate.providerId}:${candidate.modelId}`;
    if (tried.has(key)) {
      continue;
    }
    if (!isEligible(request, candidate)) {
      continue;
    }
    if (state.dead[candidate.providerId]) {
      continue;
    }

    const risk = riskFor(state, candidate.providerId);
    if (risk === 1) {
      continue;
    }

    const loss = lossFor(request, candidate);
    const fits = (state.loss + loss) <= qualityBudget;

    survivors.push({ candidate, risk, loss, fits });
  }

  if (survivors.length === 0) {
    return null;
  }

  survivors.sort((a, b) => {
    if (a.fits !== b.fits) {
      return a.fits ? -1 : 1;
    }

    if (a.fits) {
      if (a.risk !== b.risk) {
        return a.risk - b.risk;
      }
      if (a.loss !== b.loss) {
        return a.loss - b.loss;
      }
    } else {
      if (a.loss !== b.loss) {
        return a.loss - b.loss;
      }
      if (a.risk !== b.risk) {
        return a.risk - b.risk;
      }
    }

    const pCompare = a.candidate.providerId.localeCompare(b.candidate.providerId);
    if (pCompare !== 0) {
      return pCompare;
    }
    return a.candidate.modelId.localeCompare(b.candidate.modelId);
  });

  return survivors[0].candidate;
}

export function recordCall(state: PlannerState, providerId: string): number {
  state.calls[providerId] = (state.calls[providerId] || 0) + 1;
  return state.calls[providerId];
}

export function recordOutcome(
  state: PlannerState,
  providerId: string,
  callNumber: number,
  ok: boolean,
  failureText: string
): void {
  if (ok) {
    return;
  }

  if (!state.failedNumbers[providerId]) {
    state.failedNumbers[providerId] = [];
  }
  if (!state.failedNumbers[providerId].includes(callNumber)) {
    state.failedNumbers[providerId].push(callNumber);
  }

  const text = (failureText || '').toLowerCase();
  if (text.includes('budget')) {
    state.dead[providerId] = true;
  } else if (text.includes('is out')) {
    state.dead[providerId] = true;
  }
}
