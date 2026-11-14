/**
 * Upstream provider communication and response parsing module for OpenAI and Anthropic wires.
 */

export type UpstreamResult =
  | {
      ok: true;
      text: string;
      usage: { input: number; output: number };
      stop: string;
      toolCalls: Array<{ id: string; name: string; arguments: string }>;
    }
  | {
      ok: false;
      status: number;
      code: string;
      message: string;
    };

/**
 * Returns whether an HTTP status code is retryable (402, 429, or 5xx).
 */
export function isRetryable(status: number): boolean {
  if (status === 402 || status === 429) {
    return true;
  }
  if (status >= 500 && status < 600) {
    return true;
  }
  return false;
}

/**
 * Parses an OpenAI JSON response payload into a normalized UpstreamResult.
 */
export function parseOpenAIJson(value: unknown): UpstreamResult | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const obj = value as Record<string, any>;
  const choices = obj.choices;
  const usageObj = obj.usage;

  const hasChoices = Array.isArray(choices) && choices.length > 0;
  const choice = hasChoices ? choices[0] : null;
  const message = choice?.message;

  let text = '';
  if (message && message.content) {
    if (typeof message.content === 'string') {
      text = message.content;
    } else if (Array.isArray(message.content)) {
      for (const part of message.content) {
        if (part && typeof part === 'object' && typeof part.text === 'string') {
          text += part.text;
        }
      }
    }
  }

  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
  if (message && Array.isArray(message.tool_calls)) {
    for (const tc of message.tool_calls) {
      if (tc && typeof tc === 'object') {
        const id = typeof tc.id === 'string' ? tc.id : '';
        const fn = tc.function || {};
        const name = typeof fn.name === 'string' ? fn.name : '';
        let argsStr = '{}';
        if (fn.arguments !== undefined && fn.arguments !== null) {
          if (typeof fn.arguments === 'string') {
            argsStr = fn.arguments;
          } else {
            try {
              argsStr = JSON.stringify(fn.arguments);
            } catch {
              argsStr = '{}';
            }
          }
        }
        toolCalls.push({ id, name, arguments: argsStr });
      }
    }
  }

  const hasText = text.length > 0;
  const hasToolCalls = toolCalls.length > 0;

  if (!hasText && !hasToolCalls && !choice?.finish_reason && !usageObj) {
    return null;
  }

  const stop = typeof choice?.finish_reason === 'string' && choice.finish_reason.length > 0 ? choice.finish_reason : 'stop';
  const inputTokens = Number(usageObj?.prompt_tokens ?? usageObj?.input_tokens ?? 0) || 0;
  const outputTokens = Number(usageObj?.completion_tokens ?? usageObj?.output_tokens ?? 0) || 0;

  return {
    ok: true,
    text,
    usage: { input: inputTokens, output: outputTokens },
    stop,
    toolCalls,
  };
}

/**
 * Parses raw OpenAI server-sent events text into a normalized UpstreamResult.
 */
export function parseOpenAISse(rawText: string): UpstreamResult | null {
  if (!rawText || typeof rawText !== 'string') {
    return null;
  }

  let text = '';
  let stop = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  const toolCallsMap = new Map<number, { id: string; name: string; argumentsParts: string[] }>();
  let foundAnyContent = false;

  const lines = rawText.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) {
      continue;
    }
    const dataContent = trimmed.slice(5).trim();
    if (dataContent === '[DONE]' || dataContent === '') {
      continue;
    }

    let parsed: any;
    try {
      parsed = JSON.parse(dataContent);
    } catch {
      continue;
    }

    if (!parsed || typeof parsed !== 'object') {
      continue;
    }

    if (parsed.usage && typeof parsed.usage === 'object') {
      inputTokens = Number(parsed.usage.prompt_tokens ?? parsed.usage.input_tokens ?? inputTokens) || inputTokens;
      outputTokens = Number(parsed.usage.completion_tokens ?? parsed.usage.output_tokens ?? outputTokens) || outputTokens;
      foundAnyContent = true;
    }

    const choices = parsed.choices;
    if (Array.isArray(choices) && choices.length > 0) {
      const choice = choices[0];
      if (choice) {
        if (typeof choice.finish_reason === 'string' && choice.finish_reason.length > 0) {
          stop = choice.finish_reason;
          foundAnyContent = true;
        }

        const delta = choice.delta;
        if (delta && typeof delta === 'object') {
          if (typeof delta.content === 'string') {
            text += delta.content;
            foundAnyContent = true;
          } else if (Array.isArray(delta.content)) {
            for (const part of delta.content) {
              if (part && typeof part === 'object' && typeof part.text === 'string') {
                text += part.text;
                foundAnyContent = true;
              }
            }
          }

          if (Array.isArray(delta.tool_calls)) {
            for (const tc of delta.tool_calls) {
              if (tc && typeof tc === 'object') {
                const idx = typeof tc.index === 'number' ? tc.index : 0;
                let entry = toolCallsMap.get(idx);
                if (!entry) {
                  entry = { id: '', name: '', argumentsParts: [] };
                  toolCallsMap.set(idx, entry);
                }
                if (typeof tc.id === 'string' && tc.id) {
                  entry.id = tc.id;
                }
                if (tc.function && typeof tc.function === 'object') {
                  if (typeof tc.function.name === 'string' && tc.function.name) {
                    entry.name = tc.function.name;
                  }
                  if (typeof tc.function.arguments === 'string' && tc.function.arguments) {
                    entry.argumentsParts.push(tc.function.arguments);
                  }
                }
                foundAnyContent = true;
              }
            }
          }
        }
      }
    }
  }

  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
  const sortedIndices = Array.from(toolCallsMap.keys()).sort((a, b) => a - b);
  for (const idx of sortedIndices) {
    const entry = toolCallsMap.get(idx)!;
    toolCalls.push({
      id: entry.id,
      name: entry.name,
      arguments: entry.argumentsParts.join(''),
    });
  }

  if (!foundAnyContent && text.length === 0 && toolCalls.length === 0) {
    return null;
  }

  if (text.length === 0 && toolCalls.length === 0 && !inputTokens && !outputTokens) {
    return null;
  }

  return {
    ok: true,
    text,
    usage: { input: inputTokens, output: outputTokens },
    stop,
    toolCalls,
  };
}

/**
 * Returns the matching header entries for a given auth scheme and key.
 */
export function getAuthHeaders(scheme: string, key: string): Record<string, string> {
  if (!key) return {};
  switch (scheme) {
    case 'bearer':
      return { 'Authorization': `Bearer ${key}` };
    case 'x-api-key':
      return { 'x-api-key': key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': key };
    default:
      return {};
  }
}

/**
 * Performs an HTTP request to an OpenAI-compatible chat completions endpoint.
 */
export async function callOpenAI(
  baseUrl: string,
  key: string,
  modelId: string,
  body: Record<string, any>,
  stream: boolean,
  scheme: string = 'bearer'
): Promise<UpstreamResult> {
  const url = baseUrl.replace(/\/+$/, '') + '/v1/chat/completions';
  const payload: Record<string, any> = {
    ...body,
    model: modelId,
    stream,
  };
  if (stream) {
    payload.stream_options = {
      ...(payload.stream_options || {}),
      include_usage: true,
    };
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(scheme, key),
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const respText = await response.text();
      return {
        ok: false,
        status: response.status,
        code: `http_${response.status}`,
        message: respText.slice(0, 300),
      };
    }

    const rawText = await response.text();
    if (stream) {
      const parsed = parseOpenAISse(rawText);
      if (!parsed) {
        return {
          ok: true,
          text: '',
          usage: { input: 0, output: 0 },
          stop: 'stop',
          toolCalls: [],
        };
      }
      return parsed;
    } else {
      let json: unknown;
      try {
        json = JSON.parse(rawText);
      } catch (err) {
        return {
          ok: false,
          status: 200,
          code: 'json_parse_error',
          message: err instanceof Error ? err.message : String(err),
        };
      }
      const parsed = parseOpenAIJson(json);
      if (!parsed) {
        return {
          ok: true,
          text: '',
          usage: { input: 0, output: 0 },
          stop: 'stop',
          toolCalls: [],
        };
      }
      return parsed;
    }
  } catch (err) {
    return {
      ok: false,
      status: 0,
      code: 'network',
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Parses an Anthropic JSON response payload into a normalized UpstreamResult.
 */
export function parseAnthropicJson(value: unknown): UpstreamResult | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const obj = value as Record<string, any>;
  const contentBlocks = obj.content;
  const usageObj = obj.usage;

  let text = '';
  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];

  if (Array.isArray(contentBlocks)) {
    for (const block of contentBlocks) {
      if (block && typeof block === 'object') {
        if (block.type === 'text' && typeof block.text === 'string') {
          text += block.text;
        } else if (block.type === 'tool_use') {
          const id = typeof block.id === 'string' ? block.id : '';
          const name = typeof block.name === 'string' ? block.name : '';
          let argsStr = '{}';
          if (block.input !== undefined && block.input !== null) {
            if (typeof block.input === 'string') {
              argsStr = block.input;
            } else {
              try {
                argsStr = JSON.stringify(block.input);
              } catch {
                argsStr = '{}';
              }
            }
          }
          toolCalls.push({ id, name, arguments: argsStr });
        }
      }
    }
  }

  const hasText = text.length > 0;
  const hasToolCalls = toolCalls.length > 0;

  if (!hasText && !hasToolCalls && !obj.stop_reason && !usageObj) {
    return null;
  }

  const stop = typeof obj.stop_reason === 'string' && obj.stop_reason.length > 0 ? obj.stop_reason : 'end_turn';
  const inputTokens = Number(usageObj?.input_tokens ?? usageObj?.prompt_tokens ?? 0) || 0;
  const outputTokens = Number(usageObj?.output_tokens ?? usageObj?.completion_tokens ?? 0) || 0;

  return {
    ok: true,
    text,
    usage: { input: inputTokens, output: outputTokens },
    stop,
    toolCalls,
  };
}

/**
 * Parses raw Anthropic server-sent events text into a normalized UpstreamResult.
 */
export function parseAnthropicSse(rawText: string): UpstreamResult | null {
  if (!rawText || typeof rawText !== 'string') {
    return null;
  }

  let text = '';
  let stop = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  let foundAnyContent = false;

  const toolCallsMap = new Map<number, { id: string; name: string; partialJson: string }>();

  const lines = rawText.split(/\r?\n/);
  let currentEvent = '';

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('event:')) {
      currentEvent = trimmed.slice(6).trim();
      continue;
    }
    if (trimmed.startsWith('data:')) {
      const dataContent = trimmed.slice(5).trim();
      if (!dataContent) {
        continue;
      }
      let parsed: any;
      try {
        parsed = JSON.parse(dataContent);
      } catch {
        continue;
      }

      if (!parsed || typeof parsed !== 'object') {
        continue;
      }

      const type = parsed.type || currentEvent;

      if (type === 'message_start' && parsed.message && typeof parsed.message === 'object') {
        if (parsed.message.usage && typeof parsed.message.usage === 'object') {
          inputTokens = Number(parsed.message.usage.input_tokens ?? inputTokens) || inputTokens;
          outputTokens = Number(parsed.message.usage.output_tokens ?? outputTokens) || outputTokens;
          foundAnyContent = true;
        }
      } else if (type === 'content_block_start') {
        const idx = typeof parsed.index === 'number' ? parsed.index : 0;
        const cb = parsed.content_block;
        if (cb && typeof cb === 'object') {
          if (cb.type === 'tool_use') {
            toolCallsMap.set(idx, {
              id: typeof cb.id === 'string' ? cb.id : '',
              name: typeof cb.name === 'string' ? cb.name : '',
              partialJson: '',
            });
            foundAnyContent = true;
          }
        }
      } else if (type === 'content_block_delta') {
        const idx = typeof parsed.index === 'number' ? parsed.index : 0;
        const delta = parsed.delta;
        if (delta && typeof delta === 'object') {
          if (delta.type === 'text_delta' && typeof delta.text === 'string') {
            text += delta.text;
            foundAnyContent = true;
          } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
            const entry = toolCallsMap.get(idx);
            if (entry) {
              entry.partialJson += delta.partial_json;
              foundAnyContent = true;
            }
          }
        }
      } else if (type === 'message_delta') {
        if (parsed.delta && typeof parsed.delta === 'object') {
          if (typeof parsed.delta.stop_reason === 'string' && parsed.delta.stop_reason.length > 0) {
            stop = parsed.delta.stop_reason;
            foundAnyContent = true;
          }
        }
        if (parsed.usage && typeof parsed.usage === 'object') {
          outputTokens = Number(parsed.usage.output_tokens ?? outputTokens) || outputTokens;
          foundAnyContent = true;
        }
      } else if (type === 'content_block_stop') {
        foundAnyContent = true;
      }
    }
  }

  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
  const sortedIndices = Array.from(toolCallsMap.keys()).sort((a, b) => a - b);
  for (const idx of sortedIndices) {
    const entry = toolCallsMap.get(idx)!;
    let args = entry.partialJson.trim();
    if (!args) {
      args = '{}';
    } else {
      try {
        const parsedJson = JSON.parse(args);
        args = JSON.stringify(parsedJson);
      } catch {
        // keep as is if unparsable
      }
    }
    toolCalls.push({
      id: entry.id,
      name: entry.name,
      arguments: args,
    });
  }

  if (!foundAnyContent && text.length === 0 && toolCalls.length === 0) {
    return null;
  }

  if (text.length === 0 && toolCalls.length === 0 && !inputTokens && !outputTokens) {
    return null;
  }

  return {
    ok: true,
    text,
    usage: { input: inputTokens, output: outputTokens },
    stop,
    toolCalls,
  };
}

/**
 * Performs an HTTP request to an Anthropic messages endpoint.
 */
export async function callAnthropic(
  baseUrl: string,
  key: string,
  modelId: string,
  body: Record<string, any>,
  stream: boolean,
  scheme: string = 'x-api-key'
): Promise<UpstreamResult> {
  const url = baseUrl.replace(/\/+$/, '') + '/v1/messages';
  const payload: Record<string, any> = {
    max_tokens: 4096,
    ...body,
    model: modelId,
    stream,
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(scheme, key),
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const respText = await response.text();
      return {
        ok: false,
        status: response.status,
        code: `http_${response.status}`,
        message: respText.slice(0, 300),
      };
    }

    const rawText = await response.text();
    if (stream) {
      const parsed = parseAnthropicSse(rawText);
      if (!parsed) {
        return {
          ok: true,
          text: '',
          usage: { input: 0, output: 0 },
          stop: 'end_turn',
          toolCalls: [],
        };
      }
      return parsed;
    } else {
      let json: unknown;
      try {
        json = JSON.parse(rawText);
      } catch (err) {
        return {
          ok: false,
          status: 200,
          code: 'json_parse_error',
          message: err instanceof Error ? err.message : String(err),
        };
      }
      const parsed = parseAnthropicJson(json);
      if (!parsed) {
        return {
          ok: true,
          text: '',
          usage: { input: 0, output: 0 },
          stop: 'end_turn',
          toolCalls: [],
        };
      }
      return parsed;
    }
  } catch (err) {
    return {
      ok: false,
      status: 0,
      code: 'network',
      message: err instanceof Error ? err.message : String(err),
    };
  }
}
