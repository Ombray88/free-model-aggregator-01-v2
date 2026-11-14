/**
 * Gemini wire protocol adapter for Node 22 standard library.
 * Translates and communicates with Gemini models via REST endpoints.
 */

import { getAuthHeaders } from './upstream.ts';

export type GeminiResult =
  | { ok: true; text: string; usage: { input: number; output: number }; stop: string }
  | { ok: false; status: number; code: string; message: string };

/**
 * Constructs a Gemini-compatible request body from OpenAI or Anthropic format inputs.
 */
export function buildGeminiBody(client: 'openai' | 'anthropic', body: any, modelId: string): any {
  const geminiBody: any = {};
  const generationConfig: any = {};
  let systemParts: any[] = [];

  if (body && body.system) {
    if (typeof body.system === 'string') {
      systemParts.push({ text: body.system });
    } else if (Array.isArray(body.system)) {
      for (const p of body.system) {
        if (typeof p === 'string') systemParts.push({ text: p });
        else if (p && p.text) systemParts.push({ text: p.text });
      }
    }
  }

  const contents: any[] = [];
  const rawMessages = body?.messages || [];

  for (const msg of rawMessages) {
    if (!msg) continue;
    const role = msg.role;

    if (client === 'openai' && (role === 'system' || role === 'developer')) {
      const content = msg.content;
      if (typeof content === 'string') {
        systemParts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (part && part.type === 'text' && part.text) {
            systemParts.push({ text: part.text });
          }
        }
      }
      continue;
    }

    const geminiRole = role === 'assistant' ? 'model' : 'user';
    const parts: any[] = [];
    const content = msg.content;

    if (typeof content === 'string') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!part) continue;
        if (part.type === 'text' || typeof part.text === 'string') {
          parts.push({ text: part.text || part.content || '' });
        } else if (client === 'openai' && part.type === 'image_url') {
          const url = part.image_url?.url || '';
          if (url.startsWith('data:')) {
            const match = url.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
            } else {
              parts.push({ text: url });
            }
          } else {
            parts.push({ fileData: { fileUri: url } });
          }
        } else if (client === 'anthropic' && part.type === 'image') {
          const source = part.source;
          if (source) {
            if (source.type === 'base64') {
              parts.push({ inlineData: { mimeType: source.media_type || 'image/jpeg', data: source.data } });
            } else if (source.type === 'url' || source.url) {
              parts.push({ fileData: { fileUri: source.url || source.file_uri } });
            }
          }
        } else if (part.inlineData || part.fileData) {
          parts.push(part);
        } else if (part.text) {
          parts.push({ text: part.text });
        }
      }
    } else if (content && typeof content === 'object') {
      if (content.type === 'image') {
        const source = content.source;
        if (source?.type === 'base64') {
          parts.push({ inlineData: { mimeType: source.media_type || 'image/jpeg', data: source.data } });
        } else if (source?.url) {
          parts.push({ fileData: { fileUri: source.url } });
        }
      } else if (content.text) {
        parts.push({ text: content.text });
      }
    }

    if (client === 'openai' && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type === 'function' && tc.function) {
          let args = tc.function.arguments;
          if (typeof args === 'string') {
            try { args = JSON.parse(args); } catch {}
          }
          parts.push({
            functionCall: {
              name: tc.function.name,
              args: args || {}
            }
          });
        }
      }
    }

    if (client === 'openai' && role === 'tool') {
      parts.push({
        functionResponse: {
          name: msg.name || 'tool',
          response: { output: content }
        }
      });
    }

    if (client === 'anthropic' && Array.isArray(content)) {
      for (const block of content) {
        if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: block.name,
              args: block.input || {}
            }
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: block.tool_use_id || 'tool',
              response: { output: block.content }
            }
          });
        }
      }
    }

    if (parts.length > 0) {
      contents.push({ role: geminiRole, parts });
    }
  }

  if (systemParts.length > 0) {
    geminiBody.systemInstruction = { parts: systemParts };
  }

  if (contents.length > 0) {
    geminiBody.contents = contents;
  }

  const rawTools = body?.tools;
  const functionDeclarations: any[] = [];
  if (Array.isArray(rawTools)) {
    for (const t of rawTools) {
      if (!t) continue;
      if (client === 'openai') {
        if (t.type === 'function' && t.function) {
          functionDeclarations.push({
            name: t.function.name,
            description: t.function.description || '',
            parameters: t.function.parameters || { type: 'object', properties: {} }
          });
        }
      } else if (client === 'anthropic') {
        if (t.name) {
          functionDeclarations.push({
            name: t.name,
            description: t.description || '',
            parameters: t.input_schema || t.parameters || { type: 'object', properties: {} }
          });
        }
      }
    }
  }

  if (functionDeclarations.length > 0) {
    geminiBody.tools = [{ functionDeclarations }];
  }

  const maxTokens = body?.max_tokens ?? body?.maxTokens ?? body?.max_completion_tokens;
  if (typeof maxTokens === 'number') {
    generationConfig.maxOutputTokens = maxTokens;
  }

  const stop = body?.stop ?? body?.stop_sequences;
  if (stop) {
    if (Array.isArray(stop)) {
      generationConfig.stopSequences = stop;
    } else if (typeof stop === 'string') {
      generationConfig.stopSequences = [stop];
    }
  }

  const temp = body?.temperature;
  if (typeof temp === 'number') {
    generationConfig.temperature = temp;
  }

  const topP = body?.top_p ?? body?.topP;
  if (typeof topP === 'number') {
    generationConfig.topP = topP;
  }

  const responseFormat = body?.response_format;
  if (responseFormat) {
    if (responseFormat.type === 'json_object' || responseFormat.type === 'json') {
      generationConfig.responseMimeType = 'application/json';
    }
  }
  if (body?.response_mime_type) {
    generationConfig.responseMimeType = body.response_mime_type;
  }

  if (Object.keys(generationConfig).length > 0) {
    geminiBody.generationConfig = generationConfig;
  }

  return geminiBody;
}

/**
 * Sends a request to the Gemini API endpoint.
 */
export async function callGemini(
  baseUrl: string,
  key: string,
  modelId: string,
  body: any,
  stream: boolean,
  scheme: string = 'x-goog-api-key'
): Promise<GeminiResult> {
  try {
    const cleanBase = (baseUrl || 'https://generativelanguage.googleapis.com').replace(/\/+$/, '');
    const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    const url = `${cleanBase}/v1beta/models/${encodeURIComponent(modelId)}${action}`;

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        ...getAuthHeaders(scheme, key),
        'content-type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    const text = await res.text();

    if (!res.ok) {
      return {
        ok: false,
        status: res.status,
        code: 'http_' + res.status,
        message: text.slice(0, 300)
      };
    }

    if (stream) {
      const parsed = parseGeminiSse(text);
      if (!parsed) {
        return {
          ok: false,
          status: res.status,
          code: 'parse_error',
          message: 'Failed to parse SSE stream response'
        };
      }
      return parsed;
    } else {
      let json: any;
      try {
        json = JSON.parse(text);
      } catch (err: any) {
        return {
          ok: false,
          status: res.status,
          code: 'json_error',
          message: (err?.message || 'Invalid JSON').slice(0, 300)
        };
      }
      const parsed = parseGeminiJson(json);
      if (!parsed) {
        return {
          ok: false,
          status: res.status,
          code: 'parse_error',
          message: 'Invalid Gemini JSON structure'
        };
      }
      return parsed;
    }
  } catch (err: any) {
    return {
      ok: false,
      status: 0,
      code: 'network',
      message: (err?.message || String(err)).slice(0, 300)
    };
  }
}

/**
 * Parses standard Gemini JSON response structure into a unified result.
 */
export function parseGeminiJson(value: unknown): GeminiResult | null {
  if (!value || typeof value !== 'object') return null;
  const obj = value as any;
  const candidates = obj.candidates;
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return null;
  }

  const candidate = candidates[0];
  const parts = candidate?.content?.parts;
  let text = '';
  if (Array.isArray(parts)) {
    const texts: string[] = [];
    for (const p of parts) {
      if (p && typeof p.text === 'string') {
        texts.push(p.text);
      } else if (p && p.functionCall) {
        texts.push(JSON.stringify({ functionCall: p.functionCall }));
      }
    }
    text = texts.join('\n');
  } else if (typeof candidate?.output === 'string') {
    text = candidate.output;
  }

  const finishReason = typeof candidate?.finishReason === 'string' ? candidate.finishReason : 'STOP';
  const usageMeta = obj.usageMetadata || {};
  const input = typeof usageMeta.promptTokenCount === 'number' ? usageMeta.promptTokenCount : 0;
  const output = typeof usageMeta.candidatesTokenCount === 'number' ? usageMeta.candidatesTokenCount : 0;

  return {
    ok: true,
    text,
    usage: { input, output },
    stop: finishReason
  };
}

/**
 * Parses SSE text output from streaming Gemini calls.
 */
export function parseGeminiSse(rawText: string): GeminiResult | null {
  if (!rawText || typeof rawText !== 'string') return null;
  const chunks = rawText.split(/\r?\n\r?\n/);
  const textPieces: string[] = [];
  let lastUsage: any = null;
  let lastFinishReason: string = 'STOP';
  let hasData = false;

  for (const chunk of chunks) {
    const lines = chunk.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) {
        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr || jsonStr === '[DONE]') continue;
        try {
          const parsed = JSON.parse(jsonStr);
          hasData = true;
          const candRes = parseGeminiJson(parsed);
          if (candRes && candRes.ok) {
            if (candRes.text) {
              textPieces.push(candRes.text);
            }
            if (candRes.stop && candRes.stop !== 'STOP') {
              lastFinishReason = candRes.stop;
            }
          }
          if (parsed.usageMetadata) {
            lastUsage = parsed.usageMetadata;
          }
        } catch {}
      }
    }
  }

  if (!hasData && textPieces.length === 0) {
    return null;
  }

  const text = textPieces.join('');
  const input = typeof lastUsage?.promptTokenCount === 'number' ? lastUsage.promptTokenCount : 0;
  const output = typeof lastUsage?.candidatesTokenCount === 'number' ? lastUsage.candidatesTokenCount : 0;

  return {
    ok: true,
    text,
    usage: { input, output },
    stop: lastFinishReason
  };
}
