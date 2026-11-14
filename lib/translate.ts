import { buildGeminiBody } from './gemini.ts';
import type { GeminiResult } from './gemini.ts';
import type { UpstreamResult } from './upstream.ts';

export type ClientWire = 'openai' | 'anthropic';
export type ProviderWire = 'openai' | 'anthropic' | 'gemini';

type AnyResult = GeminiResult | UpstreamResult;

/**
 * Maps a provider stop reason to an OpenAI finish_reason.
 */
export function mapStopToOpenAI(stop: string, hasTools: boolean): string {
  if (hasTools) return 'tool_calls';
  const s = stop.toLowerCase();
  if (s === 'stop' || s === 'end_turn') return 'stop';
  if (s === 'max_tokens' || s === 'length') return 'length';
  return 'stop';
}

/**
 * Maps a provider stop reason to an Anthropic stop_reason.
 */
export function mapStopToAnthropic(stop: string, hasTools: boolean): string {
  if (hasTools) return 'tool_use';
  const s = stop.toLowerCase();
  if (s === 'max_tokens' || s === 'length') return 'max_tokens';
  if (s === 'stop' || s === 'end_turn') return 'end_turn';
  return 'end_turn';
}

/**
 * Translates a client request body into the format expected by the provider.
 */
export function buildUpstreamBody(
  clientWire: ClientWire,
  providerWire: ProviderWire,
  clientBody: any,
  modelId: string,
  stream: boolean
): any {
  if (providerWire === 'gemini') {
    return buildGeminiBody(clientWire, clientBody, modelId);
  }

  const body = JSON.parse(JSON.stringify(clientBody));
  body.model = modelId;
  body.stream = stream;

  if (clientWire === providerWire) {
    if (providerWire === 'openai' && stream) {
      body.stream_options = { include_usage: true };
    }
    return body;
  }

  if (clientWire === 'openai' && providerWire === 'anthropic') {
    // OpenAI -> Anthropic
    const anthropicBody: any = {
      model: modelId,
      stream: stream,
      max_tokens: body.max_tokens ?? 4096,
    };

    if (body.stop) {
      anthropicBody.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop];
    }

    const messages: any[] = [];
    for (const msg of body.messages || []) {
      if (msg.role === 'system') {
        anthropicBody.system = msg.content;
      } else {
        const content: any[] = [];
        if (Array.isArray(msg.content)) {
          for (const item of msg.content) {
            if (item.type === 'text') {
              content.push({ type: 'text', text: item.text });
            } else if (item.type === 'image_url') {
              const url = item.image_url.url;
              if (url.startsWith('data:')) {
                const match = url.match(/^data:([^;]+);base64,(.+)$/);
                if (match) {
                  content.push({
                    type: 'image',
                    source: {
                      type: 'base64',
                      media_type: match[1],
                      data: match[2],
                    },
                  });
                }
              } else {
                content.push({
                  type: 'image',
                  source: {
                    type: 'url',
                    url: url,
                  },
                });
              }
            }
          }
        } else if (typeof msg.content === 'string') {
          content.push({ type: 'text', text: msg.content });
        }
        messages.push({ role: msg.role, content });
      }
    }
    anthropicBody.messages = messages;

    if (body.tools) {
      anthropicBody.tools = body.tools.map((t: any) => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters,
      }));
    }

    return anthropicBody;
  }

  if (clientWire === 'anthropic' && providerWire === 'openai') {
    // Anthropic -> OpenAI
    const openaiBody: any = {
      model: modelId,
      stream: stream,
      max_tokens: body.max_tokens,
    };

    if (stream) {
      openaiBody.stream_options = { include_usage: true };
    }

    if (body.stop_sequences) {
      openaiBody.stop = body.stop_sequences;
    }

    const messages: any[] = [];
    if (body.system) {
      messages.push({ role: 'system', content: body.system });
    }

    for (const msg of body.messages || []) {
      const content: any[] = [];
      const tool_calls: any[] = [];
      
      const msgContent = Array.isArray(msg.content) ? msg.content : [{ type: 'text', text: msg.content }];
      
      for (const item of msgContent) {
        if (item.type === 'text') {
          content.push({ type: 'text', text: item.text });
        } else if (item.type === 'image') {
          if (item.source.type === 'base64') {
            content.push({
              type: 'image_url',
              image_url: { url: `data:${item.source.media_type};base64,${item.source.data}` },
            });
          } else if (item.source.type === 'url') {
            content.push({
              type: 'image_url',
              image_url: { url: item.source.url },
            });
          }
        } else if (item.type === 'tool_use') {
          tool_calls.push({
            id: item.id,
            type: 'function',
            function: {
              name: item.name,
              arguments: JSON.stringify(item.input),
            },
          });
        }
      }

      const openaiMsg: any = { role: msg.role, content: content.length === 1 && content[0].type === 'text' ? content[0].text : content };
      if (tool_calls.length > 0) {
        openaiMsg.tool_calls = tool_calls;
      }
      messages.push(openaiMsg);
    }
    openaiBody.messages = messages;

    if (body.tools) {
      openaiBody.tools = body.tools.map((t: any) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.input_schema,
        },
      }));
    }

    return openaiBody;
  }

  return body;
}

/**
 * Builds a standard OpenAI JSON response from a result.
 */
export function buildOpenAIJsonResponse(result: AnyResult, modelId: string): any {
  const toolCalls = (result as UpstreamResult).toolCalls || [];
  const finishReason = mapStopToOpenAI(result.stop, toolCalls.length > 0);

  const message: any = {
    role: 'assistant',
    content: result.text || null,
  };

  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls.map((tc) => ({
      id: tc.id,
      type: 'function',
      function: {
        name: tc.name,
        arguments: tc.arguments,
      },
    }));
  }

  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: 0,
    model: modelId,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: result.usage.input,
      completion_tokens: result.usage.output,
      total_tokens: result.usage.input + result.usage.output,
    },
  };
}

/**
 * Builds a standard Anthropic JSON response from a result.
 */
export function buildAnthropicJsonResponse(result: AnyResult, modelId: string): any {
  const toolCalls = (result as UpstreamResult).toolCalls || [];
  const stopReason = mapStopToAnthropic(result.stop, toolCalls.length > 0);

  const content: any[] = [];
  if (result.text) {
    content.push({ type: 'text', text: result.text });
  }

  for (const tc of toolCalls) {
    content.push({
      type: 'tool_use',
      id: tc.id,
      name: tc.name,
      input: JSON.parse(tc.arguments),
    });
  }

  return {
    id: 'msg-aggregator',
    type: 'message',
    role: 'assistant',
    content,
    model: modelId,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: result.usage.input,
      output_tokens: result.usage.output,
    },
  };
}

/**
 * Builds OpenAI-style Server-Sent Event frames.
 */
export function buildOpenAISseResponse(result: AnyResult, modelId: string): string {
  const toolCalls = (result as UpstreamResult).toolCalls || [];
  const finishReason = mapStopToOpenAI(result.stop, toolCalls.length > 0);
  const frames: any[] = [];

  const base = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: 0,
    model: modelId,
  };

  // Role chunk
  frames.push({
    ...base,
    choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
  });

  // Content chunk
  if (result.text) {
    frames.push({
      ...base,
      choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }],
    });
  }

  // Tool calls chunks
  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];
    frames.push({
      ...base,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: i,
                id: tc.id,
                type: 'function',
                function: { name: tc.name, arguments: tc.arguments },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    });
  }

  // Finish chunk
  frames.push({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  });

  // Usage chunk
  frames.push({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: result.usage.input,
      completion_tokens: result.usage.output,
      total_tokens: result.usage.input + result.usage.output,
    },
  });

  return frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join('') + 'data: [DONE]\n\n';
}

/**
 * Builds Anthropic-style Server-Sent Event frames.
 */
export function buildAnthropicSseResponse(result: AnyResult, modelId: string): string {
  const toolCalls = (result as UpstreamResult).toolCalls || [];
  const stopReason = mapStopToAnthropic(result.stop, toolCalls.length > 0);
  const lines: string[] = [];

  const send = (event: string, data: any) => {
    lines.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  send('message_start', {
    message: {
      id: 'msg-aggregator',
      type: 'message',
      role: 'assistant',
      content: [],
      model: modelId,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: result.usage.input, output_tokens: 0 },
    },
  });

  if (result.text) {
    send('content_block_start', { index: 0, content_block: { type: 'text', text: '' } });
    send('content_block_delta', { index: 0, delta: { type: 'text_delta', text: result.text } });
    send('content_block_stop', { index: 0 });
  }

  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];
    const index = (result.text ? 1 : 0) + i;
    send('content_block_start', {
      index,
      content_block: { type: 'tool_use', id: tc.id, name: tc.name, input: {} },
    });
    send('content_block_delta', {
      index,
      delta: { type: 'input_json_delta', partial_json: tc.arguments },
    });
    send('content_block_stop', { index });
  }

  send('message_delta', {
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { input_tokens: result.usage.input, output_tokens: result.usage.output },
  });

  send('message_stop', {});

  return lines.join('');
}
