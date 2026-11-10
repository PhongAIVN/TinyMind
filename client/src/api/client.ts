// Thin wrapper over the TinyMind backend (proxied at /api by Vite).

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  chat: boolean;
  think: boolean;
  tools: boolean;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: string;
}

export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  thinking: string;
  tool_calls: ToolCall[];
  model: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  created_at: number;
  updated_at: number;
}

export interface ToolInfo {
  name: string;
  description: string;
  server: string;
}

export interface Health {
  status: string;
  device: string;
  mcp_errors: Record<string, string>;
}

export interface ChatOptions {
  model: string;
  think: boolean;
  temperature: number;
  max_new_tokens: number;
}

export type ChatEvent =
  | { type: 'start'; conversation_id: string; user_message: Message }
  | { type: 'think' | 'text'; text: string }
  | ({ type: 'tool' } & ToolCall)
  | { type: 'done'; message: Message | null }
  | { type: 'error'; message: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>('/health'),
  models: () => request<ModelInfo[]>('/models'),
  tools: () => request<ToolInfo[]>('/tools'),
  conversations: () => request<Conversation[]>('/conversations'),
  conversation: (id: string) => request<Conversation & { messages: Message[] }>(`/conversations/${id}`),
  rename: (id: string, title: string) =>
    request(`/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }),
  remove: (id: string) => request(`/conversations/${id}`, { method: 'DELETE' }),
  removeAll: () => request<{ deleted: number }>('/conversations', { method: 'DELETE' }),
};

/** POST /api/chat and call onEvent for every server-sent event until the stream ends. */
export async function streamChat(
  message: string,
  conversationId: string | null,
  options: ChatOptions,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id: conversationId, ...options }),
    signal,
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';
    for (const frame of frames) {
      const data = frame.replace(/^data: /, '').trim();
      if (data) onEvent(JSON.parse(data) as ChatEvent);
    }
  }
}
