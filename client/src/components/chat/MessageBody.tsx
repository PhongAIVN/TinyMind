import React from 'react';
import type { ToolCall } from '../../api/client';
import CodeBlock from './CodeBlock';
import './MessageBody.css';

interface MessageBodyProps {
  content: string;
  thinking?: string;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

const FENCE = /```(\w*)\n?([\s\S]*?)(?:```|$)/g;

/** Split text into plain parts and ``` code fences (an unclosed fence while streaming still renders as code). */
function splitCode(text: string): { kind: 'text' | 'code'; value: string; language: string }[] {
  const parts: { kind: 'text' | 'code'; value: string; language: string }[] = [];
  let last = 0;
  for (const match of text.matchAll(FENCE)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: 'text', value: text.slice(last, index), language: '' });
    parts.push({ kind: 'code', value: match[2].trimEnd(), language: match[1] || 'code' });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: 'text', value: text.slice(last), language: '' });
  return parts;
}

const MessageBody: React.FC<MessageBodyProps> = ({ content, thinking, toolCalls, isStreaming }) => {
  const isThinkingNow = isStreaming && !!thinking && !content;
  return (
    <div className="message-body">
      {thinking && (
        <details className="thinking-block" open={isThinkingNow}>
          <summary className="font-label-md">
            <span className="material-symbols-outlined">psychology</span>
            {isThinkingNow ? 'Đang suy nghĩ…' : 'Xem quá trình suy nghĩ'}
          </summary>
          <p className="thinking-text">{thinking}</p>
        </details>
      )}

      {toolCalls?.map((call, index) => (
        <div key={index} className="tool-call font-label-md">
          <span className="material-symbols-outlined">build</span>
          <code>{call.name}({Object.values(call.arguments).map(String).join(', ')})</code>
          <span className="material-symbols-outlined arrow">arrow_forward</span>
          <code className="tool-result">{call.result}</code>
        </div>
      ))}

      {splitCode(content).map((part, index) =>
        part.kind === 'code'
          ? <CodeBlock key={index} language={part.language} code={part.value} />
          : <p key={index} className="message-text">{part.value}</p>
      )}

      {isStreaming && !content && !thinking && <span className="typing-dots" aria-label="Đang trả lời"><i /><i /><i /></span>}
    </div>
  );
};

export default MessageBody;
