import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { api, streamChat, type ChatEvent, type Message } from '../api/client';
import ChatBubble from '../components/chat/ChatBubble';
import MessageBody from '../components/chat/MessageBody';
import PromptBar from '../components/shared/PromptBar';
import { APP_NAME } from '../config/app_config';
import { useChatSettings } from '../state/ChatSettings';
import './ChatPage.css';

/** A message on screen; the streaming reply has id 0 until the server saves it. */
type ViewMessage = Pick<Message, 'role' | 'content' | 'thinking' | 'tool_calls'> & { id: number; model?: string | null };

const emptyReply = (): ViewMessage => ({ id: 0, role: 'assistant', content: '', thinking: '', tool_calls: [] });

function applyEvent(reply: ViewMessage, event: ChatEvent): ViewMessage {
  switch (event.type) {
    case 'text':
      return { ...reply, content: reply.content + event.text };
    case 'think':
      return { ...reply, thinking: reply.thinking + event.text };
    case 'tool':
      return { ...reply, tool_calls: [...reply.tool_calls, { name: event.name, arguments: event.arguments, result: event.result }] };
    default:
      return reply;
  }
}

const ChatPage: React.FC = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { options, models, bumpHistory } = useChatSettings();
  const [messages, setMessages] = useState<ViewMessage[]>([]);
  const [title, setTitle] = useState('');
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState('');
  const [memoryNote, setMemoryNote] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Id this page created mid-stream: navigating to it must not reload (and wipe) the live reply.
  const createdIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!id) {
      // Don't wipe a reply that is streaming (StrictMode re-runs this effect right after the first send).
      if (!abortRef.current) {
        setMessages([]);
        setTitle('');
      }
      return;
    }
    if (id === createdIdRef.current) return;
    const controller = new AbortController();
    api.conversation(id)
      .then((conversation) => {
        if (controller.signal.aborted) return;
        setMessages(conversation.messages);
        setTitle(conversation.title);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const send = async (text: string) => {
    setError('');
    setMemoryNote('');
    setIsBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const tempUser: ViewMessage = { id: -Date.now(), role: 'user', content: text, thinking: '', tool_calls: [] };
    setMessages((previous) => [...previous, tempUser, emptyReply()]);
    const updateReply = (change: (reply: ViewMessage) => ViewMessage) =>
      setMessages((previous) => {
        const last = previous[previous.length - 1];
        return last?.id === 0
          ? [...previous.slice(0, -1), change(last)]
          : [...previous, change(emptyReply())];
      });

    try {
      await streamChat(text, id ?? null, options, (event) => {
        if (event.type === 'start') {
          if (!id) {
            createdIdRef.current = event.conversation_id;
            setTitle(text.slice(0, 60));
            navigate(`/chat/${event.conversation_id}`, { replace: true });
          }
          bumpHistory();
        } else if (event.type === 'done') {
          if (event.message) updateReply(() => event.message as Message);
        } else if (event.type === 'memory') {
          setMemoryNote(`Đã ghi nhớ: ${event.items.map((m) => m.value).join(', ')}`);
        } else if (event.type === 'error') {
          setError(event.message);
        } else {
          updateReply((reply) => applyEvent(reply, event));
        }
      }, controller.signal);
    } catch (err: unknown) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally {
      // Drop an empty placeholder if nothing came back.
      setMessages((previous) => {
        const last = previous[previous.length - 1];
        const isEmpty = last?.id === 0 && !last.content && !last.thinking && last.tool_calls.length === 0;
        return isEmpty ? previous.slice(0, -1) : previous;
      });
      setIsBusy(false);
      abortRef.current = null;
    }
  };

  // Message typed on the home page: send once, then clear it from history state.
  const sentInitialRef = useRef<string | null>(null);
  useEffect(() => {
    const initial = (location.state as { initialMessage?: string } | null)?.initialMessage;
    // StrictMode runs effects twice in dev; the location key makes the send happen once.
    if (initial && !id && sentInitialRef.current !== location.key) {
      sentInitialRef.current = location.key;
      navigate(location.pathname, { replace: true, state: null });
      void send(initial);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once per navigation with initialMessage
  }, [location.state]);

  const modelLabel = models.find((m) => m.id === options.model)?.label ?? APP_NAME;

  return (
    <div className="chat-page-container">
      <div className="chat-header">
        <div className="chat-title">
          <span className="font-headline-md font-medium text-on-surface">{title || 'Cuộc trò chuyện mới'}</span>
          <span className="font-label-md text-on-surface-variant">· {modelLabel}</span>
        </div>
      </div>

      <div className="chat-scroll-area custom-scrollbar" ref={scrollRef}>
        <div className="chat-content">
          {messages.length === 0 && !isBusy && (
            <p className="font-body-md text-on-surface-variant chat-empty">Hỏi gì đó để bắt đầu.</p>
          )}
          {messages.map((message, index) => (
            <ChatBubble key={message.id || `live-${index}`} isUser={message.role === 'user'}>
              {message.role === 'user'
                ? <p className="message-text">{message.content}</p>
                : <MessageBody
                    content={message.content}
                    thinking={message.thinking}
                    toolCalls={message.tool_calls}
                    isStreaming={isBusy && message.id === 0}
                  />}
            </ChatBubble>
          ))}
          {memoryNote && (
            <p className="chat-memory font-label-md" role="status">
              <span className="material-symbols-outlined" aria-hidden="true">bookmark</span>{memoryNote}
            </p>
          )}
          {error && <p className="chat-error font-label-md" role="alert">{error}</p>}
        </div>
      </div>

      <div className="chat-prompt-area">
        <PromptBar onSubmit={send} isBusy={isBusy} onStop={() => abortRef.current?.abort()} />
      </div>
    </div>
  );
};

export default ChatPage;
