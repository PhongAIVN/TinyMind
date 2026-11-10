import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, type Conversation } from '../api/client';
import { useChatSettings } from '../state/ChatSettings';
import './HistoryPage.css';

function formatTime(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' });
}

const HistoryPage: React.FC = () => {
  const { historyVersion, bumpHistory, models } = useChatSettings();
  const [conversations, setConversations] = useState<Conversation[] | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.conversations()
      .then(setConversations)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, [historyVersion]);

  const handleDelete = async (conversation: Conversation) => {
    if (!window.confirm(`Xóa "${conversation.title}"? Không khôi phục được.`)) return;
    await api.remove(conversation.id).catch(() => undefined);
    bumpHistory();
  };

  const handleRename = async (conversation: Conversation) => {
    const title = window.prompt('Tên mới', conversation.title)?.trim();
    if (!title || title === conversation.title) return;
    await api.rename(conversation.id, title).catch(() => undefined);
    bumpHistory();
  };

  const labelOf = (modelId: string) => models.find((m) => m.id === modelId)?.label ?? modelId;

  return (
    <div className="history-page">
      <div className="history-container">
        <h1 className="font-display-lg text-on-surface">Lịch sử</h1>
        {error && <p className="font-label-md history-error" role="alert">{error}</p>}
        {conversations?.length === 0 && (
          <p className="font-body-md text-on-surface-variant">Chưa có cuộc trò chuyện nào.</p>
        )}
        <ul className="history-list">
          {conversations?.map((conversation) => (
            <li key={conversation.id} className="history-item">
              <Link to={`/chat/${conversation.id}`} className="history-link">
                <span className="font-label-md history-title">{conversation.title}</span>
                <span className="font-label-sm text-on-surface-variant">
                  {labelOf(conversation.model)} · {formatTime(conversation.updated_at)}
                </span>
              </Link>
              <button type="button" className="history-action" title="Đổi tên" onClick={() => handleRename(conversation)}>
                <span className="material-symbols-outlined">edit</span>
              </button>
              <button type="button" className="history-action" title="Xóa" onClick={() => handleDelete(conversation)}>
                <span className="material-symbols-outlined">delete</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default HistoryPage;
