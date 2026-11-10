import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { api, type Health, type Memory } from '../api/client';
import BentoCard from '../components/shared/BentoCard';
import ToggleSwitch from '../components/shared/ToggleSwitch';
import { DISCLAIMER } from '../config/app_config';
import { useChatSettings } from '../state/ChatSettings';
import './SettingsPage.css';

const MEMORY_LABELS: Record<string, string> = {
  name: 'Tên', age: 'Tuổi', location: 'Sống ở', job: 'Nghề', likes: 'Thích', note: 'Ghi nhớ',
};

const SettingsPage: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { options, update, bumpHistory, models } = useChatSettings();
  const [deleteNote, setDeleteNote] = useState('');
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState('');
  const [memories, setMemories] = useState<Memory[]>([]);

  const loadMemories = () => {
    api.memories().then(setMemories).catch(() => setMemories([]));
  };

  const handleForget = async (memory: Memory) => {
    await api.forget(memory.id).catch(() => undefined);
    loadMemories();
  };

  const handleForgetAll = async () => {
    if (!window.confirm('Xóa toàn bộ trí nhớ? Không khôi phục được.')) return;
    await api.forgetAll().catch(() => undefined);
    loadMemories();
  };

  useEffect(() => {
    if (containerRef.current) {
      const cards = containerRef.current.querySelectorAll('.settings-card-anim');
      gsap.fromTo(cards,
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, stagger: 0.1, ease: 'power2.out', delay: 0.1 }
      );
    }
    api.health()
      .then(setHealth)
      .catch((err: unknown) => setHealthError(err instanceof Error ? err.message : String(err)));
    loadMemories();
  }, []);

  const handleDeleteAll = async () => {
    if (!window.confirm('Xóa toàn bộ lịch sử trò chuyện? Không khôi phục được.')) return;
    try {
      const { deleted } = await api.removeAll();
      setDeleteNote(`Đã xóa ${deleted} cuộc trò chuyện.`);
      bumpHistory();
    } catch (err: unknown) {
      setDeleteNote(err instanceof Error ? err.message : String(err));
    }
  };

  const mcpErrors = Object.entries(health?.mcp_errors ?? {});

  return (
    <div className="settings-page" ref={containerRef}>
      <div className="settings-container">
        <div className="settings-header">
          <h1 className="font-display-lg text-on-surface">Cài đặt</h1>
          <p className="font-body-md text-on-surface-variant">Tham số sinh chữ, lịch sử trò chuyện và trạng thái backend.</p>
        </div>

        <div className="settings-grid">
          <BentoCard className="settings-card-anim col-span-12 setting-card">
            <div className="card-header-small">
              <span className="material-symbols-outlined text-primary">tune</span>
              <h3 className="font-label-md uppercase tracking-wider text-on-surface-variant">Sinh chữ</h3>
            </div>

            <div className="setting-row">
              <div className="setting-info">
                <p className="font-label-md font-semibold text-on-surface">Temperature: {options.temperature.toFixed(1)}</p>
                <p className="font-label-sm text-on-surface-variant">0 = luôn chọn token chắc nhất; cao hơn = đa dạng hơn, dễ sai hơn.</p>
              </div>
              <input type="range" min={0} max={1.5} step={0.1} value={options.temperature}
                aria-label="Temperature"
                onChange={(e) => update({ temperature: Number(e.target.value) })} />
            </div>

            <div className="divider"></div>

            <div className="setting-row">
              <div className="setting-info">
                <p className="font-label-md font-semibold text-on-surface">Độ dài tối đa: {options.max_new_tokens} token</p>
                <p className="font-label-sm text-on-surface-variant">Model đọc tối đa 512 token, gồm cả lịch sử trò chuyện.</p>
              </div>
              <input type="range" min={20} max={400} step={20} value={options.max_new_tokens}
                aria-label="Max new tokens"
                onChange={(e) => update({ max_new_tokens: Number(e.target.value) })} />
            </div>
          </BentoCard>

          <BentoCard className="settings-card-anim col-span-12 setting-card">
            <div className="card-header-small">
              <span className="material-symbols-outlined text-primary">bookmark</span>
              <h3 className="font-label-md uppercase tracking-wider text-on-surface-variant">Trí nhớ</h3>
            </div>
            <div className="setting-row">
              <div className="setting-info">
                <p className="font-label-md font-semibold text-on-surface">Ghi nhớ thông tin về bạn</p>
                <p className="font-label-sm text-on-surface-variant">
                  Tên, tuổi, nơi sống, nghề, sở thích hoặc câu bắt đầu bằng "Ghi nhớ:". Dùng chung cho mọi cuộc chat.
                </p>
              </div>
              <ToggleSwitch checked={options.use_memory} onChange={(checked) => update({ use_memory: checked })} />
            </div>
            <div className="divider"></div>
            {memories.length === 0 ? (
              <p className="font-label-sm text-on-surface-variant">Chưa nhớ gì.</p>
            ) : (
              <ul className="memory-list">
                {memories.map((memory) => (
                  <li key={memory.id} className="memory-item font-label-md">
                    <span className="memory-kind">{MEMORY_LABELS[memory.kind] ?? memory.kind}</span>
                    <span className="memory-value">{memory.value}</span>
                    <button type="button" className="memory-delete" title="Quên" onClick={() => handleForget(memory)}>
                      <span className="material-symbols-outlined">close</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {memories.length > 0 && (
              <button type="button" className="delete-activity-btn group" onClick={handleForgetAll}>
                <span className="font-label-md text-on-surface">Xóa toàn bộ trí nhớ</span>
                <span className="material-symbols-outlined text-on-surface-variant icon">chevron_right</span>
              </button>
            )}
          </BentoCard>

          <BentoCard className="settings-card-anim col-span-6 setting-card">
            <div className="card-header-small">
              <span className="material-symbols-outlined text-primary">history</span>
              <h3 className="font-label-md uppercase tracking-wider text-on-surface-variant">Lịch sử</h3>
            </div>
            <p className="font-label-sm text-on-surface-variant">Lưu trong SQLite ở backend/data/tinymind.db.</p>
            <button type="button" className="delete-activity-btn group" onClick={handleDeleteAll}>
              <span className="font-label-md text-on-surface">Xóa toàn bộ lịch sử</span>
              <span className="material-symbols-outlined text-on-surface-variant icon">chevron_right</span>
            </button>
            {deleteNote && <p className="font-label-sm text-on-surface-variant" role="status">{deleteNote}</p>}
          </BentoCard>

          <BentoCard className="settings-card-anim col-span-6 setting-card">
            <div className="card-header-small">
              <span className="material-symbols-outlined text-primary">dns</span>
              <h3 className="font-label-md uppercase tracking-wider text-on-surface-variant">Backend</h3>
            </div>
            {healthError ? (
              <p className="font-label-md status-error" role="alert">Không kết nối được: {healthError}</p>
            ) : (
              <dl className="status-list font-label-md">
                <dt>Thiết bị</dt><dd>{health?.device ?? '…'}</dd>
                <dt>Model</dt><dd>{models.length}</dd>
                <dt>MCP</dt><dd>{mcpErrors.length === 0 ? 'OK' : `${mcpErrors.length} server lỗi`}</dd>
              </dl>
            )}
            {mcpErrors.map(([server, message]) => (
              <p key={server} className="font-label-sm status-error">{server}: {message}</p>
            ))}
          </BentoCard>
        </div>

        <footer className="settings-footer">
          <p className="font-label-sm text-on-surface-variant">{DISCLAIMER}</p>
        </footer>
      </div>
    </div>
  );
};

export default SettingsPage;
