import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, type ToolInfo } from '../api/client';
import BentoCard from '../components/shared/BentoCard';
import { APP_NAME, DISCLAIMER } from '../config/app_config';
import { useChatSettings } from '../state/ChatSettings';
import './ExplorePage.css';

/** Prompts each model actually handles well (see the toy-math results in the README). */
const SAMPLES: { model: string; prompt: string }[] = [
  { model: 'sft', prompt: 'Thủ đô của Việt Nam là gì?' },
  { model: 'sft', prompt: 'Viết hàm Python tính giai thừa.' },
  { model: 'tool', prompt: 'Tính giúp mình 72 cộng 97 bằng bao nhiêu?' },
  { model: 'cispo', prompt: 'Tính 12 nhân 7 bằng bao nhiêu?' },
  { model: 'pretrain', prompt: 'Hà Nội là' },
];

const ExplorePage: React.FC = () => {
  const navigate = useNavigate();
  const { models, options, update } = useChatSettings();
  const [tools, setTools] = useState<ToolInfo[]>([]);

  useEffect(() => {
    api.tools().then(setTools).catch(() => setTools([]));
  }, []);

  const tryPrompt = (model: string, prompt: string) => {
    update({ model });
    navigate('/chat', { state: { initialMessage: prompt } });
  };

  const labelOf = (id: string) => models.find((m) => m.id === id)?.label ?? id;

  return (
    <div className="explore-page">
      <section className="hero-section">
        <h1 className="font-display-lg text-on-surface">Khám phá {APP_NAME}</h1>
        <p className="font-body-md text-secondary hero-subtitle">
          Mỗi phiên bản là một chặng train: pretrain, SFT, LoRA, tool use, tư duy, chưng cất, RL. Chọn một bản để trò chuyện.
        </p>
      </section>

      <h2 className="font-headline-md section-title">Phiên bản</h2>
      <div className="model-grid">
        {models.map((model) => (
          <BentoCard key={model.id} className={`model-card ${model.id === options.model ? 'selected' : ''}`}
            onClick={() => update({ model: model.id })}>
            <div className="model-card-head">
              <span className="font-label-md model-name">{model.label}</span>
              {model.id === options.model && <span className="material-symbols-outlined check">check_circle</span>}
            </div>
            <p className="font-label-sm text-on-surface-variant">{model.description}</p>
            <div className="model-tags font-label-sm">
              {model.chat ? <span>chat</span> : <span>viết tiếp</span>}
              {model.think && <span>tư duy</span>}
              {model.tools && <span>tool</span>}
            </div>
          </BentoCard>
        ))}
      </div>

      <h2 className="font-headline-md section-title">Thử nhanh</h2>
      <div className="sample-list">
        {SAMPLES.filter((s) => models.some((m) => m.id === s.model)).map((sample) => (
          <button key={sample.prompt} type="button" className="sample-item" onClick={() => tryPrompt(sample.model, sample.prompt)}>
            <span className="font-label-sm text-google-blue">{labelOf(sample.model)}</span>
            <span className="font-body-md text-on-surface">{sample.prompt}</span>
          </button>
        ))}
      </div>

      <h2 className="font-headline-md section-title">Tool (MCP)</h2>
      {tools.length === 0 ? (
        <p className="font-body-md text-on-surface-variant">Chưa có tool nào: backend chưa chạy hoặc MCP server lỗi.</p>
      ) : (
        <ul className="tool-list">
          {tools.map((tool) => (
            <li key={tool.name} className="tool-item">
              <code className="tool-name">{tool.name}</code>
              <span className="font-body-md text-on-surface-variant">{tool.description}</span>
              <span className="font-label-sm tool-server">{tool.server}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="font-label-sm text-on-surface-variant tool-note">
        Chỉ bản Tools được train để tự gọi tool, và chỉ gọi máy tính.
      </p>

      <footer className="explore-footer">
        <p className="disclaimer">{DISCLAIMER}</p>
      </footer>
    </div>
  );
};

export default ExplorePage;
