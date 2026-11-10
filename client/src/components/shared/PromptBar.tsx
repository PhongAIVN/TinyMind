import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { APP_NAME, DISCLAIMER } from '../../config/app_config';
import { useChatSettings } from '../../state/ChatSettings';
import './PromptBar.css';

interface PromptBarProps {
  variant?: 'premium' | 'default';
  /** Called with the trimmed text. Without it, the bar opens a new chat with that text. */
  onSubmit?: (text: string) => void;
  isBusy?: boolean;
  onStop?: () => void;
}

const PromptBar: React.FC<PromptBarProps> = ({ variant = 'default', onSubmit, isBusy = false, onStop }) => {
  const [isFocused, setIsFocused] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const navigate = useNavigate();
  const { models, options, update } = useChatSettings();

  const isPremium = variant === 'premium';
  const currentModel = models.find((m) => m.id === options.model);
  const canSend = inputValue.trim().length > 0 && !isBusy;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const text = inputValue.trim();
    if (!text || isBusy) return;
    setInputValue('');
    if (onSubmit) onSubmit(text);
    else navigate('/chat', { state: { initialMessage: text } });
  };

  return (
    <div className={`prompt-bar-wrapper ${isFocused ? 'focused' : ''} ${isPremium ? 'premium-variant' : 'default-variant'}`}>
      <div className="prompt-bar-container">

        {/* Render premium layers only if variant is premium */}
        {isPremium && (
          <>
            {/* Layer 2: Blurred rainbow glow following the border */}
            <div className="prompt-bar-glow-container">
              <div className="prompt-bar-glow"></div>
            </div>

            {/* Layer 1: Animated conic rainbow border */}
            <div className="prompt-bar-border-container">
              <div className="prompt-bar-border"></div>
            </div>
          </>
        )}

        {/* Content (Layer 3 & 4 styling depends on variant class) */}
        <form className={`prompt-bar-inner ${!isPremium ? 'default-style' : ''}`} onSubmit={handleSubmit}>
          {currentModel?.think && (
            <button
              type="button"
              className={`prompt-btn think-btn ${options.think ? 'active-text' : ''}`}
              title={options.think ? 'Tư duy: bật' : 'Tư duy: tắt'}
              aria-pressed={options.think}
              onClick={() => update({ think: !options.think })}
            >
              <span className="material-symbols-outlined">psychology</span>
            </button>
          )}

          <input
            type="text"
            className="prompt-input"
            placeholder={`Hỏi ${APP_NAME}`}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
          />

          <div className="prompt-actions">
            <label className="model-selector" title={currentModel?.description}>
              <select
                className="text model-select"
                value={options.model}
                onChange={(e) => update({ model: e.target.value })}
                aria-label="Model"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
              <span className="material-symbols-outlined icon">keyboard_arrow_down</span>
            </label>

            {isBusy && onStop ? (
              <button type="button" className="prompt-btn send-btn active-text" onClick={onStop} title="Dừng">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>stop_circle</span>
              </button>
            ) : (
              <button type="submit" className={`prompt-btn send-btn ${canSend ? 'active-text' : ''}`} disabled={!canSend}>
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>send</span>
              </button>
            )}
          </div>
        </form>
      </div>

      <div className="prompt-footer-container">
        <p className="prompt-footer font-label-sm">{DISCLAIMER}</p>
      </div>
    </div>
  );
};

export default PromptBar;
