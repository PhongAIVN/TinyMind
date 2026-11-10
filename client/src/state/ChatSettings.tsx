import React, { createContext, useContext, useEffect, useState } from 'react';
import { api, type ChatOptions, type ModelInfo } from '../api/client';

const STORAGE_KEY = 'tinymind.settings';
const DEFAULTS: ChatOptions = { model: 'sft', think: true, temperature: 0.7, max_new_tokens: 200 };

interface ChatSettingsValue {
  models: ModelInfo[];
  options: ChatOptions;
  update: (patch: Partial<ChatOptions>) => void;
  /** Bumped whenever history changes, so the history list can refetch. */
  historyVersion: number;
  bumpHistory: () => void;
}

const ChatSettingsContext = createContext<ChatSettingsValue | null>(null);

function loadOptions(): ChatOptions {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') };
  } catch {
    return DEFAULTS;
  }
}

export const ChatSettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [options, setOptions] = useState<ChatOptions>(loadOptions);
  const [historyVersion, setHistoryVersion] = useState(0);

  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  const update = (patch: Partial<ChatOptions>) => {
    setOptions((previous) => {
      const next = { ...previous, ...patch };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage unavailable (private mode): settings just won't persist
      }
      return next;
    });
  };

  return (
    <ChatSettingsContext.Provider
      value={{ models, options, update, historyVersion, bumpHistory: () => setHistoryVersion((v) => v + 1) }}
    >
      {children}
    </ChatSettingsContext.Provider>
  );
};

export function useChatSettings(): ChatSettingsValue {
  const value = useContext(ChatSettingsContext);
  if (!value) throw new Error('useChatSettings must be used inside ChatSettingsProvider');
  return value;
}
