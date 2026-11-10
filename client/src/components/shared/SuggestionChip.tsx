import React from 'react';

interface SuggestionChipProps {
  label: string;
  onClick?: () => void;
}

const SuggestionChip: React.FC<SuggestionChipProps> = ({ label, onClick }) => {
  return (
    <button 
      onClick={onClick}
      className="font-label-md"
      style={{
        padding: '8px 16px',
        backgroundColor: 'var(--surface-container-low)',
        border: '1px solid var(--outline-variant)',
        borderRadius: '9999px',
        color: 'var(--on-surface)',
        transition: 'all 0.2s',
        whiteSpace: 'nowrap'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--google-light-blue)';
        e.currentTarget.style.borderColor = 'var(--google-blue)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--surface-container-low)';
        e.currentTarget.style.borderColor = 'var(--outline-variant)';
      }}
    >
      {label}
    </button>
  );
};

export default SuggestionChip;
