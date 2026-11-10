import React from 'react';

interface CodeBlockProps {
  language: string;
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, code }) => {
  return (
    <div 
      style={{
        borderRadius: '16px',
        backgroundColor: 'var(--surface-container-high)',
        overflow: 'hidden',
        boxShadow: '0px 4px 20px rgba(0,0,0,0.04)',
        marginTop: '16px',
        marginBottom: '16px'
      }}
    >
      <div 
        style={{
          backgroundColor: 'var(--surface-variant)',
          padding: '8px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--outline-variant)'
        }}
      >
        <span className="font-label-md" style={{ color: 'var(--on-surface-variant)' }}>
          {language}
        </span>
        <button
          type="button"
          onClick={() => navigator.clipboard?.writeText(code).catch(() => undefined)}
          className="font-label-md"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            color: 'var(--primary)',
            padding: '4px 8px',
            borderRadius: '6px',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.5)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>content_copy</span>
          Copy
        </button>
      </div>
      <pre 
        style={{
          padding: '20px',
          overflowX: 'auto',
          fontSize: '14px',
          color: 'var(--on-surface-variant)',
          margin: 0,
          lineHeight: 1.5
        }}
      >
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
