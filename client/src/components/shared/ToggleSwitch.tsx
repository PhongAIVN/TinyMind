import React from 'react';

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange }) => {
  return (
    <div 
      onClick={() => onChange(!checked)}
      style={{
        width: '44px',
        height: '24px',
        borderRadius: '9999px',
        backgroundColor: checked ? 'var(--primary)' : 'var(--surface-dim)',
        position: 'relative',
        cursor: 'pointer',
        transition: 'background-color 0.2s',
        flexShrink: 0
      }}
    >
      <div 
        style={{
          width: '20px',
          height: '20px',
          backgroundColor: '#fff',
          borderRadius: '50%',
          position: 'absolute',
          top: '2px',
          left: checked ? '22px' : '2px',
          transition: 'left 0.2s ease-in-out',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}
      />
    </div>
  );
};

export default ToggleSwitch;
