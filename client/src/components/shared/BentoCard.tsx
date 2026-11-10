import React, { useRef } from 'react';
import gsap from 'gsap';

interface BentoCardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hoverEffect?: boolean;
}

const BentoCard: React.FC<BentoCardProps> = ({ 
  children, 
  className = '', 
  onClick,
  hoverEffect = true
}) => {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = () => {
    if (hoverEffect && cardRef.current) {
      gsap.to(cardRef.current, { 
        y: -4, 
        scale: 1.01,
        boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.08)',
        duration: 0.3,
        ease: 'power2.out'
      });
    }
  };

  const handleMouseLeave = () => {
    if (hoverEffect && cardRef.current) {
      gsap.to(cardRef.current, { 
        y: 0, 
        scale: 1,
        boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.04)',
        duration: 0.3,
        ease: 'power2.out'
      });
    }
  };

  return (
    <div 
      ref={cardRef}
      className={`bento-card ${className}`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        backgroundColor: 'var(--surface-container-lowest)',
        borderRadius: '24px',
        boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.04)',
        transition: 'border-color 0.3s',
        cursor: onClick ? 'pointer' : 'default',
        overflow: 'hidden',
        position: 'relative'
      }}
    >
      {children}
    </div>
  );
};

export default BentoCard;
