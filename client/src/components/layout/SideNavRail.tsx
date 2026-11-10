import React, { useRef } from 'react';
import { NavLink } from 'react-router-dom';
import gsap from 'gsap';
import { APP_NAME } from "../../config/app_config";
import './SideNavRail.css';

const SideNavRail: React.FC = () => {
  const navRef = useRef<HTMLElement>(null);

  const handleMouseEnter = () => {
    if (navRef.current) {
      gsap.to(navRef.current, { width: 256, duration: 0.3, ease: 'power2.out' });
      gsap.to('.nav-label', { opacity: 1, duration: 0.2, delay: 0.1 });
    }
  };

  const handleMouseLeave = () => {
    if (navRef.current) {
      gsap.to(navRef.current, { width: 64, duration: 0.3, ease: 'power2.inOut' });
      gsap.to('.nav-label', { opacity: 0, duration: 0.1 });
    }
  };

  return (
    <aside 
      ref={navRef}
      className="side-nav-rail"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="nav-top">
        <div className="nav-brand">
          <span className="material-symbols-outlined logo-icon">auto_awesome</span>
          <span className="font-headline-md nav-label brand-name">{APP_NAME}</span>
        </div>

        <nav className="nav-links">
          <NavLink to="/" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`} end>
            <span className="material-symbols-outlined icon">add</span>
            <span className="font-label-md nav-label">Trò chuyện mới</span>
          </NavLink>
          
          <NavLink to="/history" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="material-symbols-outlined icon">history</span>
            <span className="font-label-md nav-label">Lịch sử</span>
          </NavLink>

          <NavLink to="/explore" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="material-symbols-outlined icon">explore</span>
            <span className="font-label-md nav-label">Khám phá</span>
          </NavLink>
        </nav>
      </div>

      <div className="nav-bottom">
        <NavLink to="/settings" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="material-symbols-outlined icon">settings</span>
          <span className="font-label-md nav-label">Cài đặt</span>
        </NavLink>
      </div>
    </aside>
  );
};

export default SideNavRail;
