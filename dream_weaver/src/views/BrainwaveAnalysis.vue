<script setup>
import { ref } from 'vue'

// 模拟波形数据动画状态
const activeBarIndex = ref(2)
setInterval(() => {
  activeBarIndex.value = Math.floor(Math.random() * 5)
}, 800)
</script>

<template>
  <div class="brainwave-container custom-scroll">
    <div class="content-wrapper">
      <header class="main-header">
        <h1 class="glow-title">脑电波深度解析</h1>
        <p class="subtitle">BRAINWAVE PATTERN ANALYSIS</p>
      </header>

      <!-- Wave Cards Grid -->
      <div class="wave-grid">
        <!-- Alpha -->
        <div class="glass-card wave-card">
          <div class="card-header">
            <h3 class="card-title">ALPHA波</h3>
            <span class="badge badge-relax">放松</span>
          </div>
          <div class="card-value-row">
            <span class="value-xl">18.4</span>
            <span class="unit">µV²</span>
          </div>
          <div class="wave-visual">
            <div class="bar h-40"></div>
            <div class="bar h-60"></div>
            <div class="bar h-80" :class="{ 'active': activeBarIndex === 2 }"></div>
            <div class="bar h-50"></div>
            <div class="bar h-75"></div>
          </div>
        </div>

        <!-- Beta -->
        <div class="glass-card wave-card">
          <div class="card-header">
            <h3 class="card-title">BETA波</h3>
            <span class="badge badge-active">活跃</span>
          </div>
          <div class="card-value-row">
            <span class="value-xl">12.1</span>
            <span class="unit">µV²</span>
          </div>
          <div class="wave-visual">
            <div class="bar h-75"></div>
            <div class="bar h-40"></div>
            <div class="bar h-100" :class="{ 'active': activeBarIndex === 3 }"></div>
            <div class="bar h-60"></div>
            <div class="bar h-50"></div>
          </div>
        </div>

        <!-- Theta -->
        <div class="glass-card wave-card">
          <div class="card-header">
            <h3 class="card-title">THETA波</h3>
            <span class="badge badge-deep">深度</span>
          </div>
          <div class="card-value-row">
            <span class="value-xl">32.8</span>
            <span class="unit">µV²</span>
          </div>
          <div class="wave-visual">
            <div class="bar h-50"></div>
            <div class="bar h-75" :class="{ 'active': activeBarIndex === 1 }"></div>
            <div class="bar h-60"></div>
            <div class="bar h-90"></div>
            <div class="bar h-40"></div>
          </div>
        </div>

        <!-- Delta -->
        <div class="glass-card wave-card">
          <div class="card-header">
            <h3 class="card-title">DELTA波</h3>
            <span class="badge badge-sleep">睡眠</span>
          </div>
          <div class="card-value-row">
            <span class="value-xl">45.2</span>
            <span class="unit">µV²</span>
          </div>
          <div class="wave-visual">
            <div class="bar h-90" :class="{ 'active': activeBarIndex === 0 }"></div>
            <div class="bar h-70"></div>
            <div class="bar h-60"></div>
            <div class="bar h-80"></div>
            <div class="bar h-50"></div>
          </div>
        </div>
      </div>

      <!-- Analysis Charts Panel -->
      <div class="glass-panel main-analysis-panel">
        <div class="analysis-layout">
          
          <!-- Radar Chart Section -->
          <div class="chart-section radar-section">
            <div class="section-header">
              <h2>频率分布图</h2>
              <p class="sub-text">实时波段主导分布</p>
            </div>
            <div class="radar-container">
              <svg class="radar-svg" viewBox="0 0 100 100">
                <!-- Grid Polygons -->
                <polygon class="radar-grid" points="50,5 95,38 77,90 23,90 5,38"></polygon>
                <polygon class="radar-grid inner" points="50,20 80,42 68,76 32,76 20,42"></polygon>
                <polygon class="radar-grid inner-2" points="50,35 65,46 59,63 41,63 35,46"></polygon>
                
                <!-- Data Polygon with Glow -->
                <polygon class="radar-data glow-path" points="50,15 85,45 70,80 30,70 15,40"></polygon>
                
                <!-- Axis Lines -->
                <line x1="50" y1="50" x2="50" y2="5" class="axis-line" />
                <line x1="50" y1="50" x2="95" y2="38" class="axis-line" />
                <line x1="50" y1="50" x2="77" y2="90" class="axis-line" />
                <line x1="50" y1="50" x2="23" y2="90" class="axis-line" />
                <line x1="50" y1="50" x2="5" y2="38" class="axis-line" />

                <!-- Labels -->
                <text class="radar-label" x="50" y="-2" text-anchor="middle">ALPHA</text>
                <text class="radar-label" x="100" y="38" text-anchor="start">BETA</text>
                <text class="radar-label" x="82" y="98" text-anchor="start">THETA</text>
                <text class="radar-label" x="18" y="98" text-anchor="end">DELTA</text>
                <text class="radar-label" x="0" y="38" text-anchor="end">GAMMA</text>
              </svg>
            </div>
          </div>

          <!-- Timeline Chart Section -->
          <div class="chart-section timeline-section">
            <div class="timeline-header">
              <div>
                <h2>睡眠阶段时间轴</h2>
                <p class="sub-text">神经活动与睡眠图关联</p>
              </div>
              <div class="status-badge">
                <span class="status-dot pulse-anim"></span>
                <span class="highlight-text glow-text">快速眼动期 (REM)</span>
              </div>
            </div>
            
            <div class="timeline-chart">
              <!-- Y-Axis Labels overlay -->
              <div class="timeline-labels-y">
                <span>清醒</span>
                <span>REM</span>
                <span>浅睡</span>
                <span>深睡</span>
              </div>
              
              <!-- Grid Background -->
              <div class="timeline-grid">
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
              </div>
              
              <!-- SVG Curve -->
              <svg class="timeline-svg" preserveAspectRatio="none" viewBox="0 0 500 200">
                <defs>
                   <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                     <stop offset="0%" stop-color="rgba(255,255,255,0.4)" />
                     <stop offset="100%" stop-color="#34d399" />
                   </linearGradient>
                </defs>
                <path class="line-path glow-path" d="M0,20 L40,80 L80,140 L120,180 L160,20 L200,140 L240,180 L280,80 L320,20 L360,140 L400,80 L500,20" stroke="url(#lineGradient)"></path>
                
                <!-- Current Point Marker -->
                <circle cx="320" cy="20" r="4" class="current-point"></circle>
                <circle cx="320" cy="20" r="8" class="current-point-ring"></circle>
              </svg>

              <div class="floating-tooltip" style="left: 64%; top: 10%;">
                当前: REM
              </div>
            </div>
            
            <div class="timeline-labels-x">
              <span>23:00</span>
              <span>01:00</span>
              <span>03:00</span>
              <span>05:00</span>
              <span>07:00</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom Stats Grid -->
      <div class="stats-grid">
        <div class="glass-card stat-card">
          <div class="stat-left">
            <div class="icon-box">
              <span class="material-symbols-outlined">timer</span>
            </div>
            <div>
              <h3 class="stat-title">REM 频率</h3>
              <p class="sub-text">快速眼动期的发生频率</p>
            </div>
          </div>
          <div class="stat-right">
            <span class="stat-value glow-text">4.2</span>
            <span class="stat-unit">周期 / 夜</span>
          </div>
        </div>
        
        <div class="glass-card stat-card">
          <div class="stat-left">
            <div class="icon-box">
              <span class="material-symbols-outlined">verified_user</span>
            </div>
            <div>
              <h3 class="stat-title">深睡稳定性</h3>
              <p class="sub-text">DELTA波形的持续一致性</p>
            </div>
          </div>
          <div class="stat-right">
            <span class="stat-value glow-text text-green">92%</span>
            <span class="stat-unit highlight-green">稳定</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset & Base */
* {
  box-sizing: border-box;
}

.brainwave-container {
  width: 100%;
  height: 100%;
  overflow-y: auto;
  padding: 40px 20px 80px;
  color: white;
  font-family: 'Inter', sans-serif;
}

.content-wrapper {
  max-width: 1100px;
  margin: 0 auto;
}

/* Header */
.main-header {
  text-align: center;
  margin-bottom: 50px;
}

.glow-title {
  font-size: 2.5rem;
  font-weight: 300;
  letter-spacing: 0.2em;
  text-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
  margin: 0;
}

.subtitle {
  font-size: 0.75rem;
  color: rgba(156, 163, 175, 1); /* tailwind gray-400 equivalent */
  letter-spacing: 0.4em;
  margin-top: 15px;
  text-transform: uppercase;
}

/* Common Glass Styles */
.glass-card {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  border-radius: 24px;
  padding: 24px;
  transition: all 0.3s;
}

.glass-card:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}

.glass-panel {
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
  border-radius: 40px;
  padding: 40px;
}

/* Wave Cards Grid */
.wave-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 30px;
}

@media (max-width: 900px) {
  .wave-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 600px) {
  .wave-grid {
    grid-template-columns: 1fr;
  }
}

.wave-card {
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.card-title {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: #d1d5db;
  margin: 0;
}

.badge {
  font-size: 0.65rem;
  padding: 2px 8px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-weight: 500;
}
.badge-relax { color: #a7f3d0; background: rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.3); }
.badge-active { color: #fde68a; background: rgba(245, 158, 11, 0.2); border-color: rgba(245, 158, 11, 0.3); }
.badge-deep { color: #bfdbfe; background: rgba(59, 130, 246, 0.2); border-color: rgba(59, 130, 246, 0.3); }
.badge-sleep { color: #c7d2fe; background: rgba(99, 102, 241, 0.2); border-color: rgba(99, 102, 241, 0.3); }

.card-value-row {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin-bottom: 20px;
}

.value-xl {
  font-size: 2.25rem;
  font-weight: 700;
  line-height: 1;
}

.unit {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

/* Wave Visual Bars */
.wave-visual {
  height: 48px;
  display: flex;
  align-items: flex-end;
  gap: 6px;
  padding-bottom: 4px;
}

.bar {
  flex: 1;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  transition: height 0.3s ease, background-color 0.3s ease;
}

.bar.active {
  background: rgba(255, 255, 255, 0.6);
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
}

.h-40 { height: 40%; }
.h-50 { height: 50%; }
.h-60 { height: 60%; }
.h-70 { height: 70%; }
.h-75 { height: 75%; }
.h-80 { height: 80%; }
.h-90 { height: 90%; }
.h-100 { height: 100%; }

/* Analysis Layout */
.analysis-layout {
  display: flex;
  gap: 60px;
}

@media (max-width: 900px) {
  .analysis-layout {
    flex-direction: column;
    gap: 50px;
  }
}

.chart-section {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.section-header, .timeline-header {
  margin-bottom: 30px;
}

.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 10px;
}

.section-header h2, .timeline-header h2 {
  font-size: 1.25rem;
  font-weight: 500;
  margin: 0;
}

.sub-text {
  font-size: 0.75rem;
  color: #9ca3af;
  margin-top: 6px;
  margin-bottom: 0;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.1);
  padding: 6px 12px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.status-dot {
  width: 8px;
  height: 8px;
  background-color: #34d399;
  border-radius: 50%;
}

.highlight-text {
  font-size: 0.85rem;
  font-weight: 600;
  color: white;
}

.glow-text {
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
}

.text-green { color: #34d399; }

/* Radar Chart */
.radar-container {
  width: 100%;
  max-width: 340px;
  aspect-ratio: 1/1;
  margin: 0 auto;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
}

.radar-svg {
  width: 100%;
  height: 100%;
  overflow: visible;
}

.radar-grid {
  fill: none;
  stroke: rgba(255, 255, 255, 0.2);
  stroke-width: 1;
}

.radar-grid.inner { stroke-opacity: 0.6; stroke-dasharray: 2; }
.radar-grid.inner-2 { stroke-opacity: 0.3; stroke-dasharray: 2; }

.axis-line {
  stroke: rgba(255, 255, 255, 0.1);
  stroke-width: 1;
}

.radar-data {
  fill: rgba(255, 255, 255, 0.15);
  stroke: white;
  stroke-width: 2.5;
  stroke-linejoin: round;
}

.glow-path {
  filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.6));
}

.radar-label {
  fill: #d1d5db;
  font-size: 5px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

/* Timeline Chart */
.timeline-chart {
  position: relative;
  flex: 1;
  min-height: 220px;
  margin-top: 10px;
  border-left: 1px solid rgba(255, 255, 255, 0.2);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.timeline-labels-y {
  position: absolute;
  left: -40px; /* shift outside left */
  top: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 10px 0;
  text-align: right;
  width: 30px;
}

.timeline-labels-y span {
  font-size: 0.65rem;
  color: #9ca3af;
}

.timeline-grid {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.grid-line {
  width: 100%;
  height: 25%;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
}

.timeline-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.line-path {
  fill: none;
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.current-point {
  fill: white;
}

.current-point-ring {
  fill: none;
  stroke: rgba(255, 255, 255, 0.5);
  stroke-width: 1;
  animation: pulse 2s infinite;
}

.floating-tooltip {
  position: absolute;
  background: white;
  color: black;
  font-size: 0.65rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 8px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  transform: translate(-50%, -120%);
}

.floating-tooltip::after {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 4px 4px 0;
  border-style: solid;
  border-color: white transparent transparent;
}

.timeline-labels-x {
  display: flex;
  justify-content: space-between;
  margin-top: 15px;
  font-size: 0.65rem;
  color: #9ca3af;
  padding-left: 0; /* Align with chart start */
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-top: 30px;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}

.stat-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 30px;
}

.stat-left {
  display: flex;
  align-items: center;
  gap: 24px;
}

.icon-box {
  width: 64px;
  height: 64px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
}

.icon-box span {
  font-size: 32px;
  color: white;
}

.stat-title {
  font-size: 1.1rem;
  font-weight: 500;
  margin: 0;
  color: white;
}

.stat-right {
  text-align: right;
}

.stat-value {
  font-size: 2.5rem;
  font-weight: 700;
  display: block;
  line-height: 1.1;
}

.stat-unit {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  display: block;
  margin-top: 6px;
}

.highlight-green {
  color: #34d399;
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.5); opacity: 0; }
  100% { transform: scale(1); opacity: 0; }
}

.pulse-anim {
  animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Scrollbar */
.custom-scroll::-webkit-scrollbar {
  display: none;
}
.custom-scroll {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>