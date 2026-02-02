<script setup>
import { ref } from 'vue'

const activeRange = ref('1h')
const timeRanges = ['1h', '4h', '8h']
const showActivityModal = ref(false)

const statCards = [
  { 
    title: "心率变异性", 
    value: "65", 
    unit: "ms", 
    trend: "up", 
    trendText: "较昨晚 +5%", 
    dotColor: "#f87171", // tailwind red-400
    icon: "favorite", 
    isPulse: true,
    trendClass: "trend-up"
  },
  { 
    title: "呼吸频率", 
    value: "14", 
    unit: "次/分", 
    trend: null,
    trendText: "节律平稳", 
    dotColor: "#38bdf8", // tailwind sky-400
    icon: "air", 
    trendColor: "#9ca3af",
    trendClass: "trend-neutral"
  },
  { 
    title: "血氧饱和度", 
    value: "98%", 
    unit: "", 
    trend: null,
    trendText: "理想水平", 
    dotColor: "#34d399", // tailwind emerald-400
    icon: "water_drop",
    trendClass: "trend-good"
  }
]

const smallStats = [
  { icon: "bedtime", label: "睡眠质量", value: "84%", subValue: "+2%", subClass: "text-green" },
  { icon: "auto_fix_high", label: "梦境生动度", value: "高" },
  { icon: "sync", label: "分析状态", value: "已同步", valueClass: "text-green" }
]

// Mock data for activity details
const activityEvents = [
  { time: '01:15', type: '翻身', duration: '12s', intensity: 'low' },
  { time: '02:40', type: '微动', duration: '5s', intensity: 'low' },
  { time: '04:20', type: '肢体活动', duration: '45s', intensity: 'high' },
  { time: '06:10', type: '翻身', duration: '18s', intensity: 'medium' }
]
</script>

<template>
  <div class="physio-container custom-scroll relative">
    <div class="content-wrapper">
      
      <!-- Header -->
      <header class="main-header">
        <h1 class="glow-title">生理数据监控</h1>
        <p class="subtitle">PHYSIOLOGICAL DATA MONITORING</p>
      </header>

      <!-- Dashboard Panel -->
      <div class="glass-panel">
        
        <!-- Top Row: Primary Vital Signs -->
        <div class="grid-row grid-3">
          <div v-for="(card, index) in statCards" :key="index" class="glass-card stat-card">
            <div class="card-left">
              <div class="card-label">
                <span class="status-dot" :class="{ 'pulse-anim': card.isPulse }" :style="{ backgroundColor: card.dotColor }"></span>
                <h3>{{ card.title }}</h3>
              </div>
              <div class="card-value-group">
                <span class="value">{{ card.value }}</span>
                <span v-if="card.unit" class="unit">{{ card.unit }}</span>
              </div>
              <div class="card-trend" :class="card.trendClass">
                <span v-if="card.trend === 'up'" class="material-symbols-outlined icon-xs">trending_up</span>
                <span>{{ card.trendText }}</span>
              </div>
            </div>
            <div class="card-right icon-wrapper" :class="{ 'pulse-shadow': card.isPulse }">
              <span class="material-symbols-outlined icon-md">{{ card.icon }}</span>
            </div>
          </div>
        </div>

        <!-- Middle Row: Chart -->
        <div class="glass-card chart-card">
          <div class="chart-header">
            <div>
              <h2>睡眠周期活动</h2>
              <p class="sub-text">与梦境强度阶段相关</p>
            </div>
            
            <div class="time-selector">
              <button
                v-for="range in timeRanges"
                :key="range"
                @click="activeRange = range"
                class="time-btn"
                :class="{ active: activeRange === range }"
              >
                {{ range.replace('h', '小时') }}
              </button>
            </div>
          </div>

          <div class="chart-body">
            <!-- Y-Axis -->
            <div class="y-axis">
              <span>100</span><span>75</span><span>50</span><span>25</span><span>0</span>
            </div>

            <div class="chart-area">
              <!-- Grid Lines -->
              <div class="grid-lines">
                <div class="line dashed"></div><div class="line dashed"></div>
                <div class="line dashed"></div><div class="line dashed"></div>
                <div class="line solid"></div>
              </div>

              <!-- SVG Chart -->
              <svg class="chart-svg" preserveAspectRatio="none" viewBox="0 0 850 280">
                <defs>
                  <linearGradient id="chartGradient" x1="0%" x2="0%" y1="0%" y2="100%">
                    <stop offset="0%" stop-color="rgba(255, 255, 255, 0.2)"></stop>
                    <stop offset="100%" stop-color="rgba(255, 255, 255, 0)"></stop>
                  </linearGradient>
                </defs>
                <path d="M0,200 C100,180 150,80 250,120 C350,160 450,220 550,140 C650,60 750,100 850,180 L850,280 L0,280 Z" fill="url(#chartGradient)"></path>
                <path class="path-glow" d="M0,200 C100,180 150,80 250,120 C350,160 450,220 550,140 C650,60 750,100 850,180" fill="none" stroke="white" stroke-linecap="round" stroke-width="3"></path>
                <circle cx="250" cy="120" fill="white" r="5"></circle>
              </svg>

              <!-- Tooltip -->
              <div class="chart-tooltip">
                <div class="tooltip-box">
                  REM 阶段 2<br/><span class="tooltip-time">凌晨 02:45</span>
                </div>
              </div>
            </div>

            <!-- X-Axis -->
            <div class="x-axis">
              <span>23:00</span><span>01:00</span><span>03:00</span><span>05:00</span><span>07:00</span>
            </div>
          </div>
        </div>

        <!-- Third Row: Brainwave & Activity -->
        <div class="grid-row grid-2">
          <!-- Brainwave Card -->
          <div class="glass-card stat-row-card">
            <div class="icon-square">
              <span class="material-symbols-outlined">waves</span>
            </div>
            <div class="flex-grow">
              <h3>脑电波模式</h3>
              <p class="sub-text">正在分析 Theta 波</p>
            </div>
            <div class="text-right">
              <span class="value-lg glow">Theta</span>
              <span class="unit-block">4-8 Hz</span>
            </div>
          </div>

          <!-- Activity Card -->
          <div class="glass-card stat-row-card">
            <div class="icon-square activity-icon-bg">
              <span class="material-symbols-outlined">notifications_active</span>
            </div>
            <div class="flex-grow">
              <h3>检测到身体活动</h3>
              <p class="sub-text">凌晨 04:20 有轻微活动</p>
            </div>
            <button class="action-btn" @click="showActivityModal = true">查看详情</button>
          </div>
        </div>

        <!-- Bottom Row: Small Stats -->
        <div class="grid-row grid-3">
          <div v-for="(stat, index) in smallStats" :key="index" class="glass-card small-stat-card">
            <div class="icon-circle">
              <span class="material-symbols-outlined icon-sm">{{ stat.icon }}</span>
            </div>
            <div>
              <p class="label-xs">{{ stat.label }}</p>
              <div class="value-row">
                <span class="value-md" :class="stat.valueClass">{{ stat.value }}</span>
                <span v-if="stat.subValue" class="sub-value" :class="stat.subClass">{{ stat.subValue }}</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- === ACTIVITY DETAIL MODAL === -->
    <Transition name="fade">
      <div v-if="showActivityModal" class="modal-overlay" @click.self="showActivityModal = false">
        <div class="glass-modal activity-modal">
          
          <div class="modal-header">
            <div>
              <h2 class="modal-title">体动监测详情</h2>
              <p class="modal-subtitle">NOCTURNAL MOVEMENT ANALYSIS</p>
            </div>
            <button class="close-btn" @click="showActivityModal = false">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>

          <div class="modal-body custom-scroll">
            
            <!-- Actigraphy Chart (Visual Representation) -->
            <div class="actigraphy-section">
              <div class="section-title">
                <span class="material-symbols-outlined">bar_chart_4_bars</span>
                Actigraphy / 动作描记图
              </div>
              <div class="actigraphy-chart">
                <div class="acti-bars">
                  <!-- Generate random-ish bars for visualization -->
                  <div class="acti-bar" v-for="i in 40" :key="i" 
                       :style="{ 
                         height: (i === 28 ? '80%' : i === 8 ? '40%' : i === 15 ? '30%' : Math.random() * 15 + 2) + '%',
                         opacity: (i === 28 ? 1 : 0.4),
                         background: (i === 28 ? '#fbbf24' : 'white')
                       }">
                  </div>
                </div>
                <div class="acti-labels">
                  <span>23:00</span>
                  <span>02:00</span>
                  <span>05:00</span>
                  <span>08:00</span>
                </div>
              </div>
            </div>

            <!-- Stats Grid -->
            <div class="activity-stats-grid">
              <div class="stat-box">
                <span class="label">Total Events</span>
                <span class="val">4</span>
              </div>
              <div class="stat-box">
                <span class="label">Peak Intensity</span>
                <span class="val text-orange">Moderate</span>
              </div>
              <div class="stat-box">
                <span class="label">Total Duration</span>
                <span class="val">1m 20s</span>
              </div>
            </div>

            <!-- Event List -->
            <div class="event-list-section">
              <div class="section-title">
                <span class="material-symbols-outlined">history</span>
                Event Log / 事件日志
              </div>
              <div class="event-list">
                <div v-for="(event, idx) in activityEvents" :key="idx" class="event-item">
                  <div class="event-time">{{ event.time }}</div>
                  <div class="event-line">
                    <div class="dot" :class="event.intensity === 'high' ? 'bg-orange' : 'bg-white'"></div>
                    <div class="line"></div>
                  </div>
                  <div class="event-details">
                    <div class="event-type">{{ event.type }}</div>
                    <div class="event-meta">持续 {{ event.duration }} • 强度: {{ event.intensity }}</div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Insight -->
            <div class="ai-insight">
              <span class="material-symbols-outlined icon">auto_awesome</span>
              <p>
                <span class="highlight">AI 分析：</span>
                凌晨 04:20 的肢体活动与快速眼动期（REM）的结束相吻合。这种活动通常是正常的体位调整，有助于防止血液循环受阻，未检测到会对深度睡眠造成中断的异常躁动。
              </p>
            </div>

          </div>
        </div>
      </div>
    </Transition>

  </div>
</template>

<script setup>
import { ref } from 'vue';

const activeTime = ref('1小时');
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Container Layout */
.physio-container {
  width: 100%;
  height: 100%;
  overflow-y: auto;
  padding: 40px 20px 80px;
  color: white;
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
}

.content-wrapper {
  max-width: 1100px;
  margin: 0 auto;
}

/* Header */
.main-header {
  text-align: center;
  margin-bottom: 40px;
}

.glow-title {
  font-size: 2.25rem;
  font-weight: 300;
  letter-spacing: 0.2em;
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.6);
  margin: 0;
}

.subtitle {
  font-size: 0.75rem;
  color: rgba(156, 163, 175, 1);
  letter-spacing: 0.3em;
  margin-top: 10px;
  text-transform: uppercase;
}

/* Glass Panels */
.glass-panel {
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  border-radius: 40px;
  padding: 30px;
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.glass-card {
  background: rgba(255, 255, 255, 0.18);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  border-radius: 16px;
  padding: 24px;
}

/* Grids */
.grid-row {
  display: grid;
  gap: 20px;
}

.grid-3 {
  grid-template-columns: 1fr;
}

.grid-2 {
  grid-template-columns: 1fr;
}

@media (min-width: 768px) {
  .grid-3 { grid-template-columns: repeat(3, 1fr); }
  .grid-2 { grid-template-columns: repeat(2, 1fr); }
}

/* Stat Cards */
.stat-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-left {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 100%;
}

.card-label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.card-label h3 {
  font-size: 0.75rem;
  font-weight: 600;
  color: #d1d5db;
  letter-spacing: 0.05em;
  margin: 0;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.pulse-anim {
  animation: pulse 2s infinite;
}

.card-value-group {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.value {
  font-size: 2.25rem;
  font-weight: 700;
  line-height: 1;
}

.unit {
  font-size: 0.75rem;
  color: #9ca3af;
}

.card-trend {
  font-size: 0.65rem;
  margin-top: 8px;
  display: flex;
  align-items: center;
  text-transform: uppercase;
  font-weight: 500;
}

.trend-up { color: #34d399; }
.trend-neutral { color: #9ca3af; }
.trend-good { color: #34d399; }

.icon-wrapper {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.pulse-shadow {
  box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
}

/* Chart Card */
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.chart-header h2 {
  font-size: 1.1rem;
  font-weight: 500;
  margin: 0;
}

.sub-text {
  font-size: 0.7rem;
  color: #9ca3af;
  margin-top: 4px;
  margin-bottom: 0;
}

.time-selector {
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 999px;
  padding: 4px;
  display: flex;
}

.time-btn {
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.7rem;
  color: white;
  background: transparent;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}

.time-btn:hover {
  background: rgba(255, 255, 255, 0.1);
}

.time-btn.active {
  background: white;
  color: black;
  font-weight: 600;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.chart-body {
  position: relative;
  height: 300px;
}

.y-axis {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  font-size: 0.6rem;
  color: #6b7280;
  padding-bottom: 20px; /* space for x-axis */
}

.chart-area {
  margin-left: 30px;
  height: calc(100% - 24px);
  position: relative;
}

.grid-lines {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.line {
  width: 100%;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.line.dashed { border-style: dashed; }
.line.solid { border-color: rgba(255, 255, 255, 0.2); }

.chart-svg {
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
}

.path-glow {
  filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.8));
}

.chart-tooltip {
  position: absolute;
  left: 26%;
  top: 25%;
  transform: translate(-50%, -50%);
}

.tooltip-box {
  background: rgba(255, 255, 255, 0.95);
  color: black;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.6rem;
  font-weight: 700;
  border: 1px solid white;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.tooltip-time {
  font-weight: 400;
  opacity: 0.6;
  font-size: 0.55rem;
}

.x-axis {
  margin-left: 30px;
  margin-top: 8px;
  display: flex;
  justify-content: space-between;
  font-size: 0.6rem;
  color: #6b7280;
}

/* Stat Row Cards (Brainwave/Activity) */
.stat-row-card {
  display: flex;
  align-items: center;
  gap: 16px;
}

.icon-square {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.activity-icon-bg {
  background: rgba(251, 191, 36, 0.15); /* amber tint */
  border-color: rgba(251, 191, 36, 0.3);
}

.activity-icon-bg span {
  color: #fbbf24;
}

.flex-grow { flex: 1; min-width: 0; }
.flex-grow h3 { font-size: 0.9rem; font-weight: 600; margin: 0; }

.value-lg { font-size: 1.25rem; font-weight: 700; display: block; }
.unit-block { font-size: 0.6rem; color: #6b7280; text-transform: uppercase; }

.action-btn {
  padding: 6px 16px;
  border-radius: 999px;
  background: white;
  color: black;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  border: 1px solid white;
  cursor: pointer;
  transition: all 0.2s;
}
.action-btn:hover { background: rgba(255, 255, 255, 0.9); transform: scale(1.05); }

/* Small Stats */
.small-stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
}

.icon-circle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
}

.label-xs {
  font-size: 0.6rem;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 0 0 4px 0;
}

.value-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.value-md { font-size: 1.1rem; font-weight: 700; }
.sub-value { font-size: 0.65rem; }
.text-green { color: #34d399; }

/* Modal Styles */
.modal-overlay {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(0,0,0,0.8);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  padding: 20px;
}

.glass-modal {
  background: #0a0c10;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 32px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
  width: 100%; max-width: 500px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  padding: 24px 30px;
  display: flex; justify-content: space-between; align-items: flex-start;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.modal-title { font-size: 1.25rem; font-weight: 600; margin: 0; }
.modal-subtitle { font-size: 0.65rem; color: rgba(255,255,255,0.4); margin-top: 4px; letter-spacing: 0.1em; }

.close-btn {
  background: rgba(255,255,255,0.1);
  border: none;
  width: 32px; height: 32px; border-radius: 50%;
  color: white; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.2s;
}
.close-btn:hover { background: rgba(255,255,255,0.2); transform: rotate(90deg); }

.modal-body {
  padding: 30px;
  overflow-y: auto;
  display: flex; flex-direction: column; gap: 30px;
}

.section-title {
  font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}

.actigraphy-chart {
  background: rgba(255,255,255,0.05);
  border-radius: 16px;
  padding: 20px 20px 10px;
  border: 1px solid rgba(255,255,255,0.1);
}

.acti-bars {
  height: 100px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2px;
  margin-bottom: 10px;
}

.acti-bar {
  flex: 1;
  border-radius: 2px;
  transition: height 0.3s;
}

.acti-labels {
  display: flex; justify-content: space-between;
  font-size: 0.6rem; color: rgba(255,255,255,0.3);
}

.activity-stats-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
}

.stat-box {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 12px;
  display: flex; flex-direction: column; align-items: center; text-align: center;
}

.stat-box .label { font-size: 0.6rem; color: rgba(255,255,255,0.4); margin-bottom: 4px; }
.stat-box .val { font-size: 0.9rem; font-weight: 700; }
.text-orange { color: #fbbf24; }

.event-list {
  display: flex; flex-direction: column;
}

.event-item {
  display: flex; gap: 16px;
  position: relative;
  padding-bottom: 24px;
}
.event-item:last-child { padding-bottom: 0; }

.event-time {
  font-size: 0.75rem; color: rgba(255,255,255,0.5); font-family: monospace; width: 40px; padding-top: 2px;
}

.event-line {
  display: flex; flex-direction: column; align-items: center;
}
.dot { width: 8px; height: 8px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.1); }
.bg-orange { background: #fbbf24; box-shadow: 0 0 8px rgba(251, 191, 36, 0.5); }
.bg-white { background: white; }
.line { width: 1px; flex: 1; background: rgba(255,255,255,0.1); margin-top: 4px; }
.event-item:last-child .line { display: none; }

.event-details { padding-top: 0; }
.event-type { font-size: 0.9rem; font-weight: 500; }
.event-meta { font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 2px; }

.ai-insight {
  background: rgba(251, 191, 36, 0.05);
  border: 1px solid rgba(251, 191, 36, 0.2);
  border-radius: 12px;
  padding: 16px;
  display: flex; gap: 12px;
}
.ai-insight .icon { color: #fbbf24; font-size: 1.2rem; }
.ai-insight p { font-size: 0.8rem; line-height: 1.5; color: rgba(255,255,255,0.7); margin: 0; }
.ai-insight .highlight { color: #fbbf24; font-weight: 600; }

/* Icons & Typography */
.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-weight: normal;
  font-style: normal;
  display: inline-block;
  line-height: 1;
  text-transform: none;
  letter-spacing: normal;
  word-wrap: normal;
  white-space: nowrap;
  direction: ltr;
}

.icon-xs { font-size: 14px; }
.icon-sm { font-size: 18px; }
.icon-md { font-size: 24px; }
.icon-lg { font-size: 32px; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* Scrollbar */
.custom-scroll::-webkit-scrollbar {
  display: none;
}
.custom-scroll {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>