<template>
  <!-- 1. 应用你提供的 page-wrapper 类作为根容器，支持自然滚动 -->
  <div class="page-wrapper">
    
    <!-- 2. 头部样式调整为你提供的 page-header -->
    <header class="page-header">
      <h1>生理数据</h1>
      <p>身体指标与睡眠相关数据的可视化面板</p>
    </header>

    <!-- 3. 内容区域：保留之前的玻璃拟态设计，但放入文档流中 -->
    <div class="dashboard-content">
      
      <!-- 主面板：包含上、中、下三部分 -->
      <!-- 这里我保留了玻璃质感，但去掉了之前可能导致不可滚动的绝对定位或固定高度 -->
      <main class="main-glass-panel">
        
        <!-- 第一行：三个生理指标小卡片 -->
        <div class="top-row-grid">
          <!-- 心率变异性 -->
          <div class="glass-card small-card">
            <div class="card-content">
              <div class="card-header">
                <span class="dot red-pulse"></span>
                <span class="label">心率变异性</span>
              </div>
              <div class="card-value">
                65 <span class="unit">ms</span>
              </div>
              <div class="card-footer text-green">
                <span class="material-symbols-outlined icon-sm">trending_up</span>
                较昨晚 +5%
              </div>
            </div>
            <div class="card-icon-circle">
              <span class="material-symbols-outlined">favorite</span>
            </div>
          </div>

          <!-- 呼吸频率 -->
          <div class="glass-card small-card">
            <div class="card-content">
              <div class="card-header">
                <span class="dot blue-dot"></span>
                <span class="label">呼吸频率</span>
              </div>
              <div class="card-value">
                14 <span class="unit">次/分</span>
              </div>
              <div class="card-footer">
                节律平稳
              </div>
            </div>
            <div class="card-icon-circle">
              <span class="material-symbols-outlined">air</span>
            </div>
          </div>

          <!-- 血氧饱和度 -->
          <div class="glass-card small-card">
            <div class="card-content">
              <div class="card-header">
                <span class="dot green-dot"></span>
                <span class="label">血氧饱和度</span>
              </div>
              <div class="card-value">
                98%
              </div>
              <div class="card-footer text-green">
                理想水平
              </div>
            </div>
            <div class="card-icon-circle">
              <span class="material-symbols-outlined">water_drop</span>
            </div>
          </div>
        </div>

        <!-- 第二行：睡眠周期活动（大图表） -->
        <div class="glass-card chart-card">
          <div class="chart-header">
            <div>
              <h3>睡眠周期活动</h3>
              <p>与梦境强度阶段相关</p>
            </div>
            <div class="time-toggle">
              <button 
                v-for="time in ['1小时', '4小时', '8小时']" 
                :key="time"
                :class="{ active: activeTime === time }"
                @click="activeTime = time"
              >
                {{ time }}
              </button>
            </div>
          </div>
          
          <div class="chart-body">
            <div class="custom-scrollbar">
               <div class="thumb"></div>
            </div>
            <div class="svg-wrapper">
              <svg viewBox="0 0 800 200" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="rgba(255, 255, 255, 0.3)" />
                    <stop offset="100%" stop-color="rgba(255, 255, 255, 0)" />
                  </linearGradient>
                </defs>
                <path d="M0,150 C150,150 200,50 300,80 C400,110 500,160 600,120 C700,80 750,100 800,140 V200 H0 Z" fill="url(#lineGradient)" />
                <path d="M0,150 C150,150 200,50 300,80 C400,110 500,160 600,120 C700,80 750,100 800,140" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" class="glow-line" />
                <circle cx="300" cy="80" r="6" fill="white" class="glow-point" />
              </svg>
              <div class="floating-tag">
                <div class="tag-content">
                  <strong>REM 阶段 2</strong>
                  <span>凌晨 02:45</span>
                </div>
                <div class="tag-line"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- 第三行：双列布局 -->
        <div class="bottom-row-grid">
          <!-- 脑电波 -->
          <div class="glass-card wide-card">
            <div class="left-icon">
              <span class="material-symbols-outlined">waves</span>
            </div>
            <div class="middle-text">
              <h3>脑电波模式</h3>
              <p>正在分析 Theta 波</p>
            </div>
            <div class="right-stat">
              <span class="big-text">Theta</span>
              <span class="small-text">4-8 HZ</span>
            </div>
          </div>

          <!-- 身体活动 -->
          <div class="glass-card wide-card">
            <div class="left-icon">
              <span class="material-symbols-outlined">notifications_active</span>
            </div>
            <div class="middle-text">
              <h3>检测到身体活动</h3>
              <p>凌晨 04:20 有轻微活动</p>
            </div>
            <div class="right-action">
              <button class="action-btn">查看</button>
            </div>
          </div>
        </div>
      </main>

      <!-- 底部独立区域：三个并排矩形 -->
      <footer class="footer-stats">
        <div class="glass-card footer-card">
          <div class="circle-icon">
            <span class="material-symbols-outlined">bedtime</span>
          </div>
          <div class="footer-info">
            <p>睡眠质量</p>
            <div class="footer-val">
              84% <span class="text-green small">+2%</span>
            </div>
          </div>
        </div>

        <div class="glass-card footer-card">
          <div class="circle-icon">
            <span class="material-symbols-outlined">auto_fix_high</span>
          </div>
          <div class="footer-info">
            <p>梦境生动度</p>
            <div class="footer-val">高</div>
          </div>
        </div>

        <div class="glass-card footer-card">
          <div class="circle-icon">
            <span class="material-symbols-outlined">sync</span>
          </div>
          <div class="footer-info">
            <p>分析状态</p>
            <div class="footer-val text-green">已同步</div>
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const activeTime = ref('1小时');
</script>

<style scoped>
/* 引入字体 */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* --- 核心修改：应用你要求的底层样式 --- */
.page-wrapper {
  padding: 40px 40px 60px;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
  /* 确保页面可以自然滚动，去除之前的 fixed/height:100vh */
  width: 100%;
  box-sizing: border-box; 
  /* 如果你的父组件没有背景色，可以在这里加一个深色背景，
     或者保持透明由父级控制。这里暂且保持透明以融入你的系统。 */
}

/* 头部样式：严格对应你提供的代码 */
.page-header {
  margin-bottom: 32px; /* 给内容留出间距 */
}

.page-header h1 {
  font-size: 26px;
  margin: 0 0 8px;
  font-weight: 600; /* 稍微加粗一点标题使其更清晰 */
}

.page-header p {
  margin: 0;
  color: #94a3b8;
  font-size: 14px;
}

/* --- 内容区域布局 --- */
.dashboard-content {
  max-width: 1200px; /* 限制内容过宽 */
  margin: 0 auto;    /* 居中显示 */
}

/* --- 下方保留玻璃拟态的所有设计 --- */

/* 主大面板样式 */
.main-glass-panel {
  /* 对应你给的 .placeholder-card 背景逻辑，稍微调整使其适合做大容器 */
  background: rgba(15, 23, 42, 0.4); 
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 24px;
  padding: 30px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  margin-bottom: 24px;
  /* 确保不会溢出屏幕，让其自适应高度 */
}

/* 通用玻璃卡片 */
.glass-card {
  background: rgba(255, 255, 255, 0.08); /* 稍微调亮一点以在深色背景上显现 */
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}
.glass-card:hover {
  background: rgba(255, 255, 255, 0.12);
}

/* 1. 第一行 Grid */
.top-row-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.small-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 120px;
}
.card-content {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 100%;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.label { font-size: 12px; color: #cbd5e1; font-weight: 600; }
.card-value { font-size: 36px; font-weight: 700; line-height: 1; margin: 10px 0; color: white; }
.unit { font-size: 12px; font-weight: 400; color: #94a3b8; }
.card-footer { font-size: 10px; color: #94a3b8; display: flex; align-items: center; }
.card-icon-circle {
  width: 48px; height: 48px; border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex; align-items: center; justify-content: center;
  background: rgba(255, 255, 255, 0.05);
  color: white;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: block; }
.red-pulse { background: #ff6b6b; box-shadow: 0 0 8px #ff6b6b; }
.blue-dot { background: #38bdf8; }
.green-dot { background: #4ade80; }
.text-green { color: #4ade80; }
.icon-sm { font-size: 14px; margin-right: 2px; }

/* 2. 第二行 图表 Grid */
.chart-card {
  min-height: 320px;
  position: relative;
  display: flex;
  flex-direction: column;
}
.chart-header {
  display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;
}
.chart-header h3 { margin: 0; font-size: 18px; font-weight: 500; color: white; }
.chart-header p { margin: 4px 0 0; font-size: 11px; color: #94a3b8; }

.time-toggle {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 20px; padding: 4px; display: flex; gap: 4px;
}
.time-toggle button {
  background: transparent; border: none; color: #94a3b8;
  font-size: 11px; padding: 4px 12px; border-radius: 16px;
  cursor: pointer; transition: all 0.2s;
}
.time-toggle button.active {
  background: white; color: black; font-weight: bold;
}

.chart-body { flex: 1; position: relative; display: flex; align-items: flex-end; }
.svg-wrapper { width: 95%; height: 200px; position: relative; }
.glow-line { filter: drop-shadow(0 0 6px rgba(255,255,255,0.6)); }
.glow-point { filter: drop-shadow(0 0 8px white); }

.floating-tag {
  position: absolute; top: 30%; left: 37.5%;
  transform: translate(-50%, -100%);
  display: flex; flex-direction: column; align-items: center;
}
.tag-content {
  background: rgba(255, 255, 255, 0.95); color: black;
  padding: 6px 10px; border-radius: 8px; font-size: 10px;
  text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.tag-content strong { display: block; font-size: 11px; margin-bottom: 2px; }

/* 模拟滚动条 */
.custom-scrollbar {
  position: absolute; right: 0; top: 20px; bottom: 20px; width: 6px;
  background: rgba(255,255,255,0.05); border-radius: 3px;
}
.custom-scrollbar .thumb {
  width: 100%; height: 40%; background: rgba(255,255,255,0.3);
  border-radius: 3px; position: absolute; top: 30%;
}

/* 3. 第三行 双列 Grid */
.bottom-row-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
}
.wide-card {
  display: flex; align-items: center; gap: 16px; padding: 16px 24px; height: 100px;
}
.left-icon {
  width: 44px; height: 44px; background: rgba(255,255,255,0.1);
  border-radius: 10px; border: 1px solid rgba(255,255,255,0.2);
  display: flex; align-items: center; justify-content: center; color: white;
}
.middle-text { flex: 1; }
.middle-text h3 { margin: 0; font-size: 14px; color: white; }
.middle-text p { margin: 4px 0 0; font-size: 11px; color: #94a3b8; }
.big-text { font-size: 20px; font-weight: bold; color: white; }
.small-text { display: block; font-size: 9px; color: #94a3b8; text-transform: uppercase; text-align: right; }

.action-btn {
  background: white; color: black; border: none; padding: 8px 20px;
  border-radius: 20px; font-size: 11px; font-weight: bold; cursor: pointer;
}
.action-btn:hover { background: #f1f5f9; }

/* 4. 底部独立区域 */
.footer-stats {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
}
.footer-card {
  display: flex; align-items: center; gap: 16px; height: 80px;
  background: rgba(15, 23, 42, 0.6); /* 使用你提供的 placeholder 背景色 */
  border: 1px dashed rgba(148, 163, 184, 0.3); /* 呼应 dashed 边框风格，但保持圆角 */
}
.circle-icon {
  width: 40px; height: 40px; border-radius: 50%;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  display: flex; align-items: center; justify-content: center; color: white;
}
.footer-info p { margin: 0 0 4px 0; font-size: 10px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; }
.footer-val { font-size: 18px; font-weight: 700; color: white; }
.footer-val .small { font-size: 10px; margin-left: 4px; vertical-align: middle; }

/* 响应式 */
@media (max-width: 768px) {
  .top-row-grid, .bottom-row-grid, .footer-stats { grid-template-columns: 1fr; }
  .page-wrapper { padding: 20px; }
}
</style>


