<script setup>
import { ref } from 'vue'

const currentView = ref('list') // 'list' | 'weekly'
const selectedReport = ref(null) // Object | null

// 模拟媒体类型切换 (图片/视频)
const mediaType = ref('image') // 'image' | 'video'

// Sample data for reports
const reports = [
  {
    id: 1,
    date: '2023.10.24 • 03:45 AM',
    title: '霓虹之都的飞翔',
    excerpt: '我梦见自己在充满了赛博朋克风格的巨大都市中穿梭。建筑物的霓虹灯在雨水中反射，重力似乎消失了...',
    icon: 'sentiment_very_satisfied',
    iconColor: '#34d399', // emerald
    type: 'visualized',
    bgIcon: 'bubble_chart'
  },
  {
    id: 2,
    date: '2023.10.22 • 01:12 AM',
    title: '深海的寂静图书馆',
    excerpt: '在巨大的深海气泡中，我发现了一座存放着无数发光书籍的图书馆。成群的发光鱼类在书架间穿梭...',
    icon: 'water_drop',
    iconColor: '#93c5fd', // blue
    type: 'processing',
    bgIcon: 'waves'
  },
  {
    id: 3,
    date: '2023.10.20 • 05:20 AM',
    title: '时钟森林的日出',
    excerpt: '森林里的树木都是巨大的钟表机构，随着滴答声而生长。当三个太阳同时升起时，所有的钟表开始反向走动...',
    icon: 'wb_sunny',
    iconColor: '#fcd34d', // amber
    type: 'visualized',
    bgIcon: 'landscape'
  },
  {
    id: 4,
    date: '2023.10.18 • 02:30 AM',
    title: '几何碎片的迷宫',
    excerpt: '一个没有尽头的白色空间，充满了漂浮的几何形状。我需要通过旋转这些形状来拼凑出离开的阶梯...',
    icon: 'auto_fix_normal',
    iconColor: '#c084fc', // purple
    type: 'default',
    bgIcon: 'schema'
  },
  {
    id: 5,
    date: '2023.10.15 • 04:05 AM',
    title: '云端列车',
    excerpt: '坐在一列由云彩构成的透明列车上，乘客都是半透明的人影。列车不经过车站，而是穿过巨大的漂浮岛屿...',
    icon: 'bedtime',
    iconColor: '#9ca3af', // gray
    type: 'default',
    bgIcon: 'cloud'
  }
]

const openReport = (report) => {
  selectedReport.value = report
  mediaType.value = 'image' // Default to image
}

const closeReport = () => {
  selectedReport.value = null
}

const openWeekly = () => {
  currentView.value = 'weekly'
}

const closeWeekly = () => {
  currentView.value = 'list'
}
</script>

<template>
  <div class="report-container custom-scroll relative">
    
    <!-- === ARCHIVE / WEEKLY HEADER CONTROLS === -->
    <!-- Shared Header for both views based on design consistency -->
    <div class="content-wrapper">
      <header class="main-header">
        <h1 class="glow-title">我的梦境报告</h1>
        <p class="subtitle">USER DREAM REPORTS ARCHIVE</p>
      </header>

      <!-- Controls Bar -->
      <div class="controls-bar glass-panel-sm">
        <div class="search-box">
          <span class="material-symbols-outlined search-icon">search</span>
          <input class="search-input" placeholder="搜索梦境关键词..." type="text"/>
        </div>
        
        <div class="actions-group">
          <button class="icon-btn">
            <span class="material-symbols-outlined">calendar_month</span>
            按日期
          </button>
          <button class="icon-btn">
            <span class="material-symbols-outlined">mood</span>
            按情绪
          </button>
          <div class="divider"></div>
          <button @click="currentView === 'list' ? openWeekly() : closeWeekly()" class="primary-btn" :class="{ 'active': currentView === 'weekly' }">
            <span class="material-symbols-outlined icon-lg">{{ currentView === 'list' ? 'analytics' : 'view_list' }}</span>
            {{ currentView === 'list' ? '梦境周报' : '返回列表' }}
          </button>
        </div>
      </div>

      <!-- === LIST VIEW === -->
      <div v-if="currentView === 'list'" class="report-grid">
        <div v-for="report in reports" :key="report.id" class="glass-card report-card" @click="openReport(report)">
          <div class="card-visual">
             <div class="bg-icon-wrapper">
               <span class="material-symbols-outlined bg-icon">{{ report.bgIcon }}</span>
             </div>
             <div v-if="report.type !== 'default'" class="type-badge">
               {{ report.type }}
             </div>
          </div>
          <div class="card-content">
            <div class="meta-row">
              <span class="date">{{ report.date }}</span>
              <span class="material-symbols-outlined" :style="{ color: report.iconColor }">{{ report.icon }}</span>
            </div>
            <h3 class="card-title">{{ report.title }}</h3>
            <p class="excerpt">{{ report.excerpt }}</p>
            <div class="card-footer">
              <button class="read-more-btn">查看详情</button>
            </div>
          </div>
        </div>

        <!-- Add New Card -->
        <div class="glass-card add-card">
          <div class="add-circle">
            <span class="material-symbols-outlined add-icon">add</span>
          </div>
          <span class="add-text">记录新梦境</span>
        </div>
      </div>

      <!-- === WEEKLY DASHBOARD VIEW === -->
      <Transition name="fade-slide">
        <div v-if="currentView === 'weekly'" class="weekly-dashboard">
          
          <!-- Row 1: Mood (Wide) + Themes (Narrow) -->
          <div class="dashboard-row row-1">
            <!-- Mood Map -->
            <div class="glass-panel mood-map-panel">
              <div class="panel-header">
                <h2 class="panel-title">
                  <span class="material-symbols-outlined">lens_blur</span>
                  Weekly Mood Map 情绪图谱
                </h2>
                <span class="date-range">Oct 18 - Oct 24</span>
              </div>
              <div class="cloud-container">
                <span class="cloud-tag tag-lg tag-green">宁静 (40%)</span>
                <span class="cloud-tag tag-sm">好奇</span>
                <span class="cloud-tag tag-xl tag-blue">探索 (25%)</span>
                <span class="cloud-tag tag-xs">迷茫</span>
                <span class="cloud-tag tag-md tag-yellow">愉悦</span>
                <span class="cloud-tag tag-sm">怀旧</span>
                <span class="cloud-tag tag-lg tag-purple">奇幻</span>
              </div>
            </div>

            <!-- Recurring Themes -->
            <div class="glass-panel themes-panel">
              <h2 class="panel-title">
                <span class="material-symbols-outlined">auto_awesome_motion</span>
                Recurring Themes<br>核心意象
              </h2>
              <div class="themes-list">
                <div class="theme-item">
                  <div class="theme-icon"><span class="material-symbols-outlined">water</span></div>
                  <div>
                    <p class="theme-name">流动的水</p>
                    <p class="theme-count">Appeared 3 times</p>
                  </div>
                </div>
                <div class="theme-item">
                  <div class="theme-icon"><span class="material-symbols-outlined">apartment</span></div>
                  <div>
                    <p class="theme-name">无限延伸的建筑</p>
                    <p class="theme-count">Appeared 2 times</p>
                  </div>
                </div>
                <div class="theme-item">
                  <div class="theme-icon"><span class="material-symbols-outlined">flight</span></div>
                  <div>
                    <p class="theme-name">无重力飞行</p>
                    <p class="theme-count">Appeared 2 times</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Row 2: Sleep (Narrow) + Quote (Wide) -->
          <div class="dashboard-row row-2">
            <!-- Sleep Quality -->
            <div class="glass-panel sleep-panel">
              <div class="panel-header">
                <h2 class="panel-title">
                  <span class="material-symbols-outlined">bar_chart</span>
                  Sleep Quality<br>睡眠质量
                </h2>
                <div class="score-block">
                  <div class="score">82<span class="score-max">/100</span></div>
                  <div class="score-label">Weekly Avg</div>
                </div>
              </div>
              <div class="chart-container">
                <div class="bar-chart">
                  <div v-for="d in 7" :key="d" class="chart-col">
                    <div class="stacked-bar" :style="{height: (40 + Math.random() * 50) + '%'}">
                      <div class="segment rem"></div>
                      <div class="segment light"></div>
                      <div class="segment deep"></div>
                    </div>
                    <span class="day-label">{{ ['10.18', '10.19', '10.20', '10.21', '10.22', '10.23', 'Today'][d-1] }}</span>
                  </div>
                </div>
                <div class="legend">
                  <div class="legend-item"><span class="dot rem"></span>REM</div>
                  <div class="legend-item"><span class="dot light"></span>Light</div>
                  <div class="legend-item"><span class="dot deep"></span>Deep</div>
                </div>
              </div>
            </div>

            <!-- Quote -->
            <div class="glass-panel quote-panel">
              <div class="quote-bg-icon">
                <span class="material-symbols-outlined">format_quote</span>
              </div>
              <div class="quote-content">
                <p class="quote-label">本周潜意识语录 / Subconscious Quote</p>
                <h3 class="quote-text">
                  “在流动的霓虹与深邃的海洋之间，你正在寻找一种能够跨越现实维度的平衡。所有的飞翔都源于对自由的深度渴望。”
                </h3>
                <div class="quote-divider"></div>
                <p class="quote-author">— 梦境AI分析师 · Dream Weaver AI</p>
              </div>
            </div>
          </div>

        </div>
      </Transition>
    </div>

    <!-- === DETAIL MODAL === -->
    <Transition name="fade">
      <div v-if="selectedReport" class="modal-overlay" @click.self="closeReport">
        <div class="glass-modal">
          <button @click="closeReport" class="close-modal-btn">
            <span class="material-symbols-outlined">close</span>
          </button>
          
          <div class="modal-layout">
            <!-- Media Side -->
            <div class="modal-media">
              <div class="media-container">
                <img 
                  v-if="mediaType === 'image'"
                  alt="Dream Visualization" 
                  class="media-content" 
                  src="https://images.unsplash.com/photo-1518066000714-58c45f1a2c0a?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80"
                />
                <div v-else class="media-content video-placeholder">
                  <span class="material-symbols-outlined">movie</span>
                </div>
                
                <div class="ai-tag">
                  <span class="pulse-dot"></span>
                  AI GENERATED VISION & VIDEO
                </div>

                <div class="play-overlay">
                  <button class="play-btn">
                    <span class="material-symbols-outlined">play_arrow</span>
                  </button>
                  <p class="play-text">Play Dream Video</p>
                </div>
              </div>

              <!-- Media Thumbnails/Controls -->
              <div class="media-controls">
                <div 
                  class="thumb-item" 
                  :class="{ active: mediaType === 'image' }"
                  @click="mediaType = 'image'"
                >
                  <img src="https://images.unsplash.com/photo-1518066000714-58c45f1a2c0a?ixlib=rb-4.0.3&auto=format&fit=crop&w=200&q=80" />
                  <div class="thumb-icon"><span class="material-symbols-outlined">image</span></div>
                </div>
                <div 
                  class="thumb-item" 
                  :class="{ active: mediaType === 'video' }"
                  @click="mediaType = 'video'"
                >
                  <div class="video-thumb-placeholder"></div>
                  <div class="thumb-icon"><span class="material-symbols-outlined">movie</span></div>
                </div>
                
                <div class="progress-section">
                   <div class="progress-bar">
                      <div class="progress-fill" style="width: 30%"></div>
                   </div>
                   <span class="time-code">00:12 / 00:30</span>
                </div>
              </div>
            </div>

            <!-- Content Side -->
            <div class="modal-content custom-scroll">
              <section class="content-body">
                <div class="content-header">
                  <span class="date-label">{{ selectedReport.date }}</span>
                  <h2 class="detail-title">{{ selectedReport.title }}</h2>
                </div>
                
                <div class="text-body">
                  <p>{{ selectedReport.excerpt }}</p>
                  <p>在飞行的过程中，我遇到了一群同样在空中漫步的透明鲸鱼。它们的鸣叫声像合成器的旋律，在大楼间回荡。我感受到前所未有的自由，那种由于摆脱物理束缚而带来的极度快感，让我在梦中甚至能感觉到风掠过指尖的清凉感...</p>
                </div>
              </section>
              
              <div class="divider-line"></div>
              
              <section class="analysis-section">
                <h3 class="section-label">Deep Analysis / 深度分析</h3>
                <div class="analysis-grid">
                  <div class="analysis-col">
                    <div class="col-header">
                      <span class="material-symbols-outlined">favorite</span>
                      <span>生理状态</span>
                    </div>
                    <div class="data-group">
                      <div class="data-row">
                        <span>平均心率</span>
                        <span class="highlight">72 BPM</span>
                      </div>
                      <div class="data-row">
                        <span>REM 时长</span>
                        <span class="highlight">42 MIN</span>
                      </div>
                    </div>
                  </div>
                  <div class="analysis-col">
                    <div class="col-header">
                      <span class="material-symbols-outlined">psychology_alt</span>
                      <span>心理关键词</span>
                    </div>
                    <div class="tags-row">
                      <span class="tag">焦虑</span>
                      <span class="tag">自由</span>
                      <span class="tag">转换</span>
                    </div>
                  </div>
                </div>

                <div class="insight-box">
                  <div class="insight-header">
                    <span class="material-symbols-outlined highlight-blue">auto_awesome</span>
                    <span class="insight-label">潜意识洞察</span>
                  </div>
                  <p class="insight-text">
                    “飞翔象征着你近期在现实生活中对现状的超越渴望。霓虹色彩代表了创造力的迸发，而鲸鱼的出现提示你，在追求高效与速度的同时，内心深处正寻求一种更加广阔、宁静的精神归宿。这是一个极具积极意义的转化之梦。”
                  </p>
                </div>
              </section>
              
              <button class="export-btn">导出报告 Export PDF</button>
            </div>
          </div>
        </div>
      </div>
    </Transition>

  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
* { box-sizing: border-box; }

.report-container {
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
.main-header { text-align: center; margin-bottom: 40px; }
.glow-title { font-size: 2.5rem; font-weight: 300; letter-spacing: 0.15em; text-shadow: 0 0 10px rgba(255,255,255,0.5); margin: 0; }
.subtitle { font-size: 0.75rem; color: rgba(255,255,255,0.5); letter-spacing: 0.4em; margin-top: 15px; }

/* Controls */
.controls-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  margin-bottom: 40px;
  gap: 20px;
  flex-wrap: wrap;
}

.glass-panel-sm {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
}

.search-box {
  position: relative;
  flex: 1;
  max-width: 350px;
}
.search-icon { position: absolute; left: 16px; top: 50%; transform: translateY(-50%); color: rgba(255,255,255,0.3); font-size: 1.2rem; }
.search-input {
  width: 100%;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 10px 16px 10px 48px;
  color: white;
  font-size: 0.9rem;
  outline: none;
  transition: all 0.2s;
}
.search-input:focus { border-color: rgba(255,255,255,0.4); background: rgba(255,255,255,0.1); }

.actions-group { display: flex; align-items: center; gap: 12px; }
.icon-btn {
  display: flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  color: rgba(255,255,255,0.6);
  padding: 8px 16px;
  border-radius: 12px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}
.icon-btn:hover { background: rgba(255,255,255,0.1); color: white; }
.divider { width: 1px; height: 24px; background: rgba(255,255,255,0.1); margin: 0 8px; }
.primary-btn {
  display: flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.4);
  color: white;
  padding: 8px 24px;
  border-radius: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 0 10px rgba(255,255,255,0.1);
}
.primary-btn:hover, .primary-btn.active { background: white; color: black; }

/* Report Grid (List View) */
.report-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 30px;
}

.glass-card {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.2); /* Thinner border */
  border-radius: 32px;
  overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
  cursor: pointer;
}
.glass-card:hover { transform: translateY(-5px); border-color: rgba(255,255,255,0.4); }

.report-card { display: flex; flex-direction: column; }

.card-visual {
  height: 180px;
  background: rgba(255,255,255,0.05);
  position: relative;
  display: flex; align-items: center; justify-content: center;
}
.bg-icon { font-size: 4rem; color: rgba(255,255,255,0.2); }
.type-badge {
  position: absolute; top: 16px; right: 16px;
  background: rgba(0,0,0,0.5);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.2);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(255,255,255,0.9);
}

.card-content { padding: 28px; display: flex; flex-direction: column; flex: 1; }
.meta-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.date { font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
.card-title { font-size: 1.25rem; margin: 0 0 10px 0; font-weight: 600; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
.excerpt { font-size: 0.85rem; color: rgba(255,255,255,0.5); line-height: 1.6; margin: 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.card-footer { margin-top: 20px; }
.read-more-btn {
  width: 100%; padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.2);
  background: rgba(255,255,255,0.05);
  color: white;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  transition: all 0.2s;
}
.report-card:hover .read-more-btn { background: white; color: black; }

.add-card {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 300px;
  border: 2px dashed rgba(255,255,255,0.2);
  background: transparent;
}
.add-card:hover { background: rgba(255,255,255,0.05); }
.add-circle {
  width: 64px; height: 64px; border-radius: 50%;
  border: 2px dashed rgba(255,255,255,0.2);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 16px;
  transition: transform 0.2s;
}
.add-card:hover .add-circle { transform: scale(1.1); border-color: white; }
.add-icon { font-size: 2rem; color: rgba(255,255,255,0.2); }
.add-card:hover .add-icon { color: white; }
.add-text { font-size: 0.75rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; }
.add-card:hover .add-text { color: white; }

/* Weekly Dashboard View */
.weekly-dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.dashboard-row {
  display: grid;
  gap: 24px;
}

/* Row 1: Mood (2fr) Themes (1fr) */
.row-1 {
  grid-template-columns: 2fr 1fr;
}
/* Row 2: Sleep (1fr) Quote (2fr) */
.row-2 {
  grid-template-columns: 1fr 2fr;
}

@media (max-width: 900px) {
  .row-1, .row-2 { grid-template-columns: 1fr; }
}

.glass-panel {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 32px;
  padding: 32px;
}

.panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
.panel-title { font-size: 1.1rem; font-weight: 500; display: flex; align-items: center; gap: 10px; margin: 0; line-height: 1.2; }
.date-range { font-size: 0.65rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.1em; }

/* Mood Map */
.cloud-container { height: 250px; display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 16px; padding: 20px; }
.cloud-tag { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 50px; color: rgba(255,255,255,0.8); display: inline-block; transition: all 0.3s; }
.cloud-tag:hover { background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.5); transform: translateY(-2px); }
.tag-lg { padding: 12px 30px; font-size: 1.2rem; font-weight: 300; }
.tag-xl { padding: 16px 40px; font-size: 1.5rem; font-weight: 500; }
.tag-sm { padding: 8px 20px; font-size: 0.9rem; }
.tag-xs { padding: 4px 12px; font-size: 0.75rem; color: rgba(255,255,255,0.5); }
.tag-green { box-shadow: 0 0 20px rgba(52, 211, 153, 0.2); color: white; }
.tag-blue { box-shadow: 0 0 20px rgba(96, 165, 250, 0.3); color: #bfdbfe; }
.tag-yellow { box-shadow: 0 0 15px rgba(251, 191, 36, 0.2); }
.tag-purple { box-shadow: 0 0 20px rgba(192, 132, 252, 0.2); color: #e9d5ff; }

/* Themes */
.themes-panel { display: flex; flex-direction: column; }
.themes-list { display: flex; flex-direction: column; gap: 16px; margin-top: auto; flex: 1; justify-content: center; }
.theme-item { display: flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.05); padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); }
.theme-icon { width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.6); }
.theme-name { margin: 0; font-size: 0.9rem; font-weight: 500; }
.theme-count { margin: 2px 0 0 0; font-size: 0.65rem; color: rgba(255,255,255,0.3); text-transform: uppercase; }

/* Sleep */
.sleep-panel { display: flex; flex-direction: column; }
.score-block { text-align: right; }
.score { font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 10px rgba(255,255,255,0.5); }
.score-max { font-size: 0.8rem; font-weight: 400; opacity: 0.5; }
.score-label { font-size: 0.6rem; text-transform: uppercase; opacity: 0.4; letter-spacing: 0.1em; }
.chart-container { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; margin-top: 20px; }
.bar-chart { display: flex; justify-content: space-between; align-items: flex-end; height: 160px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
.chart-col { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 8px; height: 100%; justify-content: flex-end; }
.stacked-bar { width: 100%; max-width: 16px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; display: flex; flex-direction: column-reverse; }
.segment { width: 100%; }
.segment.rem { height: 20%; background: rgba(255,255,255,0.9); box-shadow: 0 0 5px white; }
.segment.light { height: 45%; background: rgba(255,255,255,0.4); }
.segment.deep { height: 35%; background: rgba(255,255,255,0.15); }
.day-label { font-size: 0.55rem; color: rgba(255,255,255,0.3); margin-top: 4px; }
.legend { display: flex; justify-content: center; gap: 12px; margin-top: 12px; }
.legend-item { font-size: 0.6rem; color: rgba(255,255,255,0.5); display: flex; align-items: center; gap: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.dot { width: 6px; height: 6px; border-radius: 50%; }
.dot.rem { background: white; }
.dot.light { background: rgba(255,255,255,0.4); }
.dot.deep { background: rgba(255,255,255,0.15); }

/* Quote */
.quote-panel { display: flex; flex-direction: column; align-items: center; text-align: center; position: relative; overflow: hidden; justify-content: center; }
.quote-bg-icon { position: absolute; top: -20px; left: -20px; opacity: 0.05; pointer-events: none; }
.quote-bg-icon span { font-size: 20rem; }
.quote-content { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; }
.quote-label { font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5em; margin-bottom: 30px; }
.quote-text { font-size: 1.8rem; font-weight: 300; font-style: italic; line-height: 1.6; max-width: 800px; text-shadow: 0 0 10px rgba(255,255,255,0.3); margin: 0; }
.quote-divider { width: 40px; height: 1px; background: rgba(255,255,255,0.2); margin: 30px 0; }
.quote-author { font-size: 0.8rem; color: rgba(255,255,255,0.6); letter-spacing: 0.1em; }

/* Detail Modal */
.modal-overlay {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(0,0,0,0.8);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  padding: 40px;
}

.glass-modal {
  width: 100%; max-width: 1200px; height: 90vh;
  background: #05070a;
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 40px;
  overflow: hidden;
  position: relative;
  box-shadow: 0 0 60px rgba(0,0,0,0.8);
  display: flex;
}

.close-modal-btn {
  position: absolute; top: 24px; right: 24px; z-index: 20;
  width: 40px; height: 40px; border-radius: 50%;
  background: rgba(0,0,0,0.5);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.2);
  color: white;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.2s;
}
.close-modal-btn:hover { background: rgba(255,255,255,0.3); }

.modal-layout { display: flex; width: 100%; height: 100%; }

/* Media Side (Left) */
.modal-media { width: 50%; background: black; display: flex; flex-direction: column; position: relative; border-right: 1px solid rgba(255,255,255,0.1); }

.media-container { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #000; }
.media-content { width: 100%; height: 100%; object-fit: cover; opacity: 0.8; transition: opacity 0.5s; }
.media-container:hover .media-content { opacity: 0.6; }
.video-placeholder { display: flex; align-items: center; justify-content: center; background: #111; }
.video-placeholder span { font-size: 4rem; color: #333; }

.ai-tag {
  position: absolute; top: 30px; left: 30px;
  background: rgba(0,0,0,0.6); backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
  padding: 8px 16px; border-radius: 30px;
  font-size: 0.6rem; color: rgba(255,255,255,0.9);
  display: flex; align-items: center; gap: 8px; letter-spacing: 0.15em; font-weight: 700;
}
.pulse-dot { width: 6px; height: 6px; background: #60a5fa; border-radius: 50%; animation: pulse 2s infinite; }

.play-overlay { 
  position: absolute; inset: 0; 
  display: flex; flex-direction: column; align-items: center; justify-content: center; 
  gap: 20px; pointer-events: none; 
}
.play-btn {
  width: 90px; height: 90px; border-radius: 50%;
  background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.3);
  color: white; font-size: 3rem; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  pointer-events: auto;
  transition: transform 0.2s, background 0.2s;
  box-shadow: 0 0 30px rgba(255,255,255,0.1);
}
.play-btn:hover { transform: scale(1.1); background: rgba(255,255,255,0.2); }
.play-text { font-size: 0.7rem; letter-spacing: 0.3em; text-transform: uppercase; color: rgba(255,255,255,0.8); opacity: 0; transition: opacity 0.3s; transform: translateY(10px); }
.play-btn:hover + .play-text { opacity: 1; transform: translateY(0); }

/* Media Thumbnails/Controls */
.media-controls { 
  height: 100px; 
  background: rgba(0,0,0,0.8); 
  border-top: 1px solid rgba(255,255,255,0.1); 
  padding: 0 30px; 
  display: flex; align-items: center; gap: 20px;
  backdrop-filter: blur(20px);
}

.thumb-item {
  width: 70px; height: 50px;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  border: 2px solid transparent;
  cursor: pointer;
  opacity: 0.5;
  transition: all 0.2s;
}
.thumb-item.active { border-color: white; opacity: 1; box-shadow: 0 0 10px rgba(255,255,255,0.3); }
.thumb-item img { width: 100%; height: 100%; object-fit: cover; }
.video-thumb-placeholder { width: 100%; height: 100%; background: #222; }
.thumb-icon { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); }
.thumb-icon span { font-size: 1.2rem; color: white; }

.progress-section { flex: 1; display: flex; align-items: center; gap: 15px; margin-left: 20px; padding-left: 20px; border-left: 1px solid rgba(255,255,255,0.1); }
.progress-bar { flex: 1; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; position: relative; overflow: hidden; }
.progress-fill { height: 100%; background: white; box-shadow: 0 0 10px white; }
.time-code { font-family: monospace; font-size: 0.7rem; color: rgba(255,255,255,0.4); }

/* Content Side (Right) */
.modal-content { width: 50%; padding: 60px; overflow-y: auto; background: linear-gradient(180deg, rgba(10,12,18,0.9) 0%, rgba(5,7,10,1) 100%); display: flex; flex-direction: column; }
.content-body { flex: 1; }

.content-header { margin-bottom: 30px; }
.date-label { font-size: 0.7rem; color: rgba(255,255,255,0.4); letter-spacing: 0.2em; text-transform: uppercase; font-weight: 700; display: block; margin-bottom: 10px; }
.detail-title { font-size: 2.5rem; font-weight: 300; margin: 0; text-shadow: 0 0 20px rgba(255,255,255,0.3); line-height: 1.2; letter-spacing: 0.05em; }

.text-body p { color: rgba(255,255,255,0.7); line-height: 1.8; margin-bottom: 24px; font-weight: 300; font-size: 1.05rem; }
.divider-line { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 40px 0; }

.section-label { font-size: 0.7rem; color: rgba(255,255,255,0.3); letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 24px; font-weight: 700; }
.analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-bottom: 40px; }

.col-header { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.1em; }
.data-group { display: flex; flex-direction: column; gap: 12px; }
.data-row { display: flex; justify-content: space-between; font-size: 0.9rem; color: rgba(255,255,255,0.4); padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.data-row .highlight { color: white; font-weight: 500; }

.tags-row { display: flex; gap: 8px; flex-wrap: wrap; }
.tag { padding: 6px 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; font-size: 0.7rem; color: rgba(255,255,255,0.7); }

.insight-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; }
.insight-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.insight-label { font-size: 0.7rem; color: #93c5fd; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; }
.highlight-blue { color: #60a5fa; font-size: 1.2rem; }
.insight-text { font-style: italic; color: rgba(255,255,255,0.6); font-size: 0.95rem; margin: 0; line-height: 1.6; }

.export-btn { width: 100%; margin-top: auto; padding: 20px; background: white; color: black; font-weight: 700; border-radius: 16px; border: none; text-transform: uppercase; letter-spacing: 0.2em; cursor: pointer; box-shadow: 0 0 30px rgba(255,255,255,0.1); transition: all 0.2s; font-size: 0.8rem; }
.export-btn:hover { background: rgba(255,255,255,0.9); transform: translateY(-2px); box-shadow: 0 0 40px rgba(255,255,255,0.2); }

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.fade-slide-enter-active, .fade-slide-leave-active { transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
.fade-slide-enter-from, .fade-slide-leave-to { opacity: 0; transform: translateY(20px); }

/* Scrollbar */
.custom-scroll::-webkit-scrollbar { display: none; }
.custom-scroll { -ms-overflow-style: none; scrollbar-width: none; }
</style>