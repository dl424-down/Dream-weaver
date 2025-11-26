
<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import * as THREE from 'three'

// --- 业务逻辑状态 ---
const dreamText = ref('')
const imageFile = ref(null)
const loading = ref(false)
const loadingImage = ref(false)
const result = ref(null)
const generatedImage = ref(null)
const error = ref('')
const canvasRef = ref(null) // 3D 画布引用

const fileName = computed(() => imageFile.value ? imageFile.value.name : '')

function onFileChange(e) {
  const files = e.target.files
  imageFile.value = files && files[0] ? files[0] : null
}

const scrollTo = (id) => {
  nextTick(() => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  })
}

async function analyze() {
  if (checkEmpty()) return
  resetState()
  loading.value = true
  
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  if (imageFile.value) form.append('image', imageFile.value)
  
  try {
    const resp = await fetch('http://localhost:8000/analyze', { method: 'POST', body: form })
    if (!resp.ok) throw new Error('星链连接失败')
    result.value = await resp.json()
    scrollTo('result-section')
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
  }
}

async function analyzeTextOnly() {
  if (checkEmpty()) return
  resetState()
  loading.value = true
  
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  
  try {
    const resp = await fetch('http://localhost:8000/analyze', { method: 'POST', body: form })
    if (!resp.ok) throw new Error('星链连接失败')
    result.value = await resp.json()
    scrollTo('result-section')
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
  }
}

async function generateImage() {
  if (checkEmpty()) return
  error.value = ''
  generatedImage.value = null
  loadingImage.value = true
  
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  
  try {
    const resp = await fetch('http://localhost:8000/generate-image', { method: 'POST', body: form })
    if (!resp.ok) throw new Error('图像具象化失败')
    const j = await resp.json()
    if (j && j.image) {
      generatedImage.value = j.image
      scrollTo('image-section')
    } else {
      throw new Error(j?.message || '虚空未返回图像')
    }
  } catch (e) {
    error.value = e.message || String(e)
    // 失败时的 SVG 占位
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450">
      <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#020617"/><stop offset="100%" stop-color="#1e1b4b"/></linearGradient></defs>
      <rect width="100%" height="100%" fill="url(#g)" />
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="serif" font-size="20" fill="#6366f1" letter-spacing="4">SIGNAL LOST</text>
    </svg>`
    generatedImage.value = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg)
  } finally {
    loadingImage.value = false
  }
}

function checkEmpty() {
  if (!dreamText.value.trim()) {
    error.value = '梦境是一片虚无，请先注入描述...'
    return true
  }
  return false
}

function resetState() {
  error.value = ''
  result.value = null
}

// --- Three.js 3D 场景逻辑 ---
let scene, camera, renderer, particles, starField
let mouseX = 0, mouseY = 0
let targetX = 0, targetY = 0
const windowHalfX = window.innerWidth / 2
const windowHalfY = window.innerHeight / 2

onMounted(() => {
  initThree()
  animate()
  document.addEventListener('mousemove', onDocumentMouseMove)
  window.addEventListener('resize', onWindowResize)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onDocumentMouseMove)
  window.removeEventListener('resize', onWindowResize)
  // 清理内存
  if (renderer) renderer.dispose()
  if (scene) scene.clear()
})

function initThree() {
  // 1. 场景与相机
  scene = new THREE.Scene()
  scene.fog = new THREE.FogExp2(0x0a0a2a, 0.001)

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 3000)
  camera.position.z = 1000

  // 2. 渲染器
  renderer = new THREE.WebGLRenderer({ canvas: canvasRef.value, antialias: true, alpha: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setClearColor(0x000000, 1)

  // 3. 创建粒子材质
  const particleTexture = (() => {
      const canvas = document.createElement('canvas')
      canvas.width = 32; canvas.height = 32;
      const ctx = canvas.getContext('2d');
      const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
      gradient.addColorStop(0, 'rgba(255,255,255,1)');
      gradient.addColorStop(0.2, 'rgba(240,240,255,0.8)');
      gradient.addColorStop(0.5, 'rgba(120,120,255,0.2)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 32, 32);
      const texture = new THREE.Texture(canvas);
      texture.needsUpdate = true;
      return texture;
  })();

  // 4. 创建主要星云粒子群
  const geometry = new THREE.BufferGeometry()
  
  // ▼▼▼ 修正了这里的空格错误 ▼▼▼
  const particleCount = 6000 
  
  const positions = new Float32Array(particleCount * 3)
  const colors = new Float32Array(particleCount * 3)

  for (let i = 0; i < particleCount; i++) {
    const x = (Math.random() - 0.5) * 2000
    const y = (Math.random() - 0.5) * 2000
    const z = Math.random() * 3000 - 1500 

    positions[i * 3] = x
    positions[i * 3 + 1] = y
    positions[i * 3 + 2] = z

    const colorType = Math.random()
    if (colorType < 0.33) { // Purple
        colors[i * 3] = 0.6 + Math.random() * 0.2
        colors[i * 3 + 1] = 0.3 + Math.random() * 0.2
        colors[i * 3 + 2] = 0.9
    } else if (colorType < 0.66) { // Blue
        colors[i * 3] = 0.2 + Math.random() * 0.2
        colors[i * 3 + 1] = 0.5 + Math.random() * 0.3
        colors[i * 3 + 2] = 1.0
    } else { // Cyan
        colors[i * 3] = 0.2
        colors[i * 3 + 1] = 0.8 + Math.random() * 0.2
        colors[i * 3 + 2] = 0.9 + Math.random() * 0.1
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

  const material = new THREE.PointsMaterial({
    size: 15,
    map: particleTexture,
    vertexColors: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.8
  })

  particles = new THREE.Points(geometry, material)
  scene.add(particles)

  // 5. 添加背景微小的星尘场
  const starGeo = new THREE.BufferGeometry()
  const starCount = 4000
  const starPos = new Float32Array(starCount * 3)
  for(let i=0; i<starCount*3; i++) {
      starPos[i] = (Math.random() - 0.5) * 4000
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
  const starMat = new THREE.PointsMaterial({
      size: 3, color: 0xaaaaaa, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.5
  })
  starField = new THREE.Points(starGeo, starMat)
  scene.add(starField)
}

function onDocumentMouseMove(event) {
  mouseX = (event.clientX - windowHalfX) / 2
  mouseY = (event.clientY - windowHalfY) / 2
}

function onWindowResize() {
  const width = window.innerWidth
  const height = window.innerHeight
  windowHalfX = width / 2
  windowHalfY = height / 2

  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

function animate() {
  requestAnimationFrame(animate)
  
  targetX = mouseX * .05
  targetY = mouseY * .05
  camera.position.x += (targetX - camera.position.x) * 0.02
  camera.position.y += (-targetY - camera.position.y) * 0.02
  camera.lookAt(scene.position)

  particles.rotation.x += 0.0005
  particles.rotation.y += 0.001
  
  const positions = particles.geometry.attributes.position.array;
  for(let i = 0; i < positions.length; i+=3) {
      positions[i+2] += 1; 
      if(positions[i+2] > 1500) {
          positions[i+2] = -1500;
      }
  }
  particles.geometry.attributes.position.needsUpdate = true;

  starField.rotation.y -= 0.0002

  renderer.render(scene, camera)
}
</script>

<template>
  <canvas ref="canvasRef" class="webgl-bg"></canvas>

  <div class="dream-universe-ui">
    <div class="content-wrapper">
      
      <header class="dream-header">
        <h1 class="glitch-title" data-text="DREAM WEAVER">DREAM WEAVER</h1>
        <div class="subtitle-line">
          <span class="line"></span>
          <span class="text-glow">解析潜意识的星图</span>
          <span class="line"></span>
        </div>
      </header>

      <div class="crystal-capsule input-enter">
        <div class="inner-content">
          <label class="holo-label">输入梦境信号</label>
          
          <div class="input-field-wrap">
            <textarea 
              v-model="dreamText" 
              class="hologram-input" 
              rows="4" 
              placeholder="我看见时间在倒流，巨大的鲸鱼游过云层..."
            ></textarea>
            <div class="corner-accents-3d">
              <div class="c-piece tl"></div><div class="c-piece tr"></div>
              <div class="c-piece bl"></div><div class="c-piece br"></div>
            </div>
          </div>

          <div class="control-deck">
            <div class="upload-module">
              <input id="file-upload" type="file" accept="image/*" @change="onFileChange" hidden />
              <label for="file-upload" class="cyber-btn small" :class="{ 'active': imageFile }">
                <span class="btn-content">
                  <i class="icon-upload"></i> {{ imageFile ? '影像已加载' : '上传视觉碎片' }}
                </span>
              </label>
              <button v-if="imageFile" @click="imageFile=null" class="remove-file">×</button>
            </div>

            <div class="action-module">
              <button class="cyber-btn primary" :disabled="loading" @click="analyze">
                <span class="btn-bg-anim"></span>
                <span class="btn-text">{{ loading ? '正在链接星弦...' : '深度解析' }}</span>
              </button>
              
              <button class="cyber-btn secondary" :disabled="loading" @click="analyzeTextOnly">
                <span class="btn-text">文本分析</span>
              </button>

              <button class="cyber-btn magic" :disabled="loading || loadingImage" @click="generateImage">
                <span class="btn-bg-anim"></span>
                <span class="btn-text">
                  {{ loadingImage ? '物质构筑中...' : '具象化梦境' }}
                </span>
              </button>
            </div>
          </div>
          
          <div v-if="error" class="system-alert">
            <span class="alert-icon">!</span> {{ error }}
          </div>
        </div>
      </div>

      <Transition name="hologram-reveal">
        <div v-if="result" id="result-section" class="analysis-deck glass-panel-3d">
          <div class="deck-header">
            <h3>解析档案</h3>
            <div class="scanner-line-anim"></div>
          </div>

          <div class="dashboard-grid">
            <div class="data-card purple floating">
              <div class="card-label">EMOTION</div>
              <div class="card-value">{{ (result.text_analysis?.emotions || []).join(' / ') || 'N/A' }}</div>
            </div>
            <div class="data-card cyan floating delay-1">
              <div class="card-label">THEME</div>
              <div class="card-value">{{ (result.text_analysis?.themes || []).join(' / ') || 'N/A' }}</div>
            </div>
            <div class="data-card pink floating delay-2">
              <div class="card-label">SYMBOLS</div>
              <div class="card-value">{{ (result.text_analysis?.keywords || []).join(' · ') || 'N/A' }}</div>
            </div>
          </div>

          <div class="text-readout">
            <div class="readout-block">
              <h4>// PSYCHOLOGICAL_MAPPING</h4>
              <p>{{ result.combined_analysis }}</p>
            </div>
            <div class="readout-block prompt-style">
              <h4>// VISUAL_PROMPT_KEY</h4>
              <p>{{ result.visualization_prompt }}</p>
            </div>
          </div>
        </div>
      </Transition>

      <Transition name="hologram-reveal">
        <div v-if="generatedImage || loadingImage" id="image-section" class="visual-deck glass-panel-3d">
           <div class="deck-header">
            <h3>梦境映射</h3>
            <div class="scanner-line-anim"></div>
          </div>
          
          <div class="image-portal-3d">
            <div v-if="loadingImage" class="loading-state-3d">
              <div class="cube-loader">
                <div class="cube-face front"></div><div class="cube-face back"></div>
                <div class="cube-face right"></div><div class="cube-face left"></div>
                <div class="cube-face top"></div><div class="cube-face bottom"></div>
              </div>
              <p class="loading-text-glitch">正在从虚空提取像素...</p>
            </div>
            <div v-else class="image-wrapper-tilt">
              <img :src="generatedImage" class="dream-result-img" />
            </div>
          </div>
        </div>
      </Transition>

    </div>
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

:root {
  --neon-purple: #a855f7;
  --neon-cyan: #06b6d4;
  --neon-pink: #ec4899;
  --glass-panel: rgba(15, 23, 42, 0.5); 
  --glass-border: rgba(255, 255, 255, 0.15);
}

.webgl-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; 
    pointer-events: none;
}

.dream-universe-ui {
  min-height: 100vh;
  color: #fff;
  font-family: 'Rajdhani', sans-serif;
  position: relative;
  z-index: 10;
  padding-bottom: 100px;
  perspective: 1000px;
}

.content-wrapper {
  max-width: 900px;
  margin: 0 auto;
  padding: 60px 20px;
}

.dream-header { text-align: center; margin-bottom: 70px; }

.glitch-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 4.5rem;
  font-weight: 900;
  letter-spacing: 8px;
  color: transparent;
  background: linear-gradient(to bottom, #fff 30%, #a5b4fc);
  -webkit-background-clip: text;
  text-shadow: 0 2px 5px rgba(0,0,0,0.5), 0 0 30px rgba(168, 85, 247, 0.8);
  animation: titlePulse 5s ease-in-out infinite alternate;
}
@keyframes titlePulse { from { filter: brightness(1); } to { filter: brightness(1.3); text-shadow: 0 0 50px rgba(168, 85, 247, 1); } }

.subtitle-line {
  display: flex; align-items: center; justify-content: center; gap: 20px;
}
.subtitle-line .line { flex: 1; max-width: 150px; height: 2px; background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent); }
.text-glow { font-size: 1.2rem; letter-spacing: 4px; color: #cbd5e1; text-shadow: 0 0 10px var(--neon-cyan); }

.crystal-capsule {
  background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(0,0,0,0.2));
  backdrop-filter: blur(30px) saturate(1.2);
  border: 1px solid var(--glass-border);
  box-shadow: 
    0 20px 50px rgba(0,0,0,0.5), 
    inset 0 2px 5px rgba(255,255,255,0.1),
    inset 0 -2px 5px rgba(0,0,0,0.3);
  border-radius: 24px;
  padding: 5px;
  transform-style: preserve-3d;
  animation: capsuleFloat 8s ease-in-out infinite;
}
@keyframes capsuleFloat { 0%, 100% { transform: translateY(0) rotateX(1deg); } 50% { transform: translateY(-15px) rotateX(-1deg); } }

.inner-content {
  background: rgba(10, 10, 30, 0.5);
  border-radius: 20px;
  padding: 35px;
  border: 1px solid rgba(168, 85, 247, 0.15);
}

.holo-label {
  font-family: 'Orbitron'; color: var(--neon-cyan); font-size: 1rem; margin-bottom: 15px; letter-spacing: 2px;
  text-shadow: 0 0 8px var(--neon-cyan);
}

.input-field-wrap { position: relative; margin-bottom: 30px; transform-style: preserve-3d;}

.hologram-input {
  width: 100%;
  background: rgba(2, 6, 23, 0.6);
  border: 1px solid rgba(99, 102, 241, 0.3);
  color: #fff;
  padding: 20px;
  font-size: 1.1rem;
  line-height: 1.8;
  border-radius: 8px;
  outline: none;
  transition: 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
  box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
}
.hologram-input:focus {
  border-color: var(--neon-purple);
  box-shadow: inset 0 0 20px rgba(0,0,0,0.8), 0 0 40px rgba(168, 85, 247, 0.3);
  background: rgba(2, 6, 23, 0.8);
  transform: translateZ(20px);
}

.corner-accents-3d {
    position: absolute; inset: -5px; pointer-events: none;
    transform: translateZ(10px);
}
.c-piece { position: absolute; width: 20px; height: 20px; border: 2px solid var(--neon-cyan); opacity: 0.7; transition: 0.3s; }
.input-field-wrap:hover .c-piece { border-color: var(--neon-pink); opacity: 1; box-shadow: 0 0 15px var(--neon-pink); }
.tl { top: 0; left: 0; border-width: 2px 0 0 2px; }
.tr { top: 0; right: 0; border-width: 2px 2px 0 0; }
.bl { bottom: 0; left: 0; border-width: 0 0 2px 2px; }
.br { bottom: 0; right: 0; border-width: 0 2px 2px 0; }

.control-deck { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 25px; margin-top: 30px;}
.action-module { display: flex; gap: 15px; flex-wrap: wrap; }

.cyber-btn {
  position: relative;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--glass-border);
  color: #fff;
  padding: 14px 30px;
  font-family: 'Rajdhani'; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;
  cursor: pointer;
  overflow: hidden;
  transition: 0.4s;
  clip-path: polygon(15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%, 0 15px);
}
.cyber-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.cyber-btn:not(:disabled):hover { transform: translateY(-3px) scale(1.02); }

.btn-bg-anim {
    position: absolute; inset: 0; 
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    transform: translateX(-100%); transition: 0.6s; z-index: 0;
}
.cyber-btn:hover .btn-bg-anim { transform: translateX(100%); }
.btn-text { position: relative; z-index: 1; text-shadow: 0 0 5px currentColor; }

.cyber-btn.primary { border-color: var(--neon-purple); box-shadow: 0 0 15px rgba(168, 85, 247, 0.2); }
.cyber-btn.primary:hover { background: rgba(168, 85, 247, 0.2); box-shadow: 0 0 40px rgba(168, 85, 247, 0.6); }

.cyber-btn.magic { border-color: var(--neon-pink); box-shadow: 0 0 15px rgba(236, 72, 153, 0.2); }
.cyber-btn.magic:hover { background: rgba(236, 72, 153, 0.2); box-shadow: 0 0 40px rgba(236, 72, 153, 0.6); }

.cyber-btn.secondary:hover { border-color: #fff; background: rgba(255,255,255,0.1); }

.cyber-btn.small { padding: 10px 20px; font-size: 0.8rem; clip-path: none; border-radius: 50px; border-color: #64748b;}
.cyber-btn.small.active { border-color: var(--neon-cyan); background: rgba(6, 182, 212, 0.2); box-shadow: 0 0 20px rgba(6, 182, 212, 0.4); }

.glass-panel-3d {
    background: rgba(20, 20, 40, 0.6);
    backdrop-filter: blur(30px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 30px;
    margin-top: 40px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    transform-style: preserve-3d;
}

.deck-header { display: flex; align-items: center; gap: 15px; margin-bottom: 25px; }
.deck-header h3 { font-family: 'Orbitron'; font-size: 1.5rem; margin: 0; color: #fff; text-shadow: 0 0 10px var(--neon-purple); }
.scanner-line-anim { flex: 1; height: 2px; background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan)); position: relative; overflow: hidden; }
.scanner-line-anim::after {
    content: ''; position: absolute; top:0; left:0; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, #fff, transparent);
    animation: scan 2s linear infinite;
}
@keyframes scan { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }

.dashboard-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 35px; transform-style: preserve-3d; }

.data-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.1);
  padding: 25px 20px;
  border-radius: 16px;
  text-align: center;
  transition: 0.4s;
  transform: translateZ(20px);
}
.data-card:hover { transform: translateZ(40px) scale(1.05); box-shadow: 0 20px 40px rgba(0,0,0,0.5); }

.data-card.purple { border-bottom: 3px solid var(--neon-purple); }
.data-card.purple:hover { box-shadow: 0 20px 40px rgba(168, 85, 247, 0.3); }
.data-card.cyan { border-bottom: 3px solid var(--neon-cyan); }
.data-card.cyan:hover { box-shadow: 0 20px 40px rgba(6, 182, 212, 0.3); }
.data-card.pink { border-bottom: 3px solid var(--neon-pink); }
.data-card.pink:hover { box-shadow: 0 20px 40px rgba(236, 72, 153, 0.3); }

.card-label { font-size: 0.8rem; color: #94a3b8; letter-spacing: 2px; margin-bottom: 8px; }
.card-value { font-size: 1.3rem; font-weight: 700; text-shadow: 0 0 10px currentColor; }

.floating { animation: cardFloat 6s ease-in-out infinite; }
.delay-1 { animation-delay: 0.2s; }
.delay-2 { animation-delay: 0.4s; }
@keyframes cardFloat { 0%, 100% { transform: translateZ(20px) translateY(0); } 50% { transform: translateZ(20px) translateY(-10px); } }

.text-readout { display: grid; gap: 25px; }
.readout-block { 
    background: rgba(0,0,0,0.3); border-left: 3px solid #64748b; padding: 25px; 
    box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
    transition: 0.3s;
}
.readout-block:hover { border-left-color: var(--neon-purple); background: rgba(0,0,0,0.5); transform: translateX(5px); }
.readout-block h4 { margin: 0 0 15px 0; color: #94a3b8; font-family: 'Orbitron'; letter-spacing: 1px; }
.readout-block p { margin: 0; line-height: 1.8; color: #e2e8f0; text-align: justify; font-size: 1.05rem; }
.prompt-style { border-left-color: var(--neon-cyan); background: rgba(6, 182, 212, 0.05); }
.prompt-style p { font-family: 'Courier New', monospace; color: #a5f3fc; }

.image-portal-3d {
    min-height: 450px;
    display: flex; align-items: center; justify-content: center;
    perspective: 1000px;
}
.image-wrapper-tilt {
    transform-style: preserve-3d;
    transition: 0.5s ease-out;
    transform: rotateX(5deg);
}
.image-wrapper-tilt:hover { transform: rotateX(0deg) scale(1.02); }

.dream-result-img {
    width: 100%; border-radius: 12px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.6), 0 0 30px rgba(168, 85, 247, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

.loading-state-3d { perspective: 800px; text-align: center;}
.cube-loader {
  width: 60px; height: 60px; position: relative; transform-style: preserve-3d;
  animation: spinCube 4s infinite linear; margin: 0 auto 30px;
}
.cube-face {
  position: absolute; width: 60px; height: 60px;
  border: 2px solid var(--neon-cyan); opacity: 0.7;
  background: rgba(6, 182, 212, 0.1);
  box-shadow: 0 0 20px var(--neon-cyan);
}
.front  { transform: rotateY(  0deg) translateZ(30px); }
.back   { transform: rotateY(180deg) translateZ(30px); }
.right  { transform: rotateY( 90deg) translateZ(30px); }
.left   { transform: rotateY(-90deg) translateZ(30px); }
.top    { transform: rotateX( 90deg) translateZ(30px); }
.bottom { transform: rotateX(-90deg) translateZ(30px); }

@keyframes spinCube { 0% { transform: rotateX(0deg) rotateY(0deg); } 100% { transform: rotateX(360deg) rotateY(360deg); } }
.loading-text-glitch { color: var(--neon-cyan); letter-spacing: 2px; font-family: 'Orbitron'; animation: textGlitch 1s infinite alternate; }
@keyframes textGlitch { 0% { opacity: 1; text-shadow: 0 0 5px var(--neon-cyan); } 100% { opacity: 0.7; text-shadow: 2px 0 10px var(--neon-purple); } }

.input-enter { animation: slideUp3D 1s cubic-bezier(0.2, 1, 0.3, 1); }
@keyframes slideUp3D { from { transform: translateY(100px) rotateX(10deg); opacity: 0; } to { transform: translateY(0) rotateX(0deg); opacity: 1; } }

.hologram-reveal-enter-active { transition: all 0.8s cubic-bezier(0.2, 1, 0.3, 1); }
.hologram-reveal-leave-active { transition: all 0.3s ease; }
.hologram-reveal-enter-from { opacity: 0; transform: translateY(50px) translateZ(-50px) rotateX(-10deg); }

@media (max-width: 768px) {
  .glitch-title { font-size: 2.8rem; }
  .dashboard-grid { grid-template-columns: 1fr; }
  .control-deck, .action-module { flex-direction: column; align-items: stretch; }
  .cyber-btn { width: 100%; }
}
</style>

