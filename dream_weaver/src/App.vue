<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'
import bgImage from './assets/image.jpg'

const dreamText = ref('')
const imageFile = ref(null)
const loading = ref(false)
const loadingImage = ref(false)
const result = ref(null)
const generatedImage = ref(null)
const error = ref('')
const canvasRef = ref(null) // 3D 画布引用
const historyVisible = ref(false)
const historyLoading = ref(false)
const historyEntries = ref([])
const selectedEntry = ref(null)
const detailLoading = ref(false)
const sidebarCollapsed = ref(false)
const deletedEntryIds = ref(new Set()) // 存储被删除的记录ID（仅前端）
const selectedEntryIds = ref(new Set()) // 存储选中的记录ID（用于综合分析）
const comprehensiveAnalysis = ref(null) // 综合分析结果
const analyzing = ref(false) // 是否正在分析
const lastEntryId = ref(null) // 最近一次分析生成的记录ID，用于绑定生成的图片
const generatedVideo = ref(null) // 生成的视频
const loadingVideo = ref(false) // 是否正在生成视频
const videoDuration = ref(5) // 视频时长（秒）
const videoSize = ref('832*480') // 视频分辨率
const videoSettingsExpanded = ref(false) // 视频参数是否展开

// 语音识别相关
let recognition = null
const speechSupported = ref(false)
const isRecording = ref(false)

const fileName = computed(() => imageFile.value ? imageFile.value.name : '')

function onFileChange(e) {
  const files = e.target.files
  imageFile.value = files && files[0] ? files[0] : null
}

async function analyze() {
  error.value = ''
  result.value = null
  if (!dreamText.value.trim()) {
    error.value = '请输入您的梦境描述'
    return
  }
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  if (imageFile.value) form.append('image', imageFile.value)
  loading.value = true
  try {
    const resp = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      body: form
    })
    if (!resp.ok) throw new Error('请求失败')
    result.value = await resp.json()
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
  }
}

async function analyzeTextOnly() {
  error.value = ''
  result.value = null
  if (!dreamText.value.trim()) {
    error.value = '请输入您的梦境描述'
    return
  }
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  loading.value = true
  try {
    const resp = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      body: form
    })
    if (!resp.ok) throw new Error('请求失败')
    result.value = await resp.json()
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
  if (lastEntryId.value) {
    form.append('entry_id', String(lastEntryId.value))
  }
  
  try {
    // 后端运行在 8000 端口
    const resp = await fetch('http://localhost:8000/generate-image', {
      method: 'POST',
      body: form
    })
    if (!resp.ok) throw new Error('请求失败')
    const j = await resp.json()
    if (j && j.image) {
      generatedImage.value = j.image
      // 如果后端返回了 entry_id，则更新 lastEntryId（防止前端状态不同步）
      if (j.entry_id) {
        lastEntryId.value = j.entry_id
      }
      scrollTo('image-section')
    } else {
      throw new Error(j?.message || '未返回图像')
    }
  } catch (e) {
    error.value = e.message || String(e)
    // 回退到一个简单的 SVG data URI，保证前端总能显示图
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450">'
      + '<rect width="100%" height="100%" fill="#0b1220" />'
      + '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
      + 'font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#d1fae5">'
      + 'Image Unavailable</text></svg>'
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
  lastEntryId.value = null
}

async function generateVideo() {
  if (checkEmpty()) return
  error.value = ''
  generatedVideo.value = null
  loadingVideo.value = true
  
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  form.append('duration', String(videoDuration.value))
  form.append('size', videoSize.value)
  if (lastEntryId.value) {
    form.append('entry_id', String(lastEntryId.value))
  }
  
  try {
    const resp = await fetch('http://localhost:8000/generate-video', { method: 'POST', body: form })
    if (!resp.ok) throw new Error('视频生成失败')
    const j = await resp.json()
    if (j && j.success && j.video_url) {
      generatedVideo.value = {
        url: j.video_url,
        local_path: j.local_path,
        message: j.message
      }
      scrollTo('video-section')
    } else {
      throw new Error(j?.error || '虚空未返回视频')
    }
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loadingVideo.value = false
  }
}

async function loadHistory() {
  historyLoading.value = true
  historyVisible.value = true
  selectedEntry.value = null
  try {
    const resp = await fetch('http://localhost:8000/dreams/history?limit=20')
    if (!resp.ok) throw new Error('获取历史记录失败')
    const data = await resp.json()
    if (data.success) {
      historyEntries.value = data.entries || []
    } else {
      error.value = data.error || '获取历史记录失败'
    }
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    historyLoading.value = false
  }
}

async function loadEntryDetail(entryId) {
  detailLoading.value = true
  try {
    const resp = await fetch(`http://localhost:8000/dreams/${entryId}`)
    if (!resp.ok) throw new Error('获取详情失败')
    const data = await resp.json()
    if (data.success) {
      selectedEntry.value = data.entry
    } else {
      error.value = data.error || '获取详情失败'
    }
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    detailLoading.value = false
  }
}

function toggleSidebar() {
  historyVisible.value = !historyVisible.value
  if (!historyVisible.value) {
    selectedEntry.value = null
  } else if (historyEntries.value.length === 0) {
    loadHistory()
  }
}

function toggleSidebarCollapse() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function closeHistory() {
  historyVisible.value = false
  selectedEntry.value = null
}

function deleteEntry(entryId, event) {
  // 阻止事件冒泡，避免触发点击查看详情
  if (event) {
    event.stopPropagation()
  }
  
  // 弹出确认对话框
  if (confirm('您确认删除吗？\n（注意：此操作仅在前端删除，数据库中的数据不会被删除）')) {
    // 添加到删除列表（仅前端）
    deletedEntryIds.value.add(entryId)
    // 如果当前查看的是被删除的记录，关闭详情
    if (selectedEntry.value && selectedEntry.value.id === entryId) {
      selectedEntry.value = null
    }
  }
}

// 计算过滤后的历史记录（排除已删除的）
const filteredHistoryEntries = computed(() => {
  return historyEntries.value.filter(entry => !deletedEntryIds.value.has(entry.id))
})

function toggleEntrySelection(entryId, event) {
  // 阻止事件冒泡，避免触发点击查看详情
  if (event) {
    event.stopPropagation()
  }
  if (selectedEntryIds.value.has(entryId)) {
    selectedEntryIds.value.delete(entryId)
  } else {
    selectedEntryIds.value.add(entryId)
  }
}

async function analyzeComprehensive() {
  if (selectedEntryIds.value.size === 0) {
    alert('请至少选择一个梦境记录进行分析')
    return
  }
  
  analyzing.value = true
  comprehensiveAnalysis.value = null
  
  try {
    const entryIds = Array.from(selectedEntryIds.value)
    const resp = await fetch('http://localhost:8000/dreams/comprehensive-analysis', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ entry_ids: entryIds })
    })
    
    if (!resp.ok) throw new Error('综合分析失败')
    const data = await resp.json()
    if (data.success) {
      comprehensiveAnalysis.value = data.analysis
      selectedEntry.value = null // 关闭详情面板
      scrollTo('comprehensive-section')
    } else {
      error.value = data.error || '综合分析失败'
    }
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    analyzing.value = false
  }
}

function clearSelection() {
  selectedEntryIds.value.clear()
  comprehensiveAnalysis.value = null
}

function formatDate(dateString) {
  if (!dateString) return '未知时间'
  try {
    // 数据库存储的是中国本地时间 "YYYY-MM-DD HH:MM:SS"
    // 直接格式化显示，不需要时区转换
    if (dateString.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)) {
      // 格式化为 "YYYY/MM/DD HH:MM"
      const [datePart, timePart] = dateString.split(' ')
      const [year, month, day] = datePart.split('-')
      const [hour, minute] = timePart.split(':')
      return `${year}/${month}/${day} ${hour}:${minute}`
    } else if (dateString.includes('T')) {
      // ISO 格式，解析后显示
      const date = new Date(dateString)
      if (!isNaN(date.getTime())) {
        return date.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        })
      }
    }
    // 其他格式，尝试直接显示
    return dateString.substring(0, 16).replace('T', ' ')
  } catch {
    // 如果解析失败，直接显示原始字符串（去掉秒数）
    if (dateString && dateString.length >= 16) {
      return dateString.substring(0, 16).replace('T', ' ')
    }
    return dateString
  }
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

// ======================
// 🎤 语音输入模块（浏览器原生 Speech Recognition）
// ======================

onMounted(() => {
  // 检查浏览器是否支持语音识别
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
  
  if (!SpeechRecognition) {
    speechSupported.value = false
    console.warn("当前浏览器不支持语音识别")
    return
  }
  
  speechSupported.value = true
  
  // 初始化语音识别
  recognition = new SpeechRecognition()
  recognition.lang = "zh-CN"          // 中文识别
  recognition.continuous = true       // 连续识别
  recognition.interimResults = false  // 只要最终结果
  
  recognition.onstart = () => {
    isRecording.value = true
    console.log("[语音] 开始录音")
  }
  
  recognition.onend = () => {
    isRecording.value = false
    console.log("[语音] 录音结束")
  }
  
  recognition.onerror = (e) => {
    console.error("[语音] 识别错误：", e)
    isRecording.value = false
    error.value = `语音识别错误: ${e.error}`
  }
  
  recognition.onresult = (event) => {
    let transcript = ""
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript
    }
    
    // 将识别结果追加到文本框中
    if (transcript.trim()) {
      dreamText.value += (dreamText.value ? " " : "") + transcript.trim()
      console.log("[语音] 识别结果：", transcript)
    }
  }
})

// 切换录音状态
function toggleRecording() {
  if (!recognition || !speechSupported.value) {
    error.value = "当前浏览器不支持语音输入"
    return
  }
  
  if (isRecording.value) {
    // 停止录音
    recognition.stop()
    isRecording.value = false
  } else {
    // 开始录音
    try {
      recognition.start()
    } catch (e) {
      console.error("[语音] 启动失败：", e)
      error.value = "无法启动语音识别，请检查麦克风权限"
    }
  }
}

function scrollTo(id) {
  const element = document.getElementById(id)
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}
</script>

<template>
  <div class="page">
    <div class="nebula" :style="{ backgroundImage: `url(${bgImage})` }" />
    <div class="container glass">
      <h1 class="title">Dream Weaver</h1>
      <p class="subtitle">在星云之间编织你的梦境</p>

      <div class="form">
        <label class="label">梦境描述</label>
        <textarea class="input" rows="5" v-model="dreamText" placeholder="例：我梦见自己在云海之上飞行..." />

        <div class="row">
          <div class="file-wrap">
            <input id="file-input" class="file" type="file" accept="image/*" @change="onFileChange" />
            <label for="file-input" class="file-btn">选择图片</label>
            <span class="file-hint">{{ imageFile ? imageFile.name : '未选择图片' }}</span>
          </div>

          <div v-if="historyLoading" class="loading-state-3d">
            <div class="cube-loader">
              <div class="cube-face front"></div><div class="cube-face back"></div>
              <div class="cube-face right"></div><div class="cube-face left"></div>
              <div class="cube-face top"></div><div class="cube-face bottom"></div>
            </div>
            <p class="loading-text-glitch">正在检索记忆库...</p>
          </div>

          <div v-else-if="historyEntries.length === 0" class="empty-history">
            <p>记忆库中暂无记录</p>
          </div>

          <div v-else class="history-list-sidebar">
            <div 
              v-for="entry in filteredHistoryEntries" 
              :key="entry.id" 
              class="history-item-sidebar"
              :class="{ 'active': selectedEntry && selectedEntry.id === entry.id }"
              @click="loadEntryDetail(entry.id)"
            >
              <div class="history-item-preview">
                <div class="preview-header">
                  <input 
                    type="checkbox" 
                    class="entry-checkbox"
                    :checked="selectedEntryIds.has(entry.id)"
                    @click.stop="toggleEntrySelection(entry.id, $event)"
                    @change="() => {}"
                  />
                  <span class="history-id">#{{ entry.id }}</span>
                  <span class="history-date-small">{{ formatDate(entry.created_at) }}</span>
                  <button 
                    class="delete-btn" 
                    @click.stop="deleteEntry(entry.id, $event)"
                    title="删除（仅前端）"
                  >
                    ×
                  </button>
                </div>
                <p class="preview-text">{{ entry.preview }}</p>
                <div class="preview-tags">
                  <span v-if="entry.has_analysis" class="tag tag-analysis">分析</span>
                  <span v-if="entry.has_image" class="tag tag-image">图片</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 主内容区域 -->
    <div class="main-content" :class="{ 'sidebar-open': historyVisible && !sidebarCollapsed }">
      <!-- 详情内容显示在主内容区域 -->
      <Transition name="fade">
        <div v-if="selectedEntry" class="detail-content-main">
          <div class="detail-header-main">
            <h2>回忆详情 #{{ selectedEntry.id }}</h2>
            <button class="close-btn" @click="selectedEntry = null">×</button>
          </div>

          <div v-if="detailLoading" class="loading-state-3d">
            <div class="cube-loader">
              <div class="cube-face front"></div><div class="cube-face back"></div>
              <div class="cube-face right"></div><div class="cube-face left"></div>
              <div class="cube-face top"></div><div class="cube-face bottom"></div>
            </div>
            <p class="loading-text-glitch">正在加载详情...</p>
          </div>

          <div v-else class="detail-content-inner">
            <!-- 1. 梦境描述 -->
            <div class="detail-section">
              <h4>梦境描述</h4>
              <p>{{ selectedEntry.dream_text }}</p>
            </div>

            <!-- 2. 文本分析（情绪 / 主题 / 关键词） -->
            <div v-if="selectedEntry.text_analysis" class="detail-section">
              <h4>文本分析</h4>
              <div class="analysis-grid">
                <div v-if="selectedEntry.text_analysis.emotions" class="analysis-item">
                  <span class="analysis-label">情绪</span>
                  <span class="analysis-value">{{ (selectedEntry.text_analysis.emotions || []).join(' / ') }}</span>
                </div>
                <div v-if="selectedEntry.text_analysis.themes" class="analysis-item">
                  <span class="analysis-label">主题</span>
                  <span class="analysis-value">{{ (selectedEntry.text_analysis.themes || []).join(' / ') }}</span>
                </div>
                <div v-if="selectedEntry.text_analysis.keywords" class="analysis-item">
                  <span class="analysis-label">关键词</span>
                  <span class="analysis-value">{{ (selectedEntry.text_analysis.keywords || []).join(' · ') }}</span>
                </div>
              </div>
            </div>

            <!-- 3. 心理分析 -->
            <div v-if="selectedEntry.combined_analysis" class="detail-section">
              <h4>心理分析</h4>
              <p>{{ selectedEntry.combined_analysis }}</p>
            </div>

            <!-- 4. 视觉化建议 -->
            <div v-if="selectedEntry.visualization_prompt" class="detail-section">
              <h4>视觉化建议</h4>
              <p class="prompt-text">{{ selectedEntry.visualization_prompt }}</p>
            </div>

            <!-- 5. 生成的图片（放在最后，沿用主界面的倾斜样式） -->
            <div v-if="selectedEntry.image_url" class="detail-section detail-image-section">
              <h4>梦境重现</h4>
              <div class="image-portal-3d">
                <div class="image-wrapper-tilt">
                  <img :src="selectedEntry.image_url" alt="梦境图片" class="dream-result-img" />
                </div>
              </div>
            </div>

            <div class="detail-footer">
              <span class="detail-date">记录时间：{{ formatDate(selectedEntry.created_at) }}</span>
            </div>
          </div>
        </div>
      </Transition>

      <!-- 如果没有选择详情，显示主界面 -->
      <Transition name="fade">
        <div v-if="!selectedEntry" class="content-wrapper">
      
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
          <label class="holo-label" >输入梦境内容</label>
          
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

              <button class="cyber-btn cinema" :disabled="loading || loadingVideo" @click="generateVideo">
                <span class="btn-bg-anim"></span>
                <span class="btn-text">
                  {{ loadingVideo ? '时光编织中...' : '时光投影' }}
                </span>
              </button>

              <button class="cyber-btn settings-toggle" @click="videoSettingsExpanded = !videoSettingsExpanded">
                <span class="btn-text">{{ videoSettingsExpanded ? '▼ 视频参数' : '▶ 视频参数' }}</span>
              </button>

              <button class="cyber-btn history" @click="toggleSidebar">
                <span class="btn-text">查找往期回忆</span>
              </button>
            </div>

            <!-- 视频参数设置 -->
            <div v-if="videoSettingsExpanded" class="video-settings-panel glass-panel-3d">
              <div class="settings-content">
                <div class="setting-item">
                  <label>视频时长</label>
                  <div class="control-row">
                    <input v-model.number="videoDuration" type="range" min="1" max="60" class="slider" />
                    <span class="value-display">{{ videoDuration }} 秒</span>
                  </div>
                </div>
                <div class="setting-item">
                  <label>视频分辨率</label>
                  <select v-model="videoSize" class="resolution-select">
                    <option value="832*480">832×480 (默认)</option>
                    <option value="1024*576">1024×576 (HD)</option>
                    <option value="1280*720">1280×720 (720p)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
          
          <div v-if="error" class="system-alert">
            <span class="alert-icon">!</span> {{ error }}
          </div>
        </div>
      </div>

      <div v-if="result" class="result">
        <h2>梦境分析结果</h2>
        <div class="kv">
          <div>
            <div class="k">主要情绪</div>
            <div class="v">{{ (result.text_analysis?.emotions || []).join('，') || '平静' }}</div>
          </div>
          <div>
            <div class="k">梦境主题</div>
            <div class="v">{{ (result.text_analysis?.themes || []).join('，') || '—' }}</div>
          </div>
          <div>
            <div class="k">关键词</div>
            <div class="v">{{ (result.text_analysis?.keywords || []).join('，') || '—' }}</div>
          </div>
        </div>

        <div class="block">
          <div class="k">心理分析</div>
          <div class="v">{{ result.combined_analysis }}</div>
        </div>

        <div class="block">
          <div class="k">视觉化建议</div>
          <div class="v">{{ result.visualization_prompt }}</div>
        </div>
      </div>

      <div id="image-section" class="image-block" v-if="generatedImage || loadingImage">
        <h2>生成图像</h2>
        <div class="image-portal-3d">
          <div v-if="loadingImage" class="img-placeholder">生成中，请稍候...</div>
          <div v-else-if="generatedImage" class="image-wrapper-tilt">
            <img :src="generatedImage" class="dream-result-img" />
          </div>
          <div v-else class="img-placeholder">暂无图像</div>
        </div>
      </div>

      <div id="video-section" class="video-block" v-if="generatedVideo || loadingVideo">
        <h2>视频生成</h2>
        <div v-if="loadingVideo" class="loading-state-3d">
          <div class="cube-loader">
            <div class="cube-face front"></div><div class="cube-face back"></div>
            <div class="cube-face right"></div><div class="cube-face left"></div>
            <div class="cube-face top"></div><div class="cube-face bottom"></div>
          </div>
          <p class="loading-text-glitch">正在编织时光...</p>
        </div>
        <div v-else-if="generatedVideo" class="video-wrapper">
          <video controls class="video-player">
            <source :src="generatedVideo.url" type="video/mp4" />
            您的浏览器不支持视频播放
          </video>
          <p v-if="generatedVideo.message" class="video-message">{{ generatedVideo.message }}</p>
        </div>
      </div>

      <div id="comprehensive-section" class="comprehensive-block" v-if="comprehensiveAnalysis">
        <h2>综合分析结果</h2>
        <div class="comprehensive-content">
          <p>{{ comprehensiveAnalysis }}</p>
        </div>
        <button @click="clearSelection" class="cyber-btn secondary">清除选择</button>
      </div>

        </div>
      </Transition>

      <!-- 侧边栏 -->
      <div v-if="historyVisible" class="sidebar-container" :class="{ 'collapsed': sidebarCollapsed }">
        <div class="sidebar-header">
          <h3>梦境历史记录</h3>
          <div style="display: flex; gap: 8px;">
            <button class="close-btn" style="width: 28px; height: 28px; font-size: 1.2rem;" @click="toggleSidebarCollapse" :title="sidebarCollapsed ? '展开' : '收起'">
              {{ sidebarCollapsed ? '◀' : '▶' }}
            </button>
            <button class="close-btn" @click="closeHistory">×</button>
          </div>
        </div>

        <div class="sidebar-content">
          <div v-if="historyLoading" class="loading-state-3d">
            <div class="cube-loader">
              <div class="cube-face front"></div><div class="cube-face back"></div>
              <div class="cube-face right"></div><div class="cube-face left"></div>
              <div class="cube-face top"></div><div class="cube-face bottom"></div>
            </div>
            <p class="loading-text-glitch">正在检索记忆库...</p>
          </div>

          <div v-else-if="filteredHistoryEntries.length === 0" class="empty-history">
            <p>记忆库中暂无记录</p>
          </div>

          <div v-else class="history-list-sidebar">
            <div class="selection-actions" v-if="selectedEntryIds.size > 0">
              <p class="selection-info">已选择 {{ selectedEntryIds.size }} 条记录</p>
              <button class="cyber-btn small" @click="analyzeComprehensive" :disabled="analyzing">
                {{ analyzing ? '分析中...' : '综合分析' }}
              </button>
            </div>

            <div 
              v-for="entry in filteredHistoryEntries" 
              :key="entry.id" 
              class="history-item-sidebar"
              :class="{ 'active': selectedEntry && selectedEntry.id === entry.id }"
              @click="loadEntryDetail(entry.id)"
            >
              <div class="history-item-preview">
                <div class="preview-header">
                  <input 
                    type="checkbox" 
                    class="entry-checkbox"
                    :checked="selectedEntryIds.has(entry.id)"
                    @click.stop="toggleEntrySelection(entry.id, $event)"
                    @change="() => {}"
                  />
                  <span class="history-id">#{{ entry.id }}</span>
                  <span class="history-date-small">{{ formatDate(entry.created_at) }}</span>
                  <button 
                    class="delete-btn" 
                    @click.stop="deleteEntry(entry.id, $event)"
                    title="删除（仅前端）"
                  >
                    ×
                  </button>
                </div>
                <p class="preview-text">{{ entry.preview }}</p>
                <div class="preview-tags">
                  <span v-if="entry.has_analysis" class="tag tag-analysis">分析</span>
                  <span v-if="entry.has_image" class="tag tag-image">图片</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <canvas ref="canvasRef" style="position: fixed; top: 0; left: 0; z-index: 0;"></canvas>
  
</template>

<style scoped>
:root {
  --neon-purple: #a855f7;
  --neon-pink: #ec4899;
  --neon-cyan: #06b6d4;
  --glass-border: rgba(168, 85, 247, 0.2);
}

.page {
  position: relative;
  min-height: 100vh;
  background: radial-gradient(1200px 600px at 20% 10%, rgba(140, 120, 255, .25), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(255, 150, 200, .2), transparent 60%),
              radial-gradient(1000px 800px at 50% 100%, rgba(60, 180, 255, .15), transparent 60%),
              #0a0f1f;
  color: #eef2ff;
  overflow: hidden;
  display: flex;
}
.nebula {
  position: absolute;
  inset: -10% -10% -10% -10%;
  background: center/cover no-repeat;
  pointer-events: none;
}
.container {
  position: relative;
  max-width: 1560px;
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
  background-clip: text;
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

.input-field-wrap { position: relative; margin-top:20px;margin-bottom: 30px; transform-style: preserve-3d;}

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

.cyber-btn.cinema { border-color: #3b82f6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); }
.cyber-btn.cinema:hover { background: rgba(59, 130, 246, 0.2); box-shadow: 0 0 40px rgba(59, 130, 246, 0.6); }

.cyber-btn.secondary:hover { border-color: #fff; background: rgba(255,255,255,0.1); }

.cyber-btn.history { border-color: var(--neon-cyan); box-shadow: 0 0 15px rgba(6, 182, 212, 0.2); }
.cyber-btn.history:hover { background: rgba(6, 182, 212, 0.2); box-shadow: 0 0 40px rgba(6, 182, 212, 0.6); }

.cyber-btn.small { padding: 10px 20px; font-size: 0.8rem; clip-path: none; border-radius: 50px; border-color: #64748b;}
.cyber-btn.small.active { border-color: var(--neon-cyan); background: rgba(6, 182, 212, 0.2); box-shadow: 0 0 20px rgba(6, 182, 212, 0.4); }

.cyber-btn.settings-toggle { 
    padding: 8px 16px; 
    font-size: 0.85rem;
    border-color: #8b5cf6;
    background: rgba(139, 92, 246, 0.1);
}
.cyber-btn.settings-toggle:hover {
    background: rgba(139, 92, 246, 0.2);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
}

.video-settings-panel {
    margin-top: 20px;
    padding: 20px;
    border-radius: 8px;
    animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.settings-content {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.setting-item {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.glass {
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.1);
  backdrop-filter: blur(14px);
}
.title {
   font-size: 52px;
  font-weight: 800;
  letter-spacing: 1px;
  margin: 0 auto 6px; 
  text-align: center;

  background: linear-gradient(135deg, #5050f3, #3f32f3, #ffd6f9, #a8e0ff);
  background-size: 300% 300%;
  animation: flow 15s ease infinite;

  -webkit-background-clip: text;
  background-clip: text;

  color: transparent; /* ✅ 必须加这一句 */
  -webkit-text-fill-color: transparent; /* ✅ 兼容 Safari / Chrome */
  
}
.subtitle { margin: 0 auto 28px; opacity: .85; font-size: 20px; text-align: center;}

.form { display: grid; gap: 14px; }
.label { opacity: .9; font-weight: 600; font-size: 18px;margin-left: 20px;}
.input {
  width: 1500px;
  margin: 0 auto;
  resize: vertical;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.18);
  color:white;
  font-size: 15px;
}
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.file { display: none; }
.file-btn {
  padding: 10px 14px;
  border-radius: 12px;
  background: linear-gradient(135deg, #a78bfa, #f0abfc);
  color: white;
  font-weight: 700;
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
  transition: background 0.2s;
  flex-shrink: 0;
}

.sidebar-toggle:hover {
  background: rgba(255,255,255,0.1);
}

.sidebar-header h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 0;
  color: #fff;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar-container.collapsed .sidebar-header h3,
.sidebar-container.collapsed .new-chat-btn {
  display: none;
}

.new-chat-btn {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  color: #fff;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: background 0.2s;
  white-space: nowrap;
}

.new-chat-btn:hover {
  background: rgba(255,255,255,0.15);
}

.sidebar-content {
  flex: 1;
  overflow: hidden;
  padding: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.sidebar-content::-webkit-scrollbar {
  width: 6px;
}

.sidebar-content::-webkit-scrollbar-track {
  background: transparent;
}

.sidebar-content::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.2);
  border-radius: 3px;
}

.sidebar-content::-webkit-scrollbar-thumb:hover {
  background: rgba(255,255,255,0.3);
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-header h3 {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.2rem;
  margin: 0;
  color: #fff;
  text-shadow: 0 0 10px var(--neon-cyan);
}

.close-btn {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.2);
  color: #fff;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 1.5rem;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: 0.3s;
  flex-shrink: 0;
}

.close-btn:hover {
  background: rgba(236, 72, 153, 0.3);
  border-color: var(--neon-pink);
  transform: rotate(90deg);
}

.empty-history {
  text-align: center;
  padding: 60px 20px;
  color: #8e8ea0;
  font-size: 13px;
  flex-shrink: 0;
}

.history-list-sidebar {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 10px;
  min-height: 0;
  /* 确保列表区域可以独立滚动，不影响侧边栏固定位置 */
  -webkit-overflow-scrolling: touch;
}

.history-item-sidebar {
  background: transparent;
  border: none;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.history-item-sidebar:hover {
  background: rgba(255,255,255,0.1);
}

.history-item-sidebar.active {
  background: rgba(168, 85, 247, 0.2);
}

.history-item-preview {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  position: relative;
}

.history-id {
  color: #8e8ea0;
  font-weight: 600;
  font-size: 12px;
  flex-shrink: 0;
}

.history-date-small {
  color: #8e8ea0;
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preview-text {
  color: #ececf1;
  font-size: 13px;
  line-height: 1.5;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 4px;
}

.tag {
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
}

.tag-analysis {
  background: rgba(168, 85, 247, 0.15);
  color: var(--neon-purple);
}

.tag-image {
  background: rgba(236, 72, 153, 0.15);
  color: var(--neon-pink);
}

.delete-btn {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #ef4444;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
  padding: 0;
  margin: 0;
  opacity: 0;
}

.history-item-sidebar:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  background: rgba(239, 68, 68, 0.2);
  border-color: #ef4444;
  transform: scale(1.1);
}

.delete-btn {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #ef4444;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
  padding: 0;
  margin: 0;
  opacity: 0;
}

.history-item-sidebar:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  background: rgba(239, 68, 68, 0.2);
  border-color: #ef4444;
  transform: scale(1.1);
}

/* 主内容区域 */
.main-content {
  flex: 1;
  margin-left: 0;
  transition: margin-left 0.3s ease;
  min-height: 100vh;
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  position: relative;
  /* 确保主内容区域的滚动不影响侧边栏 */
  z-index: 1;
}

.main-content.sidebar-open {
  margin-left: 260px;
  width: calc(100% - 260px);
}

.sidebar-container.collapsed + .main-content.sidebar-open {
  margin-left: 60px;
  width: calc(100% - 60px);
}

/* 详情内容在主内容区域 */
.detail-content-main {
  max-width: 900px;
  margin: 0 auto;
  padding: 40px 20px;
}

.detail-header-main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.detail-header-main h2 {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.8rem;
  margin: 0;
  color: #fff;
  text-shadow: 0 0 10px var(--neon-purple);
}

.detail-content-inner {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.detail-header h3 {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.2rem;
  margin: 0;
  color: #fff;
  text-shadow: 0 0 10px var(--neon-purple);
}

.detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.detail-section {
  margin-bottom: 25px;
  padding-bottom: 20px;
  border-bottom: 1px dashed rgba(255,255,255,0.1);
}

.detail-section:last-child {
  border-bottom: none;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.detail-section h4 {
  font-family: 'Orbitron', sans-serif;
  color: var(--neon-cyan);
  font-size: 1rem;
  margin: 0 0 15px 0;
  letter-spacing: 1px;
  text-shadow: 0 0 8px var(--neon-cyan);
}

.detail-section p {
  color: #e2e8f0;
  line-height: 1.8;
  margin: 0;
  text-align: justify;
}

.analysis-grid {
  display: grid;
  gap: 15px;
}

.analysis-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 12px;
  background: rgba(0,0,0,0.3);
  border-left: 3px solid var(--neon-purple);
  border-radius: 8px;
}

.analysis-label {
  font-family: 'Orbitron', sans-serif;
  color: #94a3b8;
  font-size: 0.8rem;
  letter-spacing: 1px;
}

.analysis-value {
  color: #e2e8f0;
  font-size: 0.95rem;
}

.prompt-text {
  font-family: 'Courier New', monospace;
  color: #a5f3fc;
  background: rgba(6, 182, 212, 0.05);
  padding: 15px;
  border-radius: 8px;
  border-left: 3px solid var(--neon-cyan);
}

.detail-image-section {
  border-bottom: 2px solid rgba(168, 85, 247, 0.3);
  padding-bottom: 25px;
  margin-bottom: 25px;
}

.detail-image-section h4 {
  color: var(--neon-pink);
  text-shadow: 0 0 10px var(--neon-pink);
  font-size: 1.1rem;
  margin-bottom: 20px;
}

.detail-image-wrapper {
  margin-top: 15px;
  border-radius: 12px;
  color: white;
  font-weight: 700;
  letter-spacing: .5px;
  border: none;
  cursor: pointer;
}
.btn.primary { background: linear-gradient(135deg, #60a5fa, #a78bfa); }
.btn.secondary { background: linear-gradient(135deg, #fb7185, #f472b6); }
.btn:disabled { opacity: .6; cursor: not-allowed; }
.error { color: #fecaca; }

.result { margin-top: 28px; display: grid; gap: 18px; }
.kv { display: grid; gap: 10px; grid-template-columns: repeat(3, 1fr); }
.k { opacity: .85; font-weight: 700; margin-bottom: 6px; }
.v { opacity: .95; }
.block { padding-top: 8px; border-top: 1px dashed rgba(255,255,255,.2); }

.content-wrapper {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 40px 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.sidebar-container {
  position: fixed;
  right: 0;
  top: 0;
  height: 100vh;
  width: 350px;
  background: linear-gradient(180deg, rgba(20, 20, 50, 0.95), rgba(10, 10, 30, 0.95));
  border-left: 1px solid var(--glass-border);
  backdrop-filter: blur(20px);
  display: flex;
  flex-direction: column;
  z-index: 100;
  box-shadow: -20px 0 60px rgba(0,0,0,0.8);
  transition: width 0.3s ease;
}

.sidebar-container.collapsed {
  width: 60px;
}

.entry-checkbox {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: var(--neon-cyan);
}

.selection-actions {
  padding: 15px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  background: rgba(168, 85, 247, 0.1);
}

.selection-info {
  color: #a8e0ff;
  font-size: 12px;
  margin: 0 0 10px 0;
}

.image-portal-3d {
  margin-top: 20px;
  perspective: 1000px;
}

.image-wrapper-tilt {
  transform-style: preserve-3d;
  transition: transform 0.6s ease;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(168, 85, 247, 0.2), inset 0 1px 0 rgba(255,255,255,0.1);
}

.image-wrapper-tilt:hover {
  transform: rotateX(5deg) rotateY(-5deg);
}

.dream-result-img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 12px;
}

.video-wrapper {
  margin-top: 20px;
  border-radius: 12px;
  overflow: hidden;
}

.video-player {
  width: 100%;
  height: auto;
  border-radius: 12px;
}

.video-message {
  color: #a5f3fc;
  font-size: 0.9rem;
  margin-top: 10px;
  text-align: center;
}

.comprehensive-block {
  margin-top: 40px;
  padding: 30px;
  background: rgba(168, 85, 247, 0.1);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
}

.comprehensive-content {
  color: #e2e8f0;
  line-height: 1.8;
  margin-bottom: 20px;
  white-space: pre-wrap;
}

.system-alert {
  margin-top: 20px;
  padding: 15px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-left: 3px solid #ef4444;
  border-radius: 8px;
  color: #fecaca;
  display: flex;
  align-items: center;
  gap: 10px;
}

.alert-icon {
  font-size: 1.2rem;
  font-weight: bold;
}

.image-block,
.video-block,
.comprehensive-block {
  margin-top: 40px;
  padding: 30px;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  backdrop-filter: blur(20px);
}

.loading-state-3d {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  gap: 20px;
}

.cube-loader {
  position: relative;
  width: 60px;
  height: 60px;
  transform-style: preserve-3d;
  animation: cubeRotate 3s infinite linear;
}

@keyframes cubeRotate {
  0% { transform: rotateX(0deg) rotateY(0deg); }
  100% { transform: rotateX(360deg) rotateY(360deg); }
}

.cube-face {
  position: absolute;
  width: 60px;
  height: 60px;
  background: rgba(168, 85, 247, 0.3);
  border: 2px solid var(--neon-purple);
  opacity: 0.7;
}

.cube-face.front { transform: translateZ(30px); }
.cube-face.back { transform: translateZ(-30px) rotateY(180deg); }
.cube-face.right { transform: rotateY(90deg) translateZ(30px); }
.cube-face.left { transform: rotateY(-90deg) translateZ(30px); }
.cube-face.top { transform: rotateX(90deg) translateZ(30px); }
.cube-face.bottom { transform: rotateX(-90deg) translateZ(30px); }

.loading-text-glitch {
  color: var(--neon-cyan);
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 2px;
  text-shadow: 0 0 10px var(--neon-cyan);
  margin: 0;
  animation: glitch 2s ease-in-out infinite;
}

@keyframes glitch {
  0%, 100% { text-shadow: 0 0 10px var(--neon-cyan); }
  50% { text-shadow: 0 0 20px var(--neon-purple); }
}

.img-wrap {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.img-placeholder {
  padding: 40px;
  text-align: center;
  color: #8e8ea0;
  font-size: 1.1rem;
}

.generated-img {
  max-width: 100%;
  height: auto;
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(168, 85, 247, 0.2);
}

@keyframes flow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
</style>
