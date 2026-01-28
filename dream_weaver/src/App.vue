
<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'

// --- 业务逻辑状态 ---
const dreamText = ref('')
const imageFile = ref(null)
const loading = ref(false)
const loadingImage = ref(false)
const result = ref(null)
const generatedImage = ref(null)
const error = ref('')
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

// --- 语音输入状态 ---
const isRecording = ref(false) // 是否正在录音
const speechSupported = ref(false) // 浏览器是否支持语音识别
let recognition = null // 语音识别对象

// --- 编辑模式状态 ---
const isEditing = ref(false) // 是否处于编辑模式
const editingText = ref('') // 编辑中的文本内容
const isUpdating = ref(false) // 是否正在更新

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
    lastEntryId.value = result.value?.entry_id || null
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
    lastEntryId.value = result.value?.entry_id || null
    scrollTo('result-section')
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
  }
}

async function generateImage(entryId = null, dreamTextValue = null) {
  // 如果从详情页调用，使用传入的参数；否则使用主界面的输入
  const textToUse = dreamTextValue || dreamText.value
  const entryIdToUse = entryId || lastEntryId.value
  
  if (!textToUse && !entryIdToUse) {
    if (!entryId) {
      // 主界面调用，需要检查输入
  if (checkEmpty()) return
    }
    error.value = '请先输入梦境内容或选择一条记录'
    return
  }
  
  error.value = ''
  loadingImage.value = true
  
  // 如果是从详情页调用，不重置 generatedImage
  if (!entryId) {
    generatedImage.value = null
  }
  
  const form = new FormData()
  form.append('dream_text', textToUse)
  if (entryIdToUse) {
    form.append('entry_id', String(entryIdToUse))
  }
  
  try {
    const resp = await fetch('http://localhost:8000/generate-image', { method: 'POST', body: form })
    if (!resp.ok) throw new Error('图像具象化失败')
    const j = await resp.json()
    if (j && j.image) {
      // 如果是从详情页生成的，更新详情页的图片
      if (entryIdToUse && selectedEntry.value && selectedEntry.value.id === entryIdToUse) {
        selectedEntry.value.image_url = j.image
        // 重新加载详情以获取完整数据
        await loadEntryDetail(entryIdToUse)
      } else {
        // 主界面生成图片
      generatedImage.value = j.image
        scrollTo('image-section')
      }
      // 如果后端返回了 entry_id，则更新 lastEntryId
      if (j.entry_id) {
        lastEntryId.value = j.entry_id
      }
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

async function loadHistory(clearSelection = false) {
  historyLoading.value = true
  historyVisible.value = true
  // 只有在明确要求清除选择时才清空选中的详情
  if (clearSelection) {
  selectedEntry.value = null
  }
  try {
    const resp = await fetch('http://localhost:8000/dreams/history?limit=20')
    if (!resp.ok) throw new Error('获取历史记录失败')
    const data = await resp.json()
    if (data.success) {
      historyEntries.value = data.entries || []
      // 如果当前有选中的详情，更新它（如果列表中有对应的记录）
      if (selectedEntry.value && !clearSelection) {
        const updatedEntry = data.entries.find(e => e.id === selectedEntry.value.id)
        if (updatedEntry) {
          // 只更新预览文本，不重新加载完整详情（避免闪烁）
          // selectedEntry 的完整数据会在需要时重新加载
        }
      }
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
  // 退出编辑模式（如果正在编辑）
  if (isEditing.value) {
    cancelEdit()
  }
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

// 进入编辑模式
function startEdit() {
  if (!selectedEntry.value) return
  isEditing.value = true
  editingText.value = selectedEntry.value.dream_text || ''
  error.value = ''
}

// 取消编辑
function cancelEdit() {
  isEditing.value = false
  editingText.value = ''
  error.value = ''
}

// 保存编辑
async function saveEdit() {
  if (!selectedEntry.value) return
  
  const newText = editingText.value.trim()
  if (!newText) {
    error.value = '梦境内容不能为空'
    return
  }
  
  // 检查是否有变化
  if (newText === selectedEntry.value.dream_text) {
    error.value = '内容未修改'
    return
  }
  
  isUpdating.value = true
  error.value = ''
  
  try {
    const resp = await fetch(`http://localhost:8000/dreams/${selectedEntry.value.id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dream_text: newText
      })
    })
    
    // 先尝试解析响应
    let data
    try {
      data = await resp.json()
    } catch (parseError) {
      // 如果无法解析JSON，可能是网络错误或服务器错误
      if (!resp.ok) {
        throw new Error(`服务器错误 (${resp.status}): ${resp.statusText}`)
      }
      throw new Error('无法解析服务器响应')
    }
    
    if (!resp.ok) {
      throw new Error(data.error || `更新失败 (状态码: ${resp.status})`)
    }
    
    if (data.success) {
      // 更新 selectedEntry（保持停留在详情页）
      selectedEntry.value = data.entry
      // 退出编辑模式
      isEditing.value = false
      editingText.value = ''
      // 更新 lastEntryId，以便后续生成图片时使用
      lastEntryId.value = selectedEntry.value.id
      // 刷新历史记录列表（不清空当前选中的详情）
      if (historyVisible.value) {
        loadHistory(false)  // 传入 false，不清空 selectedEntry
      }
      // 显示成功提示（可选）
      console.log('梦境记录已更新')
    } else {
      throw new Error(data.error || '更新失败')
    }
  } catch (e) {
    // 更详细的错误处理
    if (e.name === 'TypeError' && e.message.includes('fetch')) {
      error.value = '无法连接到后端服务器，请确保后端服务正在运行 (http://localhost:8000)'
    } else {
      error.value = e.message || String(e)
    }
    console.error('更新失败:', e)
  } finally {
    isUpdating.value = false
  }
}

function toggleSidebar() {
  historyVisible.value = !historyVisible.value
  if (!historyVisible.value) {
    selectedEntry.value = null
  } else if (historyEntries.value.length === 0) {
    loadHistory(true)  // 打开侧边栏时，清空选中的详情
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

<<<<<<< HEAD
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
<<<<<<< HEAD
=======
=======
>>>>>>> b8febb229b35541fc801139fe81e0b2667554f61

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
// ======================
// 🎤 语音输入模块（浏览器原生 Speech Recognition）
// ======================

<<<<<<< HEAD
const isRecording = ref(false)
const speechSupported = ref(true)

let recognition = null

onMounted(() => {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition

=======
onMounted(() => {
  // 检查浏览器是否支持语音识别
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
  
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
  if (!SpeechRecognition) {
    speechSupported.value = false
    console.warn("当前浏览器不支持语音识别")
    return
  }
<<<<<<< HEAD

=======
  
  speechSupported.value = true
  
  // 初始化语音识别
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
  recognition = new SpeechRecognition()
  recognition.lang = "zh-CN"          // 中文识别
  recognition.continuous = true       // 连续识别
  recognition.interimResults = false  // 只要最终结果
<<<<<<< HEAD

  recognition.onstart = () => {
    isRecording.value = true
  }

  recognition.onend = () => {
    isRecording.value = false
  }

  recognition.onerror = (e) => {
    console.error("语音识别错误：", e)
    isRecording.value = false
  }

=======
  
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
  
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
  recognition.onresult = (event) => {
    let transcript = ""
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript
    }
<<<<<<< HEAD

    // 🔑 核心：语音内容写入 dreamText
    dreamText.value += (dreamText.value ? " " : "") + transcript

    // 🔮 未来 AI 追问的钩子（现在不启用）
    // onSpeechSegment(transcript)
  }
})

// 开始 / 停止录音
function toggleRecording() {
  if (!recognition) return

  if (isRecording.value) {
    recognition.stop()
  } else {
    recognition.start()
  }
}

=======
    
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
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
</script>

<template>
  <div class="dream-universe-ui">
    <!-- 侧边栏 -->
    <div class="sidebar-container" :class="{ 'collapsed': sidebarCollapsed, 'visible': historyVisible }">
      <div class="sidebar">
        <div class="sidebar-header">
          <button class="sidebar-toggle" @click="toggleSidebarCollapse" v-if="historyVisible">
            <span v-if="!sidebarCollapsed">☰</span>
            <span v-else>☰</span>
          </button>
          <h3 v-if="!sidebarCollapsed || !historyVisible">往期回忆</h3>
          <button class="new-chat-btn" @click="toggleSidebar" v-if="!sidebarCollapsed || !historyVisible">
            {{ historyVisible ? '关闭' : '打开' }}
          </button>
        </div>

        <div v-if="historyVisible && !sidebarCollapsed" class="sidebar-content">
          <!-- 综合分析控制栏 -->
          <div v-if="filteredHistoryEntries.length > 0" class="comprehensive-controls">
            <div class="selection-info">
              <span>已选择 {{ selectedEntryIds.size }} 条记录</span>
            </div>
            <div class="control-buttons">
              <button 
                class="analyze-btn" 
                :disabled="selectedEntryIds.size === 0 || analyzing"
                @click="analyzeComprehensive"
              >
                {{ analyzing ? '分析中...' : '综合分析' }}
              </button>
              <button 
                class="clear-btn" 
                :disabled="selectedEntryIds.size === 0"
                @click="clearSelection"
              >
                清空
              </button>
            </div>
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
<<<<<<< HEAD
            <h2>回忆详情 #{{ selectedEntry.id }}</h2>
            <button class="close-btn" @click="selectedEntry = null">×</button>
=======
            <h2>
              {{ isEditing ? '编辑模式' : '回忆详情' }} #{{ selectedEntry.id }}
            </h2>
            <div class="header-actions">
              <button 
                v-if="!isEditing" 
                class="edit-btn" 
                @click="startEdit"
                :disabled="detailLoading"
              >
                ✏️ 编辑
              </button>
              <button 
                v-if="isEditing" 
                class="save-btn" 
                @click="saveEdit"
                :disabled="isUpdating"
              >
                {{ isUpdating ? '保存中...' : '💾 保存并重新分析' }}
              </button>
              <button 
                v-if="isEditing" 
                class="cancel-btn" 
                @click="cancelEdit"
                :disabled="isUpdating"
              >
                取消
              </button>
              <button class="close-btn" @click="selectedEntry = null; cancelEdit()">×</button>
            </div>
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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
<<<<<<< HEAD
            <div class="detail-section">
              <h4>梦境描述</h4>
              <p>{{ selectedEntry.dream_text }}</p>
            </div>
=======
            <div class="detail-section" :class="{ 'editing': isEditing }">
              <h4>梦境描述</h4>
              <div v-if="isEditing" class="edit-textarea-wrapper">
                <textarea 
                  v-model="editingText" 
                  class="edit-textarea"
                  rows="6"
                  placeholder="输入梦境内容..."
                ></textarea>
                <div v-if="error" class="error-message">{{ error }}</div>
              </div>
              <p v-else>{{ selectedEntry.dream_text }}</p>
            </div>
            
            <!-- 编辑模式提示 -->
            <div v-if="isEditing" class="edit-mode-notice">
              <p>⚠️ 编辑后将重新分析梦境内容，其他分析结果将更新</p>
            </div>
            
            <!-- 2-5. 其他分析结果（编辑模式下暂时隐藏或显示提示） -->
            <template v-if="!isEditing">
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b

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
<<<<<<< HEAD
            <div v-if="selectedEntry.image_url" class="detail-section detail-image-section">
              <h4>梦境重现</h4>
              <div class="image-portal-3d">
=======
            <div class="detail-section detail-image-section">
              <div class="image-section-header">
              <h4>梦境重现</h4>
                <button 
                  class="regenerate-image-btn" 
                  @click="generateImage(selectedEntry.id, selectedEntry.dream_text)"
                  :disabled="loadingImage"
                >
                  {{ loadingImage ? '生成中...' : '🔄 重新生成图片' }}
                </button>
              </div>
              <div v-if="loadingImage" class="image-loading">
                <div class="cube-loader">
                  <div class="cube-face front"></div><div class="cube-face back"></div>
                  <div class="cube-face right"></div><div class="cube-face left"></div>
                  <div class="cube-face top"></div><div class="cube-face bottom"></div>
                </div>
                <p class="loading-text-glitch">正在生成图片...</p>
              </div>
              <div v-else-if="selectedEntry.image_url" class="image-portal-3d">
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
                <div class="image-wrapper-tilt">
                  <img :src="selectedEntry.image_url" alt="梦境图片" class="dream-result-img" />
                </div>
              </div>
<<<<<<< HEAD
            </div>

=======
              <div v-else class="no-image-placeholder">
                <p>暂无图片，点击"重新生成图片"按钮生成</p>
              </div>
            </div>

            </template>

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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
          
<<<<<<< HEAD
          <div class="dream-input-wrapper">
            <textarea
              class="hologram-input"
              v-model="dreamText"
              placeholder="请描述你的梦境..."
            ></textarea>

            <!-- 🎤 语音输入按钮 -->
            <button
              v-if="speechSupported"
              class="voice-btn"
              :class="{ recording: isRecording }"
              @click="toggleRecording"
            >
              {{ isRecording ? "🎙 正在聆听…" : "🎤 语音输入" }}
            </button>

            <div v-else class="voice-tip">
=======
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
            <!-- 🎤 语音输入按钮 -->
            <button
              v-if="speechSupported"
              class="voice-input-btn"
              :class="{ 'recording': isRecording }"
              @click="toggleRecording"
              type="button"
            >
              <span class="voice-icon">{{ isRecording ? '🎙' : '🎤' }}</span>
              <span class="voice-text">{{ isRecording ? '正在聆听...' : '语音输入' }}</span>
            </button>
            <div v-else class="voice-unsupported-tip">
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
              ⚠️ 当前浏览器不支持语音输入
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

<<<<<<< HEAD
=======
              <button class="cyber-btn cinema" :disabled="loading || loadingVideo" @click="generateVideo">
                <span class="btn-bg-anim"></span>
                <span class="btn-text">
                  {{ loadingVideo ? '时光编织中...' : '时光投影' }}
                </span>
              </button>

              <button class="cyber-btn settings-toggle" @click="videoSettingsExpanded = !videoSettingsExpanded">
                <span class="btn-text">{{ videoSettingsExpanded ? '▼ 视频参数' : '▶ 视频参数' }}</span>
              </button>

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
              <button class="cyber-btn history" @click="toggleSidebar">
                <span class="btn-text">查找往期回忆</span>
              </button>
            </div>
<<<<<<< HEAD
=======

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
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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

<<<<<<< HEAD
=======
      <!-- 视频生成结果 -->
      <Transition name="hologram-reveal">
        <div v-if="generatedVideo || loadingVideo" id="video-section" class="cinema-deck glass-panel-3d">
          <div class="deck-header">
            <h3>时光投影</h3>
            <div class="scanner-line-anim"></div>
          </div>
          
          <div class="video-portal-3d">
            <div v-if="loadingVideo" class="loading-state-3d">
              <div class="cube-loader">
                <div class="cube-face front"></div><div class="cube-face back"></div>
                <div class="cube-face right"></div><div class="cube-face left"></div>
                <div class="cube-face top"></div><div class="cube-face bottom"></div>
              </div>
              <p class="loading-text-glitch">时光编织中，请耐心等待...</p>
            </div>
            <div v-else class="video-wrapper">
              <video 
                :src="generatedVideo.url" 
                class="dream-result-video"
                controls
                autoplay
                loop
                crossorigin="anonymous"
              ></video>
              <div class="video-info">
                <p class="info-text">{{ generatedVideo.message }}</p>
              </div>
            </div>
          </div>
        </div>
      </Transition>

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
      <!-- 综合分析结果 -->
      <Transition name="hologram-reveal">
        <div v-if="comprehensiveAnalysis" id="comprehensive-section" class="comprehensive-deck glass-panel-3d">
          <div class="deck-header">
            <h3>综合分析报告</h3>
            <div class="scanner-line-anim"></div>
            <button class="close-btn" @click="comprehensiveAnalysis = null">×</button>
          </div>

          <div class="comprehensive-content">
            <!-- 状态评分卡片 -->
            <div class="score-cards">
              <div class="score-card overall">
                <div class="score-label">综合状态</div>
                <div class="score-value">{{ comprehensiveAnalysis.overall_score }}/100</div>
                <div class="score-bar">
                  <div class="score-fill" :style="{ width: comprehensiveAnalysis.overall_score + '%' }"></div>
                </div>
              </div>
              <div class="score-card sleep">
                <div class="score-label">睡眠质量</div>
                <div class="score-value">{{ comprehensiveAnalysis.sleep_quality }}/100</div>
                <div class="score-bar">
                  <div class="score-fill sleep-fill" :style="{ width: comprehensiveAnalysis.sleep_quality + '%' }"></div>
                </div>
              </div>
              <div class="score-card emotion">
                <div class="score-label">情绪状态</div>
                <div class="score-value">{{ comprehensiveAnalysis.emotion_score }}/100</div>
                <div class="score-bar">
                  <div class="score-fill emotion-fill" :style="{ width: comprehensiveAnalysis.emotion_score + '%' }"></div>
                </div>
              </div>
            </div>

            <!-- 详细分析 -->
            <div class="analysis-sections">
              <div class="analysis-section">
                <h4>状态总结</h4>
                <p>{{ comprehensiveAnalysis.summary }}</p>
              </div>

              <div class="analysis-section">
                <h4>情绪分析</h4>
                <div class="emotion-tags">
                  <span 
                    v-for="(score, emotion) in comprehensiveAnalysis.emotion_breakdown" 
                    :key="emotion"
                    class="emotion-tag"
                    :style="{ opacity: Math.max(0.3, score / 100) }"
                  >
                    {{ emotion }} ({{ score }}%)
                  </span>
                </div>
              </div>

              <div class="analysis-section">
                <h4>睡眠质量评估</h4>
                <p>{{ comprehensiveAnalysis.sleep_analysis }}</p>
              </div>

              <div class="analysis-section">
                <h4>建议与提醒</h4>
                <ul class="suggestions-list">
                  <li v-for="(suggestion, index) in comprehensiveAnalysis.suggestions" :key="index">
                    {{ suggestion }}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </Transition>
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
  display: flex;
  perspective: 1000px;
  width: 100%;
  /* 确保不会影响固定定位的子元素 */
  overflow: visible;
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

<<<<<<< HEAD
.input-field-wrap { position: relative; margin-top:20px;margin-bottom: 30px; transform-style: preserve-3d;}
=======
.input-field-wrap { 
  position: relative; 
  margin-top:20px;
  margin-bottom: 30px; 
  transform-style: preserve-3d;
}
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b

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

<<<<<<< HEAD
=======
/* 语音输入按钮样式 */
.voice-input-btn {
  position: absolute;
  bottom: 12px;
  right: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(6, 182, 212, 0.15);
  border: 1px solid rgba(6, 182, 212, 0.4);
  border-radius: 20px;
  color: var(--neon-cyan);
  font-family: 'Rajdhani', sans-serif;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 10;
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.voice-input-btn:hover {
  background: rgba(6, 182, 212, 0.25);
  border-color: var(--neon-cyan);
  box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
  transform: translateY(-2px);
}

.voice-input-btn.recording {
  background: rgba(236, 72, 153, 0.2);
  border-color: var(--neon-pink);
  color: var(--neon-pink);
  animation: voicePulse 1.5s ease-in-out infinite;
}

.voice-input-btn.recording:hover {
  background: rgba(236, 72, 153, 0.3);
  box-shadow: 0 0 25px rgba(236, 72, 153, 0.6);
}

@keyframes voicePulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(236, 72, 153, 0.7);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(236, 72, 153, 0);
  }
}

.voice-icon {
  font-size: 16px;
  line-height: 1;
  filter: drop-shadow(0 0 4px currentColor);
}

.voice-text {
  letter-spacing: 1px;
  text-shadow: 0 0 8px currentColor;
}

.voice-unsupported-tip {
  position: absolute;
  bottom: 12px;
  right: 12px;
  padding: 6px 12px;
  background: rgba(100, 100, 100, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  color: #8e8ea0;
  font-size: 11px;
  z-index: 10;
}

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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

<<<<<<< HEAD
=======
.cyber-btn.cinema { border-color: #3b82f6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); }
.cyber-btn.cinema:hover { background: rgba(59, 130, 246, 0.2); box-shadow: 0 0 40px rgba(59, 130, 246, 0.6); }

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
.cyber-btn.secondary:hover { border-color: #fff; background: rgba(255,255,255,0.1); }

.cyber-btn.history { border-color: var(--neon-cyan); box-shadow: 0 0 15px rgba(6, 182, 212, 0.2); }
.cyber-btn.history:hover { background: rgba(6, 182, 212, 0.2); box-shadow: 0 0 40px rgba(6, 182, 212, 0.6); }

.cyber-btn.small { padding: 10px 20px; font-size: 0.8rem; clip-path: none; border-radius: 50px; border-color: #64748b;}
.cyber-btn.small.active { border-color: var(--neon-cyan); background: rgba(6, 182, 212, 0.2); box-shadow: 0 0 20px rgba(6, 182, 212, 0.4); }

<<<<<<< HEAD
=======
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

.setting-item label {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 500;
}

.control-row {
    display: flex;
    align-items: center;
    gap: 15px;
}

.slider {
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.3), rgba(139, 92, 246, 0.8));
    border-radius: 3px;
    outline: none;
    cursor: pointer;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #8b5cf6;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
}

.slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #8b5cf6;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    border: none;
}

.value-display {
    min-width: 80px;
    text-align: center;
    font-size: 0.9rem;
    color: #8b5cf6;
    font-weight: 600;
}

.resolution-select {
    padding: 10px 12px;
    background: rgba(30, 27, 75, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.5);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.9rem;
    cursor: pointer;
    transition: 0.3s;
}

.resolution-select:hover {
    border-color: #8b5cf6;
    background: rgba(30, 27, 75, 0.95);
}

.resolution-select:focus {
    outline: none;
    border-color: #8b5cf6;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
}

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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

<<<<<<< HEAD
=======
.video-wrapper {
    display: flex;
    flex-direction: column;
    gap: 15px;
    align-items: center;
}

.dream-result-video {
    width: 100%;
    max-width: 800px;
    border-radius: 12px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.6), 0 0 30px rgba(59, 130, 246, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
    background: #000;
}

.dream-result-video::-webkit-media-controls-panel {
    background-color: rgba(30, 27, 75, 0.9);
}

.video-info {
    text-align: center;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.7);
}

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
.loading-state-3d { 
  perspective: 800px; 
  text-align: center;
  flex-shrink: 0;
  padding: 40px 20px;
}
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

/* 侧边栏容器 - 固定在视口左侧 */
.sidebar-container {
  position: fixed;
  left: 0;
  top: 0;
  height: 100vh;
  width: 260px;
  background: rgba(32, 33, 35, 0.95);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,0.1);
  z-index: 1000;
  transition: width 0.3s ease, transform 0.3s ease;
  transform: translateX(-100%);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  /* 确保侧边栏始终固定在视口，不受页面滚动影响 */
  will-change: transform;
}

.sidebar-container.visible {
  transform: translateX(0);
}

.sidebar-container.collapsed {
  width: 60px;
}

.sidebar-container.collapsed.visible {
  transform: translateX(0);
}

/* 侧边栏内容 */
.sidebar {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
  /* 确保侧边栏高度固定为视口高度 */
  max-height: 100vh;
}

.sidebar-header {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  gap: 12px;
  min-height: 60px;
  flex-shrink: 0;
}

.sidebar-toggle {
  background: transparent;
  border: none;
  color: #fff;
  font-size: 20px;
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

<<<<<<< HEAD
=======
/* 编辑按钮样式 */
.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  flex-shrink: 0;
  z-index: 10;
}

.edit-btn, .save-btn, .cancel-btn {
  padding: 8px 16px;
  border-radius: 6px;
  border: 1px solid;
  font-family: 'Rajdhani', sans-serif;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  letter-spacing: 1px;
}

.edit-btn {
  background: rgba(6, 182, 212, 0.25);
  border-color: var(--neon-cyan);
  color: var(--neon-cyan);
  box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
  text-shadow: 0 0 8px var(--neon-cyan);
}

.edit-btn:hover:not(:disabled) {
  background: rgba(6, 182, 212, 0.25);
  box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);
  transform: translateY(-2px);
}

.save-btn {
  background: rgba(168, 85, 247, 0.2);
  border-color: var(--neon-purple);
  color: var(--neon-purple);
}

.save-btn:hover:not(:disabled) {
  background: rgba(168, 85, 247, 0.3);
  box-shadow: 0 0 20px rgba(168, 85, 247, 0.6);
  transform: translateY(-2px);
}

.cancel-btn {
  background: rgba(100, 100, 100, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  color: #fff;
}

.cancel-btn:hover:not(:disabled) {
  background: rgba(100, 100, 100, 0.3);
  transform: translateY(-2px);
}

.edit-btn:disabled, .save-btn:disabled, .cancel-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 编辑模式样式 */
.detail-section.editing {
  border: 2px solid var(--neon-cyan);
  border-radius: 8px;
  padding: 20px;
  background: rgba(6, 182, 212, 0.05);
  box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
}

.edit-textarea-wrapper {
  position: relative;
}

.edit-textarea {
  width: 100%;
  background: rgba(2, 6, 23, 0.8);
  border: 1px solid var(--neon-cyan);
  color: #fff;
  padding: 15px;
  font-size: 1rem;
  line-height: 1.6;
  border-radius: 6px;
  outline: none;
  transition: 0.3s;
  font-family: inherit;
  resize: vertical;
  min-height: 120px;
}

.edit-textarea:focus {
  border-color: var(--neon-purple);
  box-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
  background: rgba(2, 6, 23, 0.9);
}

.edit-mode-notice {
  margin: 20px 0;
  padding: 15px;
  background: rgba(236, 72, 153, 0.1);
  border: 1px solid var(--neon-pink);
  border-radius: 6px;
  color: var(--neon-pink);
  text-align: center;
  box-shadow: 0 0 15px rgba(236, 72, 153, 0.2);
}

.edit-mode-notice p {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.error-message {
  margin-top: 10px;
  padding: 10px;
  background: rgba(236, 72, 153, 0.2);
  border: 1px solid var(--neon-pink);
  border-radius: 6px;
  color: var(--neon-pink);
  font-size: 13px;
}

>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
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
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 4px;
<<<<<<< HEAD
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
  overflow: hidden;
  border: 2px solid rgba(168, 85, 247, 0.3);
  box-shadow: 0 10px 40px rgba(168, 85, 247, 0.4), 0 0 30px rgba(236, 72, 153, 0.3);
  transition: 0.3s;
}

.detail-image-wrapper:hover {
  border-color: var(--neon-pink);
  box-shadow: 0 15px 50px rgba(168, 85, 247, 0.6), 0 0 40px rgba(236, 72, 153, 0.5);
  transform: scale(1.02);
}

.detail-image {
  width: 100%;
  height: auto;
  display: block;
  transition: 0.3s;
}

.detail-footer {
  margin-top: 20px;
  padding-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.1);
  text-align: center;
}

.detail-date {
  color: #94a3b8;
  font-size: 0.85rem;
}

/* 动画 */
.slide-sidebar-enter-active,
.slide-sidebar-leave-active {
  transition: transform 0.3s ease;
}

.slide-sidebar-enter-from {
  transform: translateX(-100%);
}

.slide-sidebar-leave-to {
  transform: translateX(-100%);
}

.slide-detail-enter-active,
.slide-detail-leave-active {
  transition: transform 0.3s ease;
}

.slide-detail-enter-from {
  transform: translateX(100%);
}

.slide-detail-leave-to {
  transform: translateX(100%);
}

@media (max-width: 768px) {
  .glitch-title { font-size: 2.8rem; }
  .dashboard-grid { grid-template-columns: 1fr; }
  .control-deck, .action-module { flex-direction: column; align-items: stretch; }
  .cyber-btn { width: 100%; }
  .dream-universe-ui.sidebar-open,
  .dream-universe-ui.detail-open,
  .dream-universe-ui.sidebar-open.detail-open { 
    margin-left: 0;
    margin-right: 0;
  }
  .history-sidebar { width: 100%; }
  .detail-panel { width: 100%; }
  .score-cards { grid-template-columns: 1fr; }
}

.entry-checkbox {
  width: 16px;
  height: 16px;
  cursor: pointer;
  accent-color: var(--neon-purple);
  flex-shrink: 0;
}

.comprehensive-controls {
  padding: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  background: rgba(0,0,0,0.2);
  flex-shrink: 0;
}

.selection-info {
  color: #8e8ea0;
  font-size: 12px;
  margin-bottom: 8px;
}

.control-buttons {
  display: flex;
  gap: 8px;
}

.analyze-btn,
.clear-btn {
  flex: 1;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.2);
  background: rgba(255,255,255,0.05);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.analyze-btn {
  background: rgba(168, 85, 247, 0.2);
  border-color: var(--neon-purple);
}

.analyze-btn:hover:not(:disabled) {
  background: rgba(168, 85, 247, 0.3);
  transform: translateY(-1px);
}

.analyze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.clear-btn:hover:not(:disabled) {
  background: rgba(255,255,255,0.1);
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.comprehensive-deck {
  margin-top: 40px;
}

.comprehensive-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.score-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.score-card {
  background: rgba(0,0,0,0.3);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}

.score-card.overall {
  border-left: 3px solid var(--neon-purple);
}

.score-card.sleep {
  border-left: 3px solid var(--neon-cyan);
}

.score-card.emotion {
  border-left: 3px solid var(--neon-pink);
}

.score-label {
  color: #94a3b8;
  font-size: 0.9rem;
  margin-bottom: 10px;
}

.score-value {
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 15px;
}

.score-bar {
  width: 100%;
  height: 8px;
  background: rgba(255,255,255,0.1);
  border-radius: 4px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan));
  transition: width 1s ease;
}

.sleep-fill {
  background: linear-gradient(90deg, var(--neon-cyan), #06b6d4);
}

.emotion-fill {
  background: linear-gradient(90deg, var(--neon-pink), #ec4899);
}

.analysis-sections {
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.analysis-section {
  background: rgba(0,0,0,0.3);
  border-left: 3px solid var(--neon-cyan);
  padding: 20px;
  border-radius: 8px;
}

.analysis-section h4 {
  font-family: 'Orbitron', sans-serif;
  color: var(--neon-cyan);
  font-size: 1rem;
  margin: 0 0 15px 0;
  letter-spacing: 1px;
}

.analysis-section p {
  color: #e2e8f0;
  line-height: 1.8;
  margin: 0;
}

.emotion-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.emotion-tag {
  padding: 6px 12px;
  background: rgba(168, 85, 247, 0.2);
  border: 1px solid var(--neon-purple);
  border-radius: 16px;
  color: #fff;
  font-size: 0.85rem;
}

.suggestions-list {
  margin: 0;
  padding-left: 20px;
  color: #e2e8f0;
  line-height: 1.8;
}

.suggestions-list li {
  margin-bottom: 8px;
}

.dream-input-wrapper {
  position: relative;
}

.voice-btn {
  margin-top: 8px;
  padding: 6px 14px;
  border-radius: 20px;
  border: none;
  cursor: pointer;
  font-size: 14px;
  background: rgba(120, 120, 255, 0.15);
  color: #bfc8ff;
  transition: all 0.3s ease;
}

.voice-btn:hover {
  background: rgba(120, 120, 255, 0.3);
}

.voice-btn.recording {
  background: rgba(255, 80, 80, 0.25);
  color: #ffb3b3;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0.5); }
  70% { box-shadow: 0 0 0 10px rgba(255, 80, 80, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0); }
}

.voice-tip {
  margin-top: 6px;
  font-size: 12px;
  color: #999;
}

=======
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
  gap: 15px;
  flex-wrap: wrap;
}

.detail-header-main h2 {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.8rem;
  margin: 0;
  color: #fff;
  text-shadow: 0 0 10px var(--neon-purple);
  flex: 1;
  min-width: 200px;
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

.image-section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  gap: 15px;
  flex-wrap: wrap;
}

.regenerate-image-btn {
  padding: 8px 16px;
  border-radius: 6px;
  border: 1px solid var(--neon-purple);
  background: rgba(168, 85, 247, 0.2);
  color: var(--neon-purple);
  font-family: 'Rajdhani', sans-serif;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  letter-spacing: 1px;
  box-shadow: 0 0 10px rgba(168, 85, 247, 0.3);
  text-shadow: 0 0 8px var(--neon-purple);
}

.regenerate-image-btn:hover:not(:disabled) {
  background: rgba(168, 85, 247, 0.3);
  box-shadow: 0 0 20px rgba(168, 85, 247, 0.6);
  transform: translateY(-2px);
}

.regenerate-image-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.image-loading {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
}

.no-image-placeholder {
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #8e8ea0;
  font-size: 14px;
  border: 2px dashed rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.2);
}

.detail-image-wrapper {
  margin-top: 15px;
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid rgba(168, 85, 247, 0.3);
  box-shadow: 0 10px 40px rgba(168, 85, 247, 0.4), 0 0 30px rgba(236, 72, 153, 0.3);
  transition: 0.3s;
}

.detail-image-wrapper:hover {
  border-color: var(--neon-pink);
  box-shadow: 0 15px 50px rgba(168, 85, 247, 0.6), 0 0 40px rgba(236, 72, 153, 0.5);
  transform: scale(1.02);
}

.detail-image {
  width: 100%;
  height: auto;
  display: block;
  transition: 0.3s;
}

.detail-footer {
  margin-top: 20px;
  padding-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.1);
  text-align: center;
}

.detail-date {
  color: #94a3b8;
  font-size: 0.85rem;
}

/* 动画 */
.slide-sidebar-enter-active,
.slide-sidebar-leave-active {
  transition: transform 0.3s ease;
}

.slide-sidebar-enter-from {
  transform: translateX(-100%);
}

.slide-sidebar-leave-to {
  transform: translateX(-100%);
}

.slide-detail-enter-active,
.slide-detail-leave-active {
  transition: transform 0.3s ease;
}

.slide-detail-enter-from {
  transform: translateX(100%);
}

.slide-detail-leave-to {
  transform: translateX(100%);
}

@media (max-width: 768px) {
  .glitch-title { font-size: 2.8rem; }
  .dashboard-grid { grid-template-columns: 1fr; }
  .control-deck, .action-module { flex-direction: column; align-items: stretch; }
  .cyber-btn { width: 100%; }
  .dream-universe-ui.sidebar-open,
  .dream-universe-ui.detail-open,
  .dream-universe-ui.sidebar-open.detail-open { 
    margin-left: 0;
    margin-right: 0;
  }
  .history-sidebar { width: 100%; }
  .detail-panel { width: 100%; }
  .score-cards { grid-template-columns: 1fr; }
}

.entry-checkbox {
  width: 16px;
  height: 16px;
  cursor: pointer;
  accent-color: var(--neon-purple);
  flex-shrink: 0;
}

.comprehensive-controls {
  padding: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  background: rgba(0,0,0,0.2);
  flex-shrink: 0;
}

.selection-info {
  color: #8e8ea0;
  font-size: 12px;
  margin-bottom: 8px;
}

.control-buttons {
  display: flex;
  gap: 8px;
}

.analyze-btn,
.clear-btn {
  flex: 1;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.2);
  background: rgba(255,255,255,0.05);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.analyze-btn {
  background: rgba(168, 85, 247, 0.2);
  border-color: var(--neon-purple);
}

.analyze-btn:hover:not(:disabled) {
  background: rgba(168, 85, 247, 0.3);
  transform: translateY(-1px);
}

.analyze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.clear-btn:hover:not(:disabled) {
  background: rgba(255,255,255,0.1);
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.comprehensive-deck {
  margin-top: 40px;
}

.comprehensive-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.score-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.score-card {
  background: rgba(0,0,0,0.3);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}

.score-card.overall {
  border-left: 3px solid var(--neon-purple);
}

.score-card.sleep {
  border-left: 3px solid var(--neon-cyan);
}

.score-card.emotion {
  border-left: 3px solid var(--neon-pink);
}

.score-label {
  color: #94a3b8;
  font-size: 0.9rem;
  margin-bottom: 10px;
}

.score-value {
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 15px;
}

.score-bar {
  width: 100%;
  height: 8px;
  background: rgba(255,255,255,0.1);
  border-radius: 4px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan));
  transition: width 1s ease;
}

.sleep-fill {
  background: linear-gradient(90deg, var(--neon-cyan), #06b6d4);
}

.emotion-fill {
  background: linear-gradient(90deg, var(--neon-pink), #ec4899);
}

.analysis-sections {
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.analysis-section {
  background: rgba(0,0,0,0.3);
  border-left: 3px solid var(--neon-cyan);
  padding: 20px;
  border-radius: 8px;
}

.analysis-section h4 {
  font-family: 'Orbitron', sans-serif;
  color: var(--neon-cyan);
  font-size: 1rem;
  margin: 0 0 15px 0;
  letter-spacing: 1px;
}

.analysis-section p {
  color: #e2e8f0;
  line-height: 1.8;
  margin: 0;
}

.emotion-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.emotion-tag {
  padding: 6px 12px;
  background: rgba(168, 85, 247, 0.2);
  border: 1px solid var(--neon-purple);
  border-radius: 16px;
  color: #fff;
  font-size: 0.85rem;
}

.suggestions-list {
  margin: 0;
  padding-left: 20px;
  color: #e2e8f0;
  line-height: 1.8;
}

.suggestions-list li {
  margin-bottom: 8px;
}
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
</style>

