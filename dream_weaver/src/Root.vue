<script setup>
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { ref, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'

const route = useRoute()

const menuItems = [
  { path: '/dream-analysis', label: '梦境分析', icon: '🌙' },
  { path: '/physio-data', label: '生理数据', icon: '💓' },
  { path: '/brainwave-analysis', label: '脑电波分析', icon: '📡' },
  { path: '/my-report', label: '我的报告', icon: '📑' }
]

// === Three.js 全局背景 ===
const canvasRef = ref(null)
let scene, camera, renderer, particles, starField
let mouseX = 0, mouseY = 0
let targetX = 0, targetY = 0
let windowHalfX = window.innerWidth / 2
let windowHalfY = window.innerHeight / 2

onMounted(() => {
  initThree()
  animate()
  document.addEventListener('mousemove', onDocumentMouseMove)
  window.addEventListener('resize', onWindowResize)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onDocumentMouseMove)
  window.removeEventListener('resize', onWindowResize)
  if (renderer) renderer.dispose()
  if (scene) scene.clear()
})

function initThree() {
  scene = new THREE.Scene()
  scene.fog = new THREE.FogExp2(0x0a0a2a, 0.001)

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 3000)
  camera.position.z = 1000

  renderer = new THREE.WebGLRenderer({ canvas: canvasRef.value, antialias: true, alpha: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setClearColor(0x000000, 1)

  const particleTexture = (() => {
    const canvas = document.createElement('canvas')
    canvas.width = 32
    canvas.height = 32
    const ctx = canvas.getContext('2d')
    const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16)
    gradient.addColorStop(0, 'rgba(255,255,255,1)')
    gradient.addColorStop(0.2, 'rgba(240,240,255,0.8)')
    gradient.addColorStop(0.5, 'rgba(120,120,255,0.2)')
    gradient.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, 32, 32)
    const texture = new THREE.Texture(canvas)
    texture.needsUpdate = true
    return texture
  })()

  const geometry = new THREE.BufferGeometry()
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
    if (colorType < 0.33) {
      colors[i * 3] = 0.6 + Math.random() * 0.2
      colors[i * 3 + 1] = 0.3 + Math.random() * 0.2
      colors[i * 3 + 2] = 0.9
    } else if (colorType < 0.66) {
      colors[i * 3] = 0.2 + Math.random() * 0.2
      colors[i * 3 + 1] = 0.5 + Math.random() * 0.3
      colors[i * 3 + 2] = 1.0
    } else {
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

  const starGeo = new THREE.BufferGeometry()
  const starCount = 4000
  const starPos = new Float32Array(starCount * 3)
  for (let i = 0; i < starCount * 3; i++) {
    starPos[i] = (Math.random() - 0.5) * 4000
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
  const starMat = new THREE.PointsMaterial({
    size: 3,
    color: 0xaaaaaa,
    blending: THREE.AdditiveBlending,
    transparent: true,
    opacity: 0.5
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

  targetX = mouseX * 0.05
  targetY = mouseY * 0.05
  camera.position.x += (targetX - camera.position.x) * 0.02
  camera.position.y += (-targetY - camera.position.y) * 0.02
  camera.lookAt(scene.position)

  particles.rotation.x += 0.0005
  particles.rotation.y += 0.001

  const positions = particles.geometry.attributes.position.array
  for (let i = 0; i < positions.length; i += 3) {
    positions[i + 2] += 1
    if (positions[i + 2] > 1500) {
      positions[i + 2] = -1500
    }
  }
  particles.geometry.attributes.position.needsUpdate = true

  starField.rotation.y -= 0.0002

  renderer.render(scene, camera)
}
</script>

<template>
  <div class="app-shell">
    <canvas ref="canvasRef" class="webgl-bg"></canvas>
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-icon">DW</div>
        <div class="brand-text">
          <div class="title">Dream Weaver</div>
          <div class="subtitle">解析潜意识的星图</div>
        </div>
      </div>

      <nav class="menu">
        <RouterLink
          v-for="item in menuItems"
          :key="item.path"
          :to="item.path"
          class="menu-item"
          :class="{ active: route.path === item.path }"
        >
          <span class="icon">{{ item.icon }}</span>
          <span class="text">{{ item.label }}</span>
        </RouterLink>
      </nav>
    </aside>

    <main class="main-area">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.webgl-bg {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
  pointer-events: none;
}

.app-shell {
  display: flex;
  height: 100vh;
  width: 100vw;
  color: #e5e7eb;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.sidebar {
  width: 256px;
  padding: 32px 18px 24px;
  /* 仿 glass-panel 效果 */
  backdrop-filter: blur(24px);
  background: radial-gradient(circle at top left, rgba(148, 163, 184, 0.2), transparent 55%),
    rgba(15, 23, 42, 0.82);
  border-right: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow:
    0 8px 32px rgba(15, 23, 42, 0.9),
    0 0 40px rgba(15, 23, 42, 0.9);
  display: flex;
  flex-direction: column;
  gap: 28px;
  position: relative;
  z-index: 20;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 10px;
}

.brand-icon {
  width: 40px;
  height: 40px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ffffff;
  color: #020617;
  font-weight: 800;
  font-size: 16px;
  box-shadow: 0 0 18px rgba(255, 255, 255, 0.45);
  border: 1px solid rgba(248, 250, 252, 0.9);
}

.brand-text .title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: #e5e7eb;
}

.brand-text .subtitle {
  margin-top: 4px;
  font-size: 11px;
  color: #94a3b8;
}

.menu {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 6px;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  color: #94a3b8;
  text-decoration: none;
  font-size: 13px;
  transition: background 0.18s ease, color 0.18s ease, transform 0.12s ease, box-shadow 0.18s ease,
    border-color 0.18s ease;
  border: 1px solid transparent;
}

.menu-item .icon {
  width: 22px;
  text-align: center;
}

.menu-item .text {
  flex: 1;
}

.menu-item:hover {
  background: rgba(15, 23, 42, 0.75);
  color: #f9fafb;
  border-color: rgba(148, 163, 184, 0.6);
  transform: translateY(-1px);
}

.menu-item.active {
  background: rgba(15, 23, 42, 0.9);
  color: #ffffff;
  border-color: rgba(255, 255, 255, 0.7);
  box-shadow:
    0 0 25px rgba(248, 250, 252, 0.4),
    0 0 35px rgba(148, 163, 184, 0.7);
}

.main-area {
  flex: 1;
  position: relative;
  overflow: hidden;
}

@media (max-width: 768px) {
  .app-shell {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding-inline: 20px;
  }

  .menu {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;
  }

  .menu-item {
    padding: 6px 10px;
    font-size: 12px;
  }
}
</style>


