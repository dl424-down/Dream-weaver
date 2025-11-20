<script setup>
import { ref } from 'vue'
import bgImage from './assets/image.jpg'

const dreamText = ref('')
const imageFile = ref(null)
const loading = ref(false)
const result = ref(null)
const error = ref('')

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
    const resp = await fetch('http://localhost:8000/analyze/text', {
      method: 'POST',
      body: form
    })
    if (!resp.ok) throw new Error('请求失败')
    const responseData = await resp.json()
    
    // 转换数据结构以匹配模板
    result.value = {
      text_analysis: {
        emotions: responseData.data?.emotions || [],
        themes: responseData.data?.themes || [],
        keywords: responseData.data?.keywords || []
      },
      combined_analysis: responseData.data?.detailed_analysis || '',
      visualization_prompt: responseData.data?.visualization_prompt || '',
      image_caption: responseData.data?.image_caption || ''
    }
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
    const resp = await fetch('http://localhost:8000/analyze/text', {
      method: 'POST',
      body: form
    })
    if (!resp.ok) throw new Error('请求失败')
     const responseData = await resp.json()
    
    // 转换数据结构以匹配模板
    result.value = {
      text_analysis: {
        emotions: responseData.data?.emotions || [],
        themes: responseData.data?.themes || [],
        keywords: responseData.data?.keywords || []
      },
      combined_analysis: responseData.data?.detailed_analysis || '',
      visualization_prompt: responseData.data?.visualization_prompt || ''
    }
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
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

          <div class="btns">
            <button class="btn primary" :disabled="loading" @click="analyze">
              {{ loading ? '解析中...' : '开始解析' }}
            </button>
            <button class="btn secondary" :disabled="loading" @click="analyzeTextOnly">
              梦境解析
            </button>
          </div>
        </div>

        <p v-if="error" class="error">{{ error }}</p>
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
    </div>
  </div>
  
</template>

<style scoped>
.page {
  position: relative;
  min-height: 100vh;
  background: radial-gradient(1200px 600px at 20% 10%, rgba(140, 120, 255, .25), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(255, 150, 200, .2), transparent 60%),
              radial-gradient(1000px 800px at 50% 100%, rgba(60, 180, 255, .15), transparent 60%),
              #0a0f1f;
  color: #eef2ff;
  overflow: hidden;
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
  margin-top: 20px;
  padding: 48px 28px;
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
  border: 1px solid rgba(255,255,255,.25);
}
.file-hint { margin-left: 10px; opacity: .8; }
.btns { display: flex; gap: 10px; }
.btn {
  margin-top: 4px;
  padding: 12px 16px;
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
</style>
