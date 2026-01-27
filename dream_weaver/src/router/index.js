import { createRouter, createWebHistory } from 'vue-router'

// 页面组件
import DreamAnalysis from '../App.vue'
import PhysioData from '../views/PhysioData.vue'
import BrainwaveAnalysis from '../views/BrainwaveAnalysis.vue'
import MyReport from '../views/MyReport.vue'

const routes = [
  {
    path: '/',
    redirect: '/dream-analysis'
  },
  {
    path: '/dream-analysis',
    name: 'DreamAnalysis',
    component: DreamAnalysis
  },
  {
    path: '/physio-data',
    name: 'PhysioData',
    component: PhysioData
  },
  {
    path: '/brainwave-analysis',
    name: 'BrainwaveAnalysis',
    component: BrainwaveAnalysis
  },
  {
    path: '/my-report',
    name: 'MyReport',
    component: MyReport
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router


