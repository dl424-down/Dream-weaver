#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梦境分析系统
基于BLIP模型实现梦境描述分析、情绪推测和视觉化生成
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import requests
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# 尝试导入PyTorch相关模块，如果失败则使用演示模式
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告：PyTorch未安装，将使用演示模式（不包含真实的BLIP模型推理）")

# 使用HuggingFace的BLIP模型（避免本地BLIP依赖与transformers版本冲突）
BLIP_AVAILABLE = TORCH_AVAILABLE

class DreamAnalyzer:
    """梦境分析器主类"""
    
    def __init__(self, device='cpu'):
        """
        初始化梦境分析器
        Args:
            device: 运行设备，默认CPU
        """
        if TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = device
        self.image_size = 224  # 为CPU优化，使用较小尺寸
        
        # 情绪关键词字典
        self.emotion_keywords = {
            '快乐': ['开心', '高兴', '愉快', '欢乐', '兴奋', '满足', '幸福', '喜悦'],
            '焦虑': ['担心', '紧张', '不安', '恐慌', '压力', '忧虑', '烦躁', '焦急'],
            '恐惧': ['害怕', '恐怖', '惊吓', '可怕', '威胁', '危险', '噩梦', '惊恐'],
            '悲伤': ['难过', '伤心', '痛苦', '沮丧', '失落', '绝望', '哭泣', '忧郁'],
            '愤怒': ['生气', '愤怒', '恼火', '暴躁', '愤恨', '怒火', '激怒', '愤慨'],
            '平静': ['安静', '平和', '宁静', '放松', '舒适', '安详', '祥和', '淡定'],
            '困惑': ['迷茫', '困惑', '不解', '疑惑', '混乱', '迷失', '不明白', '茫然']
        }
        
        # 梦境主题分类
        self.dream_themes = {
            '飞行': ['飞', '飞翔', '天空', '云朵', '鸟', '翅膀'],
            '追逐': ['追', '跑', '逃跑', '追赶', '逃避', '奔跑'],
            '水': ['水', '海', '河', '湖', '游泳', '淹没', '洪水'],
            '动物': ['狗', '猫', '蛇', '老虎', '狮子', '鸟', '鱼'],
            '人物': ['朋友', '家人', '陌生人', '老师', '同学', '父母'],
            '场所': ['学校', '家', '医院', '商店', '森林', '山', '城市'],
            '考试': ['考试', '测试', '答题', '成绩', '分数', '及格']
        }
        
        self.models_loaded = False
        
    def load_models(self):
        """加载BLIP模型"""
        if self.models_loaded:
            return
            
        if not TORCH_AVAILABLE:
            return
            
        print("正在加载BLIP模型(HuggingFace)...")
        try:
            # 使用HuggingFace权重
            self.hf_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.hf_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.hf_caption_model.eval()
            self.hf_caption_model = self.hf_caption_model.to(self.device)
            
            self.models_loaded = True
            print("模型加载完成！")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保网络连接正常，或手动下载模型文件")
    
    def preprocess_image(self, image_path: str):
        """
        预处理图像
        Args:
            image_path: 图像路径
        Returns:
            处理后的图像张量
        """
        try:
            if image_path.startswith('http'):
                raw_image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                raw_image = Image.open(image_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), 
                                interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                   (0.26862954, 0.26130258, 0.27577711))
            ])
            
            image = transform(raw_image).unsqueeze(0).to(self.device)
            return image, raw_image
            
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None, None
    
    def analyze_dream_text(self, dream_text: str) -> Dict:
        """
        分析梦境文本描述
        Args:
            dream_text: 梦境文本描述
        Returns:
            分析结果字典
        """
        result = {
            'emotions': [],
            'themes': [],
            'keywords': [],
            'analysis': ''
        }
        
        # 情绪分析
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in dream_text)
            if score > 0:
                emotion_scores[emotion] = score
        
        # 按分数排序情绪
        if emotion_scores:
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            result['emotions'] = [emotion for emotion, score in sorted_emotions[:3]]
        else:
            result['emotions'] = ['平静']
        
        # 主题分析
        theme_scores = {}
        for theme, keywords in self.dream_themes.items():
            score = sum(1 for keyword in keywords if keyword in dream_text)
            if score > 0:
                theme_scores[theme] = score
        
        if theme_scores:
            sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
            result['themes'] = [theme for theme, score in sorted_themes[:3]]
        
        # 关键词提取（简单实现）
        # 移除标点符号，提取名词性词汇
        clean_text = re.sub(r'[^\w\s]', '', dream_text)
        words = clean_text.split()
        # 过滤常见词汇，保留可能的关键词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        result['keywords'] = list(set(keywords))[:10]  # 去重并限制数量
        
        # 生成分析报告
        analysis_parts = []
        if result['emotions']:
            analysis_parts.append(f"主要情绪倾向：{', '.join(result['emotions'])}")
        if result['themes']:
            analysis_parts.append(f"梦境主题：{', '.join(result['themes'])}")
        
        # 简单的心理学解释
        primary_emotion = result['emotions'][0] if result['emotions'] else '平静'
        psychological_meanings = {
            '快乐': '可能反映了现实生活中的满足感和积极心态',
            '焦虑': '可能反映了对未来的担忧或当前面临的压力',
            '恐惧': '可能代表内心深处的不安全感或对未知的恐惧',
            '悲伤': '可能反映了内心的失落感或对过去的眷恋',
            '愤怒': '可能表示对某些情况的不满或压抑的情绪',
            '平静': '反映了内心的平和状态和良好的心理健康',
            '困惑': '可能表示对人生方向或某些问题的迷茫'
        }
        
        if primary_emotion in psychological_meanings:
            analysis_parts.append(f"心理解释：{psychological_meanings[primary_emotion]}")
        
        result['analysis'] = '。'.join(analysis_parts) + '。'
        
        return result
    
    def generate_image_caption(self, image_path: str) -> str:
        """
        为图像生成描述
        Args:
            image_path: 图像路径
        Returns:
            图像描述文本
        """
        if not TORCH_AVAILABLE:
            # 演示模式：返回模拟的图像描述
            return "演示模式：这是一张包含梦境相关元素的图片，可能包含象征性的物体或场景。"
            
        if not self.models_loaded:
            self.load_models()
        
        # 使用HuggingFace BLIP生成描述
        try:
            if image_path.startswith('http'):
                raw_image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            else:
                raw_image = Image.open(image_path).convert('RGB')
            inputs = self.hf_processor(images=raw_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.hf_caption_model.generate(**inputs, num_beams=3, max_length=30, min_length=10)
            return self.hf_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"图像描述生成失败: {e}")
            return "图像描述生成失败"
    
    def analyze_dream(self, dream_text: str, image_path: str = None) -> Dict:
        """
        综合分析梦境
        Args:
            dream_text: 梦境文本描述
            image_path: 相关图像路径（可选）
        Returns:
            完整的分析结果
        """
        result = {
            'text_analysis': self.analyze_dream_text(dream_text),
            'image_caption': None,
            'combined_analysis': '',
            'visualization_prompt': ''
        }
        
        # 如果有图像，生成图像描述
        if image_path:
            result['image_caption'] = self.generate_image_caption(image_path)
        
        # 生成综合分析
        text_analysis = result['text_analysis']
        combined_parts = [text_analysis['analysis']]
        
        if result['image_caption']:
            combined_parts.append(f"相关图像显示：{result['image_caption']}")
        
        result['combined_analysis'] = ' '.join(combined_parts)
        
        # 生成视觉化提示词
        emotions = text_analysis['emotions']
        themes = text_analysis['themes']
        keywords = text_analysis['keywords'][:5]  # 取前5个关键词
        
        prompt_parts = []
        if themes:
            prompt_parts.append(f"梦境场景包含{', '.join(themes)}")
        if keywords:
            prompt_parts.append(f"关键元素：{', '.join(keywords)}")
        if emotions:
            emotion_styles = {
                '快乐': '明亮温暖的色调，阳光灿烂',
                '焦虑': '紧张的氛围，不安定的构图',
                '恐惧': '阴暗神秘的环境，戏剧性的光影',
                '悲伤': '柔和忧郁的色彩，雨天或黄昏',
                '愤怒': '强烈对比的色彩，动态的构图',
                '平静': '和谐宁静的画面，柔和的光线',
                '困惑': '迷雾缭绕，模糊不清的边界'
            }
            primary_emotion = emotions[0]
            if primary_emotion in emotion_styles:
                prompt_parts.append(emotion_styles[primary_emotion])
        
        result['visualization_prompt'] = '，'.join(prompt_parts)
        
        return result

def main():
    """主函数，支持命令行参数传入梦境文本"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python dream_analyzer.py \"梦境描述文本\"")
        print("示例: python dream_analyzer.py \"我梦见自己在天空中飞翔\"")
        return
    
    # 从命令行参数获取梦境文本
    dream_text = sys.argv[1]
    
    analyzer = DreamAnalyzer()
    result = analyzer.analyze_dream(dream_text)
    
    print("=== 梦境分析结果 ===")
    print(f"情绪分析: {result['text_analysis']['emotions']}")
    print(f"主题分析: {result['text_analysis']['themes']}")
    print(f"关键词: {result['text_analysis']['keywords']}")
    print(f"心理分析: {result['text_analysis']['analysis']}")
    print(f"视觉化提示: {result['visualization_prompt']}")

if __name__ == "__main__":
    main()