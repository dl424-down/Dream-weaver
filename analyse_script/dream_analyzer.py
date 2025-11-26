#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梦境分析系统
基于BLIP模型和DashScope LLM实现梦境描述分析、情绪推测和视觉化生成
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

# 尝试导入DashScope
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告：DashScope未安装，将使用关键词匹配模式（不包含LLM分析）")

# 使用HuggingFace的BLIP模型（避免本地BLIP依赖与transformers版本冲突）
BLIP_AVAILABLE = TORCH_AVAILABLE

# 加载环境变量
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

class DashScopeLLM:
    """DashScope LLM封装类"""
    
    def __init__(self):
        """初始化DashScope LLM"""
        if not DASHSCOPE_AVAILABLE:
            self.available = False
            return
        
        self.available = True
        # 从环境变量获取API Key
        api_key = os.environ.get("DASHSCOPE_API_KEY") or "sk-1bb88c7976254a628f3fa470a25b83c0"
        dashscope.api_key = api_key
        self.model = "qwen-turbo"
    
    def call(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> Optional[str]:
        """调用LLM"""
        if not self.available:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                max_tokens=max_tokens,
                timeout=30
            )
            
            if response and getattr(response, 'status_code', None) == 200:
                output = getattr(response, 'output', None)
                if output:
                    choices = getattr(output, 'choices', None)
                    if choices and len(choices) > 0:
                        msg_content = getattr(choices[0], 'message', None)
                        if msg_content:
                            content = getattr(msg_content, 'content', None)
                            if content:
                                return content.strip()
        except Exception as e:
            print(f"LLM调用失败: {e}")
        
        return None


class DreamAnalyzer:
    """梦境分析器主类"""
    
    def __init__(self, device='cpu', use_qwen=True):
        """
        初始化梦境分析器
        Args:
            device: 运行设备，默认CPU
            use_qwen: 是否使用通义千问LLM进行高级分析，默认True
        """
        if TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = device
        self.image_size = 224  # 为CPU优化，使用较小尺寸
        self.use_qwen = use_qwen and DASHSCOPE_AVAILABLE
        
        # 初始化LLM（如果可用）
        if self.use_qwen:
            self.llm = DashScopeLLM()
            print("✅ 已启用DashScope LLM（通义千问）进行高级分析")
        else:
            self.llm = None
            print("⚠️  使用关键词匹配模式（LLM不可用）")
        
        # 情绪关键词字典（作为备用）
        self.emotion_keywords = {
            '快乐': ['开心', '高兴', '愉快', '欢乐', '兴奋', '满足', '幸福', '喜悦'],
            '焦虑': ['担心', '紧张', '不安', '恐慌', '压力', '忧虑', '烦躁', '焦急'],
            '恐惧': ['害怕', '恐怖', '惊吓', '可怕', '威胁', '危险', '噩梦', '惊恐'],
            '悲伤': ['难过', '伤心', '痛苦', '沮丧', '失落', '绝望', '哭泣', '忧郁'],
            '愤怒': ['生气', '愤怒', '恼火', '暴躁', '愤恨', '怒火', '激怒', '愤慨'],
            '平静': ['安静', '平和', '宁静', '放松', '舒适', '安详', '祥和', '淡定'],
            '困惑': ['迷茫', '困惑', '不解', '疑惑', '混乱', '迷失', '不明白', '茫然']
        }
        
        # 梦境主题分类（作为备用）
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
    
    def analyze_dream_with_qwen(self, dream_text: str) -> Optional[Dict]:
        """
        使用通义千问LLM进行梦境分析
        Args:
            dream_text: 梦境文本描述
        Returns:
            分析结果字典，如果失败返回None
        """
        if not self.llm or not self.llm.available:
            return None
        
        prompt = f"""请分析以下梦境描述，提取情绪、主题和关键词。

梦境描述：{dream_text}

请以JSON格式返回分析结果，格式如下：
{{
    "emotions": ["情绪1", "情绪2", "情绪3"],
    "themes": ["主题1", "主题2", "主题3"],
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}}

要求：
1. emotions：识别梦境中的主要情绪（3个以内），如：快乐、焦虑、恐惧、悲伤、愤怒、平静、困惑等
2. themes：识别梦境主题（3个以内），如：飞行、追逐、水、动物、人物、场所、考试等
3. keywords：提取5个最重要的关键词

只返回JSON，不要其他文字。"""
        
        response = self.llm.call(prompt)
        if not response:
            return None
        
        # 尝试解析JSON
        try:
            # 尝试直接解析
            result = json.loads(response)
            if isinstance(result, dict) and 'emotions' in result:
                return result
        except:
            pass
        
        # 如果直接解析失败，尝试提取JSON部分
        try:
            # 查找JSON部分
            json_match = re.search(r'\{[^{}]*"emotions"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, dict) and 'emotions' in result:
                    return result
        except:
            pass
        
        return None
    
    def generate_detailed_analysis(self, dream_text: str, emotions: List[str], themes: List[str]) -> str:
        """
        使用LLM生成详细的心理分析
        Args:
            dream_text: 梦境文本
            emotions: 情绪列表
            themes: 主题列表
        Returns:
            详细的心理分析文本
        """
        if not self.llm or not self.llm.available:
            return ""
        
        emotion_str = "、".join(emotions) if emotions else "未知"
        theme_str = "、".join(themes) if themes else "未知"
        
        prompt = f"""请对以下梦境进行深入的心理分析。

梦境描述：{dream_text}
识别出的情绪：{emotion_str}
识别出的主题：{theme_str}

请从心理学角度进行详细分析，包括：
1. 情绪背后的心理含义
2. 梦境主题的象征意义
3. 可能的现实生活关联
4. 心理建议

分析要深入、专业，字数在200-400字之间。"""
        
        response = self.llm.call(prompt, max_tokens=1000)
        return response if response else ""
    
    def generate_visualization_prompt(self, dream_text: str, emotions: List[str], themes: List[str], keywords: List[str]) -> str:
        """
        使用LLM生成视觉化提示词
        Args:
            dream_text: 梦境文本
            emotions: 情绪列表
            themes: 主题列表
            keywords: 关键词列表
        Returns:
            视觉化提示词
        """
        if not self.llm or not self.llm.available:
            return ""
        
        emotion_str = "、".join(emotions) if emotions else "未知"
        theme_str = "、".join(themes) if themes else "未知"
        keyword_str = "、".join(keywords[:5]) if keywords else "未知"
        
        prompt = f"""请为以下梦境生成一个详细的图像生成提示词（用于AI绘图）。

梦境描述：{dream_text}
情绪：{emotion_str}
主题：{theme_str}
关键词：{keyword_str}

要求：
1. 用中文描述，详细且具体
2. 包含场景、氛围、色调、光影等视觉元素
3. 体现梦境的情绪和主题
4. 适合用于AI图像生成
5. 字数在100-200字之间

只返回提示词，不要其他说明文字。"""
        
        response = self.llm.call(prompt, max_tokens=500)
        return response if response else ""
    
    def analyze_dream_text(self, dream_text: str) -> Dict:
        """
        分析梦境文本描述（优先使用LLM，失败则使用关键词匹配）
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
        
        # 优先尝试使用LLM分析
        if self.use_qwen:
            llm_result = self.analyze_dream_with_qwen(dream_text)
            if llm_result:
                result['emotions'] = llm_result.get('emotions', [])
                result['themes'] = llm_result.get('themes', [])
                result['keywords'] = llm_result.get('keywords', [])
                print("✅ 使用LLM分析成功")
            else:
                print("⚠️  LLM分析失败，使用关键词匹配")
                # 回退到关键词匹配
                result = self._analyze_with_keywords(dream_text)
        else:
            # 直接使用关键词匹配
            result = self._analyze_with_keywords(dream_text)
        
        # 确保有默认值
        if not result['emotions']:
            result['emotions'] = ['平静']
        if not result['themes']:
            result['themes'] = []
        if not result['keywords']:
            result['keywords'] = []
        
        return result
    
    def _analyze_with_keywords(self, dream_text: str) -> Dict:
        """
        使用关键词匹配进行基础分析（备用方法）
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
        clean_text = re.sub(r'[^\w\s]', '', dream_text)
        words = clean_text.split()
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        result['keywords'] = list(set(keywords))[:10]
        
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
        综合分析梦境（使用LLM进行高级分析）
        Args:
            dream_text: 梦境文本描述
            image_path: 相关图像路径（可选）
        Returns:
            完整的分析结果
        """
        # 基础文本分析
        text_analysis = self.analyze_dream_text(dream_text)
        
        result = {
            'text_analysis': text_analysis,
            'image_caption': None,
            'combined_analysis': '',
            'detailed_analysis': '',  # 详细心理分析
            'visualization_prompt': ''
        }
        
        # 如果有图像，生成图像描述
        if image_path:
            result['image_caption'] = self.generate_image_caption(image_path)
        
        # 使用LLM生成详细的心理分析
        if self.use_qwen:
            detailed_analysis = self.generate_detailed_analysis(
                dream_text,
                text_analysis.get('emotions', []),
                text_analysis.get('themes', [])
            )
            if detailed_analysis:
                result['detailed_analysis'] = detailed_analysis
                print("✅ 已生成详细心理分析")
        
        # 生成综合分析（合并文本分析和图像描述）
        combined_parts = []
        if result['detailed_analysis']:
            combined_parts.append(result['detailed_analysis'])
        elif text_analysis.get('analysis'):
            combined_parts.append(text_analysis['analysis'])
        
        if result['image_caption']:
            combined_parts.append(f"相关图像显示：{result['image_caption']}")
        
        result['combined_analysis'] = ' '.join(combined_parts) if combined_parts else '暂无分析结果'
        
        # 使用LLM生成视觉化提示词
        if self.use_qwen:
            visualization_prompt = self.generate_visualization_prompt(
                dream_text,
                text_analysis.get('emotions', []),
                text_analysis.get('themes', []),
                text_analysis.get('keywords', [])
            )
            if visualization_prompt:
                result['visualization_prompt'] = visualization_prompt
                print("✅ 已生成视觉化提示词")
        
        # 如果LLM生成失败，使用备用方法
        if not result['visualization_prompt']:
            emotions = text_analysis.get('emotions', [])
            themes = text_analysis.get('themes', [])
            keywords = text_analysis.get('keywords', [])[:5]
            
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
                primary_emotion = emotions[0] if emotions else '平静'
                if primary_emotion in emotion_styles:
                    prompt_parts.append(emotion_styles[primary_emotion])
            
            result['visualization_prompt'] = '，'.join(prompt_parts) if prompt_parts else ''
        
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