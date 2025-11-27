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

# 尝试导入DashScope，如果失败则使用关键词匹配
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告：DashScope未安装，将使用关键词匹配模式（分析结果较简单）")

class DashScopeLLM:
    """DashScope LLM封装类"""
    
    def __init__(self):
        """初始化DashScope LLM"""
        self.api_key = None
        self._load_api_key()
    
    def _load_api_key(self):
        """从环境变量加载API密钥"""
        # 尝试从.env文件加载
        from dotenv import load_dotenv
        import os
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        load_dotenv(env_path)
        
        self.api_key = os.environ.get("DASHSCOPE_API_KEY") or "sk-1bb88c7976254a628f3fa470a25b83c0"
        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.api_key
    
    def call(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> Optional[str]:
        """
        调用DashScope LLM
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            max_tokens: 最大生成token数
        Returns:
            LLM生成的文本，失败返回None
        """
        if not DASHSCOPE_AVAILABLE:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = Generation.call(
                model="qwen-turbo",
                messages=messages,
                result_format="message",
                timeout=30,
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
            print(f"DashScope LLM调用失败: {e}")
        
        return None

class DreamAnalyzer:
    """梦境分析器主类"""
    
    def __init__(self, device='cpu', use_qwen=True):
        """
        初始化梦境分析器
        Args:
            device: 运行设备，默认CPU
            use_qwen: 是否使用DashScope LLM（通义千问），默认True
        """
        if TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = device
        self.image_size = 224  # 为CPU优化，使用较小尺寸
        self.use_qwen = use_qwen and DASHSCOPE_AVAILABLE
        
        # 初始化DashScope LLM
        if self.use_qwen:
            try:
                self.llm = DashScopeLLM()
            except Exception as e:
                print(f"DashScope LLM初始化失败: {e}，将使用关键词匹配模式")
                self.use_qwen = False
                self.llm = None
        else:
            self.llm = None
        
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
    
    def analyze_dream_with_qwen(self, dream_text: str) -> Optional[Dict]:
        """
        使用DashScope LLM分析梦境（优先使用）
        Args:
            dream_text: 梦境文本描述
        Returns:
            分析结果字典，失败返回None
        """
        if not self.use_qwen or not self.llm:
            return None
        
        system_prompt = """你是一个专业的梦境心理分析师。请分析用户提供的梦境描述，识别其中的情绪、主题和关键词。
请以JSON格式返回结果，格式如下：
{
    "emotions": ["情绪1", "情绪2", "情绪3"],
    "themes": ["主题1", "主题2", "主题3"],
    "keywords": ["关键词1", "关键词2", "关键词3", ...]
}

情绪可选值：快乐、焦虑、恐惧、悲伤、愤怒、平静、困惑
主题可选值：飞行、追逐、水、动物、人物、场所、考试
关键词：提取梦境中的关键名词和重要概念，最多10个

只返回JSON，不要其他文字。"""
        
        prompt = f"请分析以下梦境描述：\n{dream_text}\n\n请返回JSON格式的分析结果。"
        
        response = self.llm.call(prompt, system_prompt=system_prompt, max_tokens=500)
        if not response:
            return None
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 验证结果格式
                if isinstance(result, dict) and 'emotions' in result and 'themes' in result and 'keywords' in result:
                    return result
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
        
        return None
    
    def generate_detailed_analysis(self, dream_text: str, emotions: List[str], themes: List[str]) -> str:
        """
        生成详细的心理分析（使用LLM生成几百字的详细分析）
        Args:
            dream_text: 梦境文本描述
            emotions: 识别出的情绪列表
            themes: 识别出的主题列表
        Returns:
            详细的心理分析文本（200-400字）
        """
        if not self.use_qwen or not self.llm:
            # 回退到简单分析
            primary_emotion = emotions[0] if emotions else '平静'
            psychological_meanings = {
                '快乐': '可能反映了现实生活中的满足感和积极心态',
                '焦虑': '可能反映了对未来的担忧或当前面临的压力',
                '恐惧': '可能代表内心深处的不安全感或对未知的恐惧',
                '悲伤': '可能反映了内心的失落感或对过去的眷恋',
                '愤怒': '可能表示对某些情况的不满或压抑的情绪',
                '平静': '反映了内心的平和状态和良好的心理健康',
                '困惑': '可能表示对人生方向或某些问题的迷茫'
            }
            return psychological_meanings.get(primary_emotion, '需要进一步分析')
        
        system_prompt = """你是一个资深的梦境心理分析师，擅长从心理学、精神分析学和象征主义的角度解读梦境。
请提供专业、深入、详细的心理分析，字数控制在200-400字之间。"""
        
        emotions_str = '、'.join(emotions) if emotions else '未明确'
        themes_str = '、'.join(themes) if themes else '未明确'
        
        prompt = f"""请对以下梦境进行详细的心理分析：

梦境描述：{dream_text}

识别出的主要情绪：{emotions_str}
识别出的主题：{themes_str}

请从以下角度进行分析：
1. 情绪层面的心理意义（这些情绪反映了什么心理状态）
2. 主题和象征意义的深层解读（这些主题在心理学中的含义）
3. 可能反映的现实生活问题或内心冲突
4. 建议和启示

请用专业但易懂的语言，提供200-400字的详细分析。"""
        
        analysis = self.llm.call(prompt, system_prompt=system_prompt, max_tokens=1500)
        if analysis:
            return analysis.strip()
        
        # 回退到简单分析
        primary_emotion = emotions[0] if emotions else '平静'
        psychological_meanings = {
            '快乐': '可能反映了现实生活中的满足感和积极心态',
            '焦虑': '可能反映了对未来的担忧或当前面临的压力',
            '恐惧': '可能代表内心深处的不安全感或对未知的恐惧',
            '悲伤': '可能反映了内心的失落感或对过去的眷恋',
            '愤怒': '可能表示对某些情况的不满或压抑的情绪',
            '平静': '反映了内心的平和状态和良好的心理健康',
            '困惑': '可能表示对人生方向或某些问题的迷茫'
        }
        return psychological_meanings.get(primary_emotion, '需要进一步分析')
    
    def generate_visualization_prompt(self, dream_text: str, emotions: List[str], themes: List[str], keywords: List[str]) -> str:
        """
        生成详细的视觉化提示词（使用LLM生成100-200字的详细提示词）
        Args:
            dream_text: 梦境文本描述
            emotions: 识别出的情绪列表
            themes: 识别出的主题列表
            keywords: 提取的关键词列表
        Returns:
            详细的视觉化提示词（100-200字）
        """
        if not self.use_qwen or not self.llm:
            # 回退到简单提示词
            prompt_parts = []
            if themes:
                prompt_parts.append(f"梦境场景包含{', '.join(themes)}")
            if keywords:
                prompt_parts.append(f"关键元素：{', '.join(keywords[:5])}")
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
            return '，'.join(prompt_parts)
        
        system_prompt = """你是一名擅长中文叙事的AI视觉提示词专家。请使用中文描述梦境画面，语言应富有画面感与氛围感，便于艺术家或图像模型理解。每条提示保持120~200个汉字，涵盖场景、主体、光影、色彩、构图与情绪。"""
        
        emotions_str = '、'.join(emotions) if emotions else '未明确'
        themes_str = '、'.join(themes) if themes else '未明确'
        keywords_str = '、'.join(keywords[:5]) if keywords else '未明确'
        
        prompt = f"""请为以下梦境生成详细的中文图像生成提示词：

梦境描述：{dream_text}
主要情绪：{emotions_str}
主题：{themes_str}
关键词：{keywords_str}

要求：
1. 使用中文，120~200个汉字
2. 详细描述场景、氛围、色彩、光影、构图与镜头
3. 体现梦境的神秘感与超现实气息
4. 可加入情绪基调和材质细节
5. 只输出提示词本身，不要额外解释

只返回提示词，不要其他说明文字。"""
        
        visualization_prompt = self.llm.call(prompt, system_prompt=system_prompt, max_tokens=800)
        if visualization_prompt:
            return visualization_prompt.strip()
        
        # 回退到简单提示词
        prompt_parts = []
        if themes:
            prompt_parts.append(f"梦境场景包含{', '.join(themes)}")
        if keywords:
            prompt_parts.append(f"关键元素：{', '.join(keywords[:5])}")
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
        return '，'.join(prompt_parts)
    
    def _analyze_with_keywords(self, dream_text: str) -> Dict:
        """
        使用关键词匹配分析梦境（回退方案）
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
        
        return result
    
    def analyze_dream_text(self, dream_text: str) -> Dict:
        """
        分析梦境文本描述（优先使用LLM，失败时回退到关键词匹配）
        Args:
            dream_text: 梦境文本描述
        Returns:
            分析结果字典
        """
        # 优先尝试使用LLM分析
        llm_result = self.analyze_dream_with_qwen(dream_text)
        
        if llm_result:
            # LLM分析成功，生成详细的心理分析
            emotions = llm_result.get('emotions', [])
            themes = llm_result.get('themes', [])
            keywords = llm_result.get('keywords', [])
            
            # 生成详细的心理分析（几百字）
            detailed_analysis = self.generate_detailed_analysis(dream_text, emotions, themes)
            
            return {
                'emotions': emotions,
                'themes': themes,
                'keywords': keywords,
                'analysis': detailed_analysis
            }
        
        # LLM分析失败，回退到关键词匹配
        result = self._analyze_with_keywords(dream_text)
        
        # 生成简单的分析报告
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
        
        # 生成视觉化提示词（使用LLM生成详细提示词）
        emotions = text_analysis['emotions']
        themes = text_analysis['themes']
        keywords = text_analysis['keywords']
        
        result['visualization_prompt'] = self.generate_visualization_prompt(
            dream_text, emotions, themes, keywords
        )
        
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