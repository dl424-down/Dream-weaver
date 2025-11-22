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
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()
api_key=os.getenv("DASHSCOPE_API_KEY")
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

class DashScopeLLM:
    """
    封装对通义千问（DashScope）API 的调用。
    使用示例：
        llm = DashScopeLLM()
        result = llm.generate("我梦见我正在被人追杀")
    """
    #client:any
    #model:str="qwen-plus"
    #temperature:float=0.7
    def __init__(self, api_key=None, model="qwen-plus", temperature=0.7):
        self.api_key =os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.temperature = temperature
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        if not self.api_key:
            raise ValueError("❌ 未找到 DASHSCOPE_API_KEY，请在 .env 文件中设置。")

    def generate(self, prompt: str, system_prompt: str = "你是一名梦境情绪与象征分析专家。") -> str:
        #向通义千问发送文本请求
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = {
                "model": self.model,
                "input":{
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters":{
                    "temperature": self.temperature,
                    "max_tokens": 512
                }
            }
            print(f"正在调用模型: {self.model}") #调试
            response = requests.post(self.base_url, headers=headers, json=body)
            print(f"响应状态码: {response.status_code}")
            if response.status_code != 200:
                print(f"错误响应: {response.text}")
            response.raise_for_status()
            result = response.json()

            # 修正响应解析 - 处理不同的返回格式
            if "output" in result:
                output = result["output"]
                
                # 情况1: 直接返回文本内容
                if "text" in output:
                    content = output["text"].strip()
                    print(f"模型返回内容: {content}")
                    return content
                
                # 情况2: 通过choices返回
                elif "choices" in output and len(output["choices"]) > 0:
                    choice = output["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"].strip()
            
            print(f"意外响应格式: {result}")
            return ""

        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求失败: {e}")
            return ""
        except Exception as e:
            print(f"❌ 调用通义千问失败: {e}")
            return ""

class DreamAnalyzer:
    """梦境分析器主类"""
    def __init__(self, device='cpu', use_qwen=True):
        self.use_qwen = use_qwen
        self.device = torch.device(device) if TORCH_AVAILABLE else device
        self.image_size = 224
        
        # API配置
        self.API_KEY = os.getenv("DASHSCOPE_API_KEY") 
        self.MODEL = "qwen-plus"
        self.llm = DashScopeLLM(api_key=self.API_KEY, model=self.MODEL)

        '''
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
        '''
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
    def analyze_dream_with_qwen(self, dream_text: str) -> dict:
        """
        使用通义千问模型分析梦境文本，提取情绪、主题、关键词等。
        """
        prompt = f"""
        请分析以下梦境描述，提取以下三个核心信息并以 JSON 格式返回：

        1. emotions: 梦者在梦中表现出的主要情绪（如焦虑、恐惧、平静、快乐、悲伤、愤怒等），列出1-5个最显著的情绪
        2. themes: 梦境的主要主题和场景（如飞行、追逐、坠落、考试、迷路、重逢等），概括出1-2个核心主题
        3. keywords: 梦境中的关键元素和象征物（如人物、物品、环境、动作等），提取5-8个最重要的关键词
        梦境描述：{dream_text}

        请严格按照以下JSON格式输出，不要添加任何其他文字：
        {{
            "emotions": ["情绪1", "情绪2"，"情绪3"，"情绪4"，"情绪5"],
            "themes": ["主题1", "主题2"], 
            "keywords": ["关键词1", "关键词2", "关键词3"]
        }}
        """

        try:
            content = self.llm.generate(prompt)
            print(f"模型原始响应: {content}")  # 调试信息
            # 清理响应内容，移除可能的Markdown代码块标记
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            # 尝试解析JSON
            try:
                data = json.loads(content)
                print(f"成功解析JSON: {data}")  # 调试信息
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"清理后的内容: {cleaned_content}")
                # 尝试从文本中提取信息
                data = self._parse_analysis_response(cleaned_content)
             # 确保所有必需的字段都存在
            if "emotions" not in data:
               data["emotions"] = []
            if "themes" not in data:
                data["themes"] = []
            if "keywords" not in data:
                data["keywords"] = []
                
            return data
        
        except Exception as e:
            print(f"调用通义千问失败: {e}")
            return {
                "emotions": [],
                "themes": [],
                "keywords": []
            }

    def _parse_analysis_response(self, content: str) -> dict:
        #备用方法：当模型返回非标准JSON时手动解析
        data = {"emotions": [], "themes": [], "keywords": []}
        # 简单正则匹配
        try:
            emotion_matches = re.findall(r'"emotions":\s*\[(.*?)\]', content, re.DOTALL)
            theme_matches = re.findall(r'"themes":\s*\[(.*?)\]', content, re.DOTALL)
            keyword_matches = re.findall(r'"keywords":\s*\[(.*?)\]', content, re.DOTALL)
            
            if emotion_matches:
                emotions_str = emotion_matches[0]
                # 处理引号和逗号分隔的值
                emotions = re.findall(r'"([^"]*)"', emotions_str)
                if not emotions:
                    emotions = [e.strip() for e in emotions_str.split(",") if e.strip()]
                data["emotions"] = [e for e in emotions if e]
            
            if theme_matches:
                themes_str = theme_matches[0]
                themes = re.findall(r'"([^"]*)"', themes_str)
                if not themes:
                    themes = [t.strip() for t in themes_str.split(",") if t.strip()]
                data["themes"] = [t for t in themes if t]
            
            if keyword_matches:
                keywords_str = keyword_matches[0]
                keywords = re.findall(r'"([^"]*)"', keywords_str)
                if not keywords:
                    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                data["keywords"] = [k for k in keywords if k]
        except Exception as e:
            print(f"手动解析失败: {e}")
        
        return data

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
        综合分析梦境，整合三个核心功能
        """
        print("开始分析梦境...")
        
        # 功能1: 提取三要素（情绪、主题、关键词）
        print("正在提取梦境核心要素...")
        core_elements = self.analyze_dream_with_qwen(dream_text)
        
        # 功能2: 生成详细分析报告
        print("正在生成详细分析...")
        detailed_analysis = self.generate_detailed_analysis(
            dream_text, 
            core_elements.get('emotions', []),
            core_elements.get('themes', []), 
            core_elements.get('keywords', [])
        )
        
        # 功能3: 生成视觉化提示（可选）
        print(" 正在生成视觉化提示")
        visualization_prompt = self.generate_visualization_prompt(
            core_elements.get('emotions', []),
            core_elements.get('themes', []),
            core_elements.get('keywords', [])
        )
        
        # 整合结果
        result = {
            'core_elements': {
                'emotions': core_elements.get('emotions', []),
                'themes': core_elements.get('themes', []),
                'keywords': core_elements.get('keywords', [])
            },
            'detailed_analysis': detailed_analysis,
            'visualization_prompt': visualization_prompt,
            'image_caption': None
        }
        
        # 如果有图像，生成图像描述
        if image_path:
            print("正在分析梦境图像...")
            result['image_caption'] = self.generate_image_caption(image_path)
        
        print("梦境分析完成！")
        return result

    def generate_visualization_prompt(self, emotions: list, themes: list, keywords: list) -> str:
        """
        基于三要素生成视觉化提示词
        """
        if not any([emotions, themes, keywords]):
            return "一个抽象的艺术表达"
        
        prompt_parts = []
        
        # 添加主题
        if themes:
            prompt_parts.append(f"{'、'.join(themes)}场景")
        
        # 添加关键元素
        if keywords:
            key_elements = keywords[:3]  # 取前3个最重要的关键词
            prompt_parts.append(f"包含{'、'.join(key_elements)}")
        
        # 添加情绪氛围
        if emotions:
            emotion_mapping = {
                '快乐': '明亮温暖、阳光灿烂的氛围',
                '焦虑': '紧张不安、扭曲变形的风格',
                '恐惧': '阴暗神秘、戏剧性光影',
                '悲伤': '柔和忧郁、雨天黄昏色调', 
                '愤怒': '强烈对比、动态混乱的构图',
                '平静': '和谐宁静、柔和光线的画面',
                '困惑': '迷雾缭绕、模糊边界的超现实'
            }
            primary_emotion = emotions[0] if emotions else '平静'
            mood = emotion_mapping.get(primary_emotion, '超现实梦幻风格')
            prompt_parts.append(mood)
        
        # 添加艺术风格
        prompt_parts.append("梦幻般的超现实主义艺术风格，细腻的质感和氛围")
        
        return '，'.join(prompt_parts)
    def generate_detailed_analysis(self, dream_text: str, emotions: list, themes: list, keywords: list) -> str:
        """
        基于提取的三要素生成详细的梦境分析报告
        """
        if not emotions and not themes and not keywords:
            return "无法从梦境描述中提取足够的信息进行详细分析。"
            
        analysis_prompt = f"""
        基于以下梦境分析结果，生成一段详细的心理分析解释：

        梦境描述：{dream_text}
        识别出的情绪：{', '.join(emotions) if emotions else '未识别出明显情绪'}
        梦境主题：{', '.join(themes) if themes else '未识别出明显主题'} 
        关键元素：{', '.join(keywords) if keywords else '未提取到关键元素'}

        请从心理学角度分析这个梦境可能反映的心理状态、潜在的压力源或内心冲突，
        并提供一些建设性的解读建议。分析要专业且有同理心，长度在100-150字左右。
        请直接返回分析内容，不要添加额外的说明或标记。
        """

        try:
            analysis = self.llm.generate(analysis_prompt, "你是一名专业的梦境心理分析师")
            return analysis if analysis else "暂时无法生成详细分析。"
        except Exception as e:
            print(f"生成详细分析失败: {e}")
            return "梦境分析暂时无法提供详细解读。"

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
    
    print("\n" + "="*60)
    print("梦境分析结果")
    print("="*60)
    
    # 输出功能1: 核心三要素
    core_elements = result['core_elements']
    print(f"\n 核心分析要素:")
    print(f"  情绪识别: {', '.join(core_elements['emotions']) if core_elements['emotions'] else '暂无'}")
    print(f"  主题概括: {', '.join(core_elements['themes']) if core_elements['themes'] else '暂无'}")
    print(f"  关键词: {', '.join(core_elements['keywords']) if core_elements['keywords'] else '暂无'}")
    
    # 输出功能2: 详细分析
    print(f"\n📝 详细心理分析:")
    detailed_analysis = result.get('detailed_analysis', '分析失败')
    # 格式化输出，每行适当长度
    import textwrap
    for line in textwrap.wrap(detailed_analysis, width=50):
        print(f"  {line}")
    
    # 输出功能3: 视觉化提示
    visualization_prompt = result.get('visualization_prompt', '')
    if visualization_prompt:
        print(f"\n视觉化提示:")
        print(f"  {visualization_prompt}")
    
    print("="*60)
    
    # 如果有图像分析，也输出
    if result['image_caption']:
        print(f"\n图像描述:")
        print(f"  {result['image_caption']}")
    
    print("="*60)
if __name__ == "__main__":
    main()