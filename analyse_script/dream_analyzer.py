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

<<<<<<< HEAD
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
=======
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
        
        # 只从环境变量读取，不再在代码中硬编码密钥
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("警告：未在环境变量中找到 DASHSCOPE_API_KEY，将使用关键词匹配模式（不调用 DashScope LLM）")
            return

        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.api_key
            print("[DashScope] 已从环境变量加载 API Key")
    
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
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
        
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
<<<<<<< HEAD
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
=======
    
    def translate_image_caption(self, english_caption: str) -> str:
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
        """
        将英文图片描述翻译成中文
        Args:
            english_caption: 英文图片描述
        Returns:
            中文图片描述
        """
        if not self.use_qwen or not self.llm:
            # 如果没有LLM，返回原始英文（或简单处理）
            return english_caption
        
        try:
            prompt = f"请将以下英文图片描述翻译成中文，保持简洁自然：\n{english_caption}\n\n只返回中文翻译，不要其他说明。"
            translation = self.llm.call(prompt, max_tokens=200)
            if translation:
                return translation.strip()
        except Exception as e:
            print(f"翻译图片描述失败: {e}")
        
        # 翻译失败，返回原始英文
        return english_caption
    
    def analyze_dream_with_qwen(self, dream_text: str, image_caption_cn: Optional[str] = None) -> Optional[Dict]:
        """
        使用DashScope LLM分析梦境（优先使用）
        Args:
            dream_text: 梦境文本描述
            image_caption_cn: 图片的中文描述（可选）
        Returns:
            分析结果字典，失败返回None
        """
        if not self.use_qwen or not self.llm:
            return None
        
        system_prompt = """你是一个专业的梦境心理分析师。请综合分析用户提供的梦境描述和图片信息，识别其中的情绪、主题和关键词。
请以JSON格式返回结果，格式如下：
{
    "emotions": ["情绪1", "情绪2", "情绪3"],
    "themes": ["主题1", "主题2", "主题3"],
    "keywords": ["关键词1", "关键词2", "关键词3", ...]
}

情绪可选值：快乐、焦虑、恐惧、悲伤、愤怒、平静、困惑
主题可选值：飞行、追逐、水、动物、人物、场所、考试
关键词：提取梦境中的关键名词和重要概念，最多10个

如果提供了图片信息，请同时考虑文本描述和图片内容，进行融合分析。
只返回JSON，不要其他文字。"""
        
        if image_caption_cn:
            prompt = f"""请综合分析以下梦境：

文本描述：{dream_text}
图片显示：{image_caption_cn}

请同时考虑文本描述和图片内容，识别融合的情绪、主题和关键词。
如果图片和文本都显示相同的情绪或主题，可以增强该情绪/主题的权重。
如果图片和文本有差异，请综合考虑两者，给出更全面的分析。

请返回JSON格式的分析结果。"""
        else:
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
    
    def generate_detailed_analysis(
        self, 
        dream_text: str, 
        emotions: List[str], 
        themes: List[str],
        image_caption_cn: Optional[str] = None
    ) -> str:
        """
        生成详细的心理分析（使用LLM生成几百字的详细分析）
        如果提供了图片信息，会融合到分析中
        Args:
            dream_text: 梦境文本描述
            emotions: 识别出的情绪列表
            themes: 识别出的主题列表
            image_caption_cn: 图片的中文描述（可选）
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
请提供专业、深入、详细的心理分析，字数控制在200-400字之间。
如果提供了图片信息，请综合分析文本和图片，让分析更加全面和深入。"""
        
        emotions_str = '、'.join(emotions) if emotions else '未明确'
        themes_str = '、'.join(themes) if themes else '未明确'
        
        prompt = f"""请对以下梦境进行详细的心理分析：

梦境描述：{dream_text}

识别出的主要情绪：{emotions_str}
识别出的主题：{themes_str}"""
        
        if image_caption_cn:
            prompt += f"""

相关图片显示：{image_caption_cn}

请综合分析文本描述和图片内容，注意：
- 如果图片和文本都显示相同的情绪或主题，可以增强该情绪/主题的分析
- 如果图片和文本有差异，可以探讨这种差异的心理学意义
- 图片中的视觉元素可能提供额外的象征意义和深层信息
- 请将图片信息自然地融入到分析中，而不是简单地提及"""
        
        prompt += """

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
    
    def generate_visualization_prompt(
        self, 
        dream_text: str, 
        emotions: List[str], 
        themes: List[str], 
        keywords: List[str],
        image_caption_cn: Optional[str] = None
    ) -> str:
        """
        生成详细的视觉化提示词（使用LLM生成100-200字的详细提示词）
        如果提供了图片信息，会融合到提示词生成中
        Args:
            dream_text: 梦境文本描述
            emotions: 识别出的情绪列表
            themes: 识别出的主题列表
            keywords: 提取的关键词列表
            image_caption_cn: 图片的中文描述（可选）
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
            if image_caption_cn:
                prompt_parts.append(f"参考图片：{image_caption_cn}")
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
        
        system_prompt = """你是一名擅长中文叙事的AI视觉提示词专家。请使用中文描述梦境画面，语言应富有画面感与氛围感，便于艺术家或图像模型理解。每条提示保持120~200个汉字，涵盖场景、主体、光影、色彩、构图与情绪。
如果提供了参考图片信息，请结合文本描述和图片内容，生成融合的视觉化提示词。"""
        
        emotions_str = '、'.join(emotions) if emotions else '未明确'
        themes_str = '、'.join(themes) if themes else '未明确'
        keywords_str = '、'.join(keywords[:5]) if keywords else '未明确'
        
        prompt = f"""请为以下梦境生成详细的中文图像生成提示词：

梦境描述：{dream_text}
主要情绪：{emotions_str}
主题：{themes_str}
关键词：{keywords_str}"""
        
        if image_caption_cn:
            prompt += f"""

参考图片显示：{image_caption_cn}

请结合文本描述和参考图片，生成融合的视觉化提示词：
- 如果图片和文本都包含相同元素，可以增强该元素的描述
- 如果图片提供了视觉细节（如色彩、构图、氛围），可以融入这些细节
- 请自然地融合文本和图片信息，生成一个统一的视觉化提示词"""
        
        prompt += """

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
        if image_caption_cn:
            prompt_parts.append(f"参考图片：{image_caption_cn}")
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
    
    def analyze_dream_text(self, dream_text: str, image_path: Optional[str] = None) -> Dict:
        """
        分析梦境文本描述（优先使用LLM，失败时回退到关键词匹配）
        如果提供了图片，会将图片信息融入分析中
        Args:
            dream_text: 梦境文本描述
            image_path: 相关图像路径（可选）
        Returns:
            分析结果字典
        """
        # 如果有图片，先获取图片描述并翻译
        image_caption_cn = None
        if image_path:
            try:
                # 生成英文图片描述
                image_caption_en = self.generate_image_caption(image_path)
                if image_caption_en and image_caption_en != "图像描述生成失败":
                    # 翻译成中文
                    image_caption_cn = self.translate_image_caption(image_caption_en)
                    print(f"[图片分析] 图片描述（中文）：{image_caption_cn}")
            except Exception as e:
                print(f"[图片分析] 获取图片信息失败: {e}")
        
        # 优先尝试使用LLM分析（如果提供了图片，会融合图片信息）
        llm_result = self.analyze_dream_with_qwen(dream_text, image_caption_cn=image_caption_cn)
        
        if llm_result:
            # LLM分析成功，生成详细的心理分析
            emotions = llm_result.get('emotions', [])
            themes = llm_result.get('themes', [])
            keywords = llm_result.get('keywords', [])
            
            # 生成详细的心理分析（几百字），传入图片信息
            detailed_analysis = self.generate_detailed_analysis(
                dream_text, emotions, themes, image_caption_cn=image_caption_cn
            )
            
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
        
        # 如果有图片信息，也添加到分析中
        if image_caption_cn:
            analysis_parts.append(f"相关图片显示：{image_caption_cn}")
        
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
<<<<<<< HEAD
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
=======
        综合分析梦境
        如果提供了图片，图片信息会融入到文本分析中，而不是简单拼接
        Args:
            dream_text: 梦境文本描述
            image_path: 相关图像路径（可选）
        Returns:
            完整的分析结果
        """
        # 在文本分析阶段就传入图片路径，让图片信息融入到分析中
        result = {
            'text_analysis': self.analyze_dream_text(dream_text, image_path=image_path),
            'image_caption': None,
            'combined_analysis': '',
            'visualization_prompt': ''
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b
        }
        
        # 如果有图像，保存图片描述（用于显示，但分析已经融合了）
        if image_path:
<<<<<<< HEAD
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
=======
            try:
                image_caption_en = self.generate_image_caption(image_path)
                if image_caption_en and image_caption_en != "图像描述生成失败":
                    # 翻译成中文保存
                    result['image_caption'] = self.translate_image_caption(image_caption_en)
            except Exception as e:
                print(f"[图片分析] 获取图片描述失败: {e}")
        
        # 综合分析就是文本分析的结果（已经融合了图片信息）
        text_analysis = result['text_analysis']
        result['combined_analysis'] = text_analysis['analysis']
        
        # 生成视觉化提示词（使用LLM生成详细提示词，传入图片信息）
        emotions = text_analysis['emotions']
        themes = text_analysis['themes']
        keywords = text_analysis['keywords']
        
        # 传入图片信息，让提示词生成也考虑图片
        result['visualization_prompt'] = self.generate_visualization_prompt(
            dream_text, emotions, themes, keywords, image_caption_cn=result.get('image_caption')
        )
        
        return result
    
    def analyze_comprehensive(self, entries: List[Dict]) -> Dict:
        """
        综合分析多个梦境记录
        Args:
            entries: 梦境记录列表，每个记录包含 dream_text, text_analysis 等字段
        Returns:
            综合分析结果，包含评分、总结、建议等
        """
        if not entries:
            return {
                "overall_score": 50,
                "sleep_quality": 50,
                "emotion_score": 50,
                "summary": "未找到有效的梦境记录",
                "emotion_breakdown": {},
                "sleep_analysis": "无法评估",
                "suggestions": []
            }
        
        # 收集所有梦境文本和分析结果
        all_dreams = []
        all_emotions = []
        all_themes = []
        
        for entry in entries:
            dream_text = entry.get("dream_text", "")
            text_analysis = entry.get("text_analysis", {})
            if isinstance(text_analysis, str):
                try:
                    import json
                    text_analysis = json.loads(text_analysis)
                except:
                    text_analysis = {}
            
            all_dreams.append(dream_text)
            if text_analysis:
                all_emotions.extend(text_analysis.get("emotions", []))
                all_themes.extend(text_analysis.get("themes", []))
        
        # 使用LLM进行综合分析
        if self.use_qwen and self.llm:
            return self._comprehensive_analysis_with_llm(all_dreams, all_emotions, all_themes, len(entries))
        else:
            return self._comprehensive_analysis_with_keywords(all_dreams, all_emotions, all_themes, len(entries))
    
    def _comprehensive_analysis_with_llm(self, dreams: List[str], emotions: List[str], themes: List[str], count: int) -> Dict:
        """使用LLM进行综合分析"""
        dreams_text = "\n".join([f"梦境{i+1}: {dream}" for i, dream in enumerate(dreams)])
        
        prompt = f"""请对以下{count}个梦境进行综合分析，给出精准的评分和评估。

{dreams_text}

请从以下维度进行分析：
1. 综合状态评分（0-100分）：基于所有梦境的整体情绪、主题、内容，评估用户当前的心理状态
2. 睡眠质量评分（0-100分）：基于梦境的内容、情绪强度、主题类型，评估睡眠质量
3. 情绪状态评分（0-100分）：基于情绪分析，评估整体情绪健康度
4. 情绪分布：统计各种情绪的出现频率和强度
5. 睡眠质量分析：详细分析睡眠质量的原因
6. 建议：提供3-5条具体的改善建议

请以JSON格式返回结果，格式如下：
{{
    "overall_score": 75,
    "sleep_quality": 80,
    "emotion_score": 70,
    "summary": "综合分析总结（200字左右）",
    "emotion_breakdown": {{"焦虑": 60, "平静": 30, "快乐": 10}},
    "sleep_analysis": "睡眠质量详细分析（150字左右）",
    "suggestions": ["建议1", "建议2", "建议3"]
}}

请确保评分精准，分析深入，建议实用。只返回JSON，不要其他文字。"""
        
        try:
            response = self.llm.call(prompt, max_tokens=2000)
            if response:
                # 尝试提取JSON
                import json
                import re
                # 查找JSON部分
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    # 验证和规范化结果
                    normalized = self._normalize_comprehensive_result(result)

                    # 如果返回的仍然是“默认占位值”，说明大模型没有按要求返回结构化结果，
                    # 此时回退到关键词综合分析，保证前端看到的是有内容的、基于梦境的报告。
                    if (
                        normalized.get("overall_score") == 50
                        and normalized.get("sleep_quality") == 50
                        and normalized.get("emotion_score") == 50
                        and normalized.get("summary") in ("综合分析完成", "", None)
                        and not normalized.get("emotion_breakdown")
                    ):
                        print("[综合分析] LLM 返回结果疑似占位，回退到关键词综合分析")
                        return self._comprehensive_analysis_with_keywords(dreams, emotions, themes, count)

                    return normalized
        except Exception as e:
            print(f"[WARN] LLM综合分析失败: {e}")
        
        # LLM失败，回退到关键词分析
        return self._comprehensive_analysis_with_keywords(dreams, emotions, themes, count)
    
    def _comprehensive_analysis_with_keywords(self, dreams: List[str], emotions: List[str], themes: List[str], count: int) -> Dict:
        """
        使用规则对多个梦境进行严谨的综合打分（完全本地，不依赖 LLM）。

        设计思路：
        - 情绪维度：从情绪标签中统计正负面比例，得到情绪健康度（0-100）
        - 睡眠维度：从梦境文本中识别睡眠相关正负关键词，得到睡眠质量（0-100）
        - 综合状态：情绪 60% + 睡眠 40%，更偏向心理状态本身
        """
        # -------- 1. 统计情绪标签，计算情绪分布 --------
        emotion_count: Dict[str, int] = {}
        for emotion in emotions:
            if not emotion:
                continue
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

        total_emotions = sum(emotion_count.values())
        if total_emotions == 0:
            # 没有情绪标签时，视为中性
            total_emotions = 1

        emotion_breakdown = {
            emotion: round(cnt / total_emotions * 100)
            for emotion, cnt in emotion_count.items()
        }

        # 正负情绪集合（可以根据需要继续细化）
        positive_set = {"快乐", "喜悦", "满足", "幸福", "平静", "放松"}
        negative_set = {"焦虑", "恐惧", "悲伤", "愤怒", "抑郁", "压力", "不安", "噩梦", "恐慌", "孤独"}

        positive_total = sum(cnt for emo, cnt in emotion_count.items() if emo in positive_set)
        negative_total = sum(cnt for emo, cnt in emotion_count.items() if emo in negative_set)

        # 情绪指数 E ∈ [-1, 1]：正面越多越接近 1，负面越多越接近 -1
        balance_den = positive_total + negative_total
        if balance_den == 0:
            emotion_index = 0.0
        else:
            emotion_index = (positive_total - negative_total) / balance_den

        # 映射到 0-100 分，50 为中性
        emotion_score = int(round((emotion_index + 1) / 2 * 100))
        emotion_score = max(0, min(100, emotion_score))

        # -------- 2. 从文本中抽取睡眠相关信号，计算睡眠质量 --------
        all_text = "。".join(dreams) if dreams else ""

        positive_sleep_words = ["放松", "舒适", "平静", "安详", "入睡", "熟睡", "安心", "安稳", "清醒而愉快"]
        negative_sleep_words = ["噩梦", "惊醒", "失眠", "难以入睡", "反复醒来", "睡不着", "压迫感", "窒息", "崩溃", "焦虑", "恐惧", "考试", "追赶", "坠落"]

        pos_hits = sum(all_text.count(w) for w in positive_sleep_words)
        neg_hits = sum(all_text.count(w) for w in negative_sleep_words)

        # 以 70 为基准分，正面每命中一次 +5，负面每命中一次 -8
        sleep_score_base = 70
        sleep_score = sleep_score_base + pos_hits * 5 - neg_hits * 8

        # 情绪会影响睡眠评分：极端负面情绪会拉低睡眠，极端正面略微抬高
        if emotion_score < 40:
            sleep_score -= 10
        elif emotion_score > 75:
            sleep_score += 5

        sleep_quality = max(0, min(100, sleep_score))

        # -------- 3. 计算综合状态分：更偏向情绪状态 --------
        overall_score = int(round(emotion_score * 0.6 + sleep_quality * 0.4))
        overall_score = max(0, min(100, overall_score))

        # -------- 4. 生成文字总结 --------
        if emotion_count:
            dominant_emotion = max(emotion_count.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "平静"

        summary_parts = [f"基于{count}个梦境的分析，您当前的主要情绪倾向是{dominant_emotion}。"]
        if overall_score >= 80:
            summary_parts.append("整体心理状态较为健康稳定，能够较好地应对生活中的压力与变化。")
        elif overall_score >= 60:
            summary_parts.append("整体状态尚可，但存在一定的情绪波动，可能与近期压力或重要事件有关。")
        elif overall_score >= 40:
            summary_parts.append("整体状态偏向紧张或低落，梦境中负面情绪信号较为明显，建议适当减压并关注自我照顾。")
        else:
            summary_parts.append("整体状态处于较高风险区间，梦境中反复出现强烈的焦虑、恐惧或无助感，建议认真对待并考虑寻求专业支持。")

        summary = "".join(summary_parts)

        # -------- 5. 睡眠质量文字分析 --------
        if sleep_quality >= 80:
            sleep_analysis = (
                "整体睡眠质量处于较为理想的水平，梦境内容多为可控或正向场景，即使出现少量紧张情节，"
                "也往往带有解决、化解或顺利收尾的倾向。这通常说明你的身心恢复能力较强，夜间能够较好地完成对白天信息的整理与整合。"
                "在当前基础上，可以继续保持规律作息和适度运动，避免在睡前大量摄入咖啡因或进行高强度用脑活动，以巩固这种稳定的睡眠状态。"
            )
        elif sleep_quality >= 60:
            sleep_analysis = (
                f"当前睡眠质量大致处于中等水平（约{sleep_quality}分），梦境中既包含一定程度的紧张、压力或矛盾情节，"
                "也仍然保留了一些相对平和或可控的场景。这样的梦境模式往往提示：你在日常生活中承受了一定压力，"
                "但整体仍具备应对与自我调节的能力。通过更加规律的作息、适当减少睡前使用电子设备，以及在白天有意识地安排放松时段，"
                "可以进一步提高入睡速度与睡眠深度，从而让夜间恢复更加充分。"
            )
        elif sleep_quality >= 40:
            sleep_analysis = (
                f"当前睡眠质量已经出现明显波动（约{sleep_quality}分），梦境中较多出现焦虑、压迫、失败或被追赶等情节，"
                "这些内容往往反映出白天累积的紧张与担忧尚未得到有效释放。长此以往，可能会让你在醒来时依然感到疲惫，"
                "甚至影响白天的专注度与情绪稳定。建议你在日间为自己安排一些“缓冲区”，例如在工作与睡眠之间留出放松时间，"
                "通过散步、拉伸、轻度运动或与信任的人交流来减轻心理负担，同时逐步建立固定的睡前仪式，帮助大脑形成“可以休息”的信号。"
            )
        else:
            sleep_analysis = (
                f"当前睡眠质量处于较低水平（约{sleep_quality}分），梦境频繁呈现噩梦、惊醒、无助或强烈恐惧等情节，"
                "提示身心正承受较大的持续性压力。这类梦境往往并非偶发现象，而是长期紧张、情绪压抑或生活节奏失衡的“报警信号”。"
                "如果你已经明显感受到白天的疲惫、易怒、注意力难以集中，或对睡眠本身产生担忧，建议尽早正视这一状况，"
                "一方面从作息、运动、饮食和环境等方面系统改善睡眠条件，另一方面在条件允许时，考虑与专业心理咨询师或精神科医生进行深入评估，"
                "以获得更有针对性的支持与帮助。"
            )

        # -------- 6. 生成具体建议 --------
        suggestions: List[str] = []

        if overall_score < 60:
            suggestions.append(
                "为自己预留出固定的放松时段，例如每天睡前30分钟刻意远离工作与屏幕，"
                "可以进行缓慢深呼吸、简单拉伸、冥想或听舒缓音乐，让身体逐步从“警觉模式”过渡到“休息模式”。"
            )
        if negative_total > positive_total:
            suggestions.append(
                "建议尝试记录梦境与当天的情绪变化，例如在睡前花5分钟写下令人印象深刻的画面与当下心情，"
                "这有助于你更清晰地看到压力来源，并在白天找到可以调整或求助的具体方向，而不是把一切都压在心里。"
            )
        if "焦虑" in emotion_count or emotion_score < 50:
            suggestions.append(
                "如果长期感到紧张、担心或难以放松，且这种状态已经影响到学习、工作或人际关系，"
                "建议与你信任的家人、朋友或老师交流感受；在条件允许的情况下，考虑预约专业心理咨询，"
                "通过系统性的评估与对话寻找更深层的原因和应对策略。"
            )
        if sleep_quality < 60:
            suggestions.append(
                "优化睡眠环境非常关键：尽量让卧室保持安静、昏暗、温度适宜，"
                "避免在床上长时间刷手机或处理学习/工作任务，让“大脑知道床是用来休息的”，"
                "从而减少入睡前的大脑过度活跃。"
            )
            suggestions.append(
                "尽量形成相对稳定的作息节奏，例如在固定时间上床和起床，"
                "白天适度增加光照与活动量，晚间逐步降低节奏，让生物钟重新形成规律，这对改善睡眠质量非常重要。"
            )
        if not suggestions:
            suggestions.append(
                "当前整体状态相对稳定，可以继续保持规律的作息与适度的运动，"
                "同时保持对自身情绪与梦境的觉察，一旦发现长时间的情绪低落或睡眠明显变差，及时做出调整或寻求支持。"
            )
            suggestions.append(
                "建议定期给自己安排一些真正愉悦而非“刷手机式放空”的活动，例如户外散步、培养一项兴趣爱好，"
                "或与信任的人进行高质量的面对面交流，这些都有助于在日常生活中不断补充心理能量。"
            )

        return {
            "overall_score": overall_score,
            "sleep_quality": sleep_quality,
            "emotion_score": emotion_score,
            "summary": summary,
            "emotion_breakdown": emotion_breakdown,
            "sleep_analysis": sleep_analysis,
            "suggestions": suggestions[:5],
        }
    
    def _normalize_comprehensive_result(self, result: Dict) -> Dict:
        """规范化综合分析结果"""
        # 确保所有必需的字段存在
        normalized = {
            "overall_score": max(0, min(100, int(result.get("overall_score", 50)))),
            "sleep_quality": max(0, min(100, int(result.get("sleep_quality", 50)))),
            "emotion_score": max(0, min(100, int(result.get("emotion_score", 50)))),
            "summary": result.get("summary", "综合分析完成"),
            "emotion_breakdown": result.get("emotion_breakdown", {}),
            "sleep_analysis": result.get("sleep_analysis", "睡眠质量评估完成"),
            "suggestions": result.get("suggestions", [])
        }
        
        # 确保建议是列表
        if not isinstance(normalized["suggestions"], list):
            normalized["suggestions"] = []
        
        # 限制建议数量
        normalized["suggestions"] = normalized["suggestions"][:5]
        
        return normalized
>>>>>>> d03ecce35c11d08008d4e0265dfea3455de45e7b

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