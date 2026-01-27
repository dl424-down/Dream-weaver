#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 缓存工具
提供梦境分析结果的缓存功能
"""

import json
import hashlib
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# 加载环境变量
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

# 尝试导入 Redis，如果失败则使用空实现
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[警告] Redis 未安装，缓存功能将被禁用。请运行: pip install redis")


class RedisCache:
    """Redis 缓存管理类"""
    
    def __init__(self):
        """初始化 Redis 连接"""
        self.redis_client = None
        self.enabled = False
        
        if not REDIS_AVAILABLE:
            print("[缓存] Redis 库未安装，缓存功能已禁用")
            return
        
        try:
            # 从环境变量读取 Redis 配置，如果没有则使用默认值
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            redis_db = int(os.environ.get("REDIS_DB", 0))
            redis_password = os.environ.get("REDIS_PASSWORD", None)
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,  # 自动解码为字符串
                socket_connect_timeout=3,  # 连接超时3秒
                socket_timeout=3,  # 操作超时3秒
            )
            
            # 测试连接
            self.redis_client.ping()
            self.enabled = True
            print(f"[缓存] Redis 连接成功 (host={redis_host}, port={redis_port}, db={redis_db})")
            
        except Exception as e:
            print(f"[警告] Redis 连接失败: {e}，缓存功能已禁用")
            self.enabled = False
            self.redis_client = None
    
    def _generate_cache_key(self, dream_text: str, has_image: bool = False) -> str:
        """
        生成缓存键
        Args:
            dream_text: 梦境文本
            has_image: 是否包含图片
        Returns:
            缓存键字符串
        """
        # 使用文本内容生成哈希值
        text_hash = hashlib.md5(dream_text.encode('utf-8')).hexdigest()
        image_flag = "with_image" if has_image else "text_only"
        return f"dream:analysis:{image_flag}:{text_hash}"
    
    def get_analysis_cache(self, dream_text: str, has_image: bool = False) -> Optional[Dict[str, Any]]:
        """
        获取分析结果缓存
        Args:
            dream_text: 梦境文本
            has_image: 是否包含图片
        Returns:
            缓存的分析结果，如果不存在则返回 None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(dream_text, has_image)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                print(f"[缓存] 命中缓存: {cache_key[:50]}...")
                return result
            
            return None
            
        except Exception as e:
            print(f"[警告] 读取缓存失败: {e}")
            return None
    
    def set_analysis_cache(
        self, 
        dream_text: str, 
        analysis_result: Dict[str, Any],
        has_image: bool = False,
        ttl: int = 7 * 24 * 60 * 60  # 默认7天
    ) -> bool:
        """
        设置分析结果缓存
        Args:
            dream_text: 梦境文本
            analysis_result: 分析结果字典
            has_image: 是否包含图片
            ttl: 缓存过期时间（秒），默认7天
        Returns:
            是否设置成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(dream_text, has_image)
            # 将结果序列化为 JSON
            cache_data = json.dumps(analysis_result, ensure_ascii=False)
            # 设置缓存，带过期时间
            self.redis_client.setex(cache_key, ttl, cache_data)
            print(f"[缓存] 已缓存分析结果: {cache_key[:50]}... (TTL: {ttl}秒)")
            return True
            
        except Exception as e:
            print(f"[警告] 写入缓存失败: {e}")
            return False
    
    def delete_analysis_cache(self, dream_text: str, has_image: bool = False) -> bool:
        """
        删除分析结果缓存
        Args:
            dream_text: 梦境文本
            has_image: 是否包含图片
        Returns:
            是否删除成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(dream_text, has_image)
            deleted = self.redis_client.delete(cache_key)
            if deleted:
                print(f"[缓存] 已删除缓存: {cache_key[:50]}...")
            return bool(deleted)
            
        except Exception as e:
            print(f"[警告] 删除缓存失败: {e}")
            return False
    
    def clear_all_analysis_cache(self) -> int:
        """
        清除所有分析结果缓存
        Returns:
            删除的缓存数量
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            # 查找所有以 dream:analysis: 开头的键
            pattern = "dream:analysis:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                print(f"[缓存] 已清除 {deleted} 条分析缓存")
                return deleted
            return 0
            
        except Exception as e:
            print(f"[警告] 清除缓存失败: {e}")
            return 0
    
    # ========== 图像生成结果缓存 ==========
    
    def get_image_generation_cache(self, dream_text: str) -> Optional[Dict[str, Any]]:
        """
        获取图像生成结果缓存
        Args:
            dream_text: 梦境文本
        Returns:
            缓存的图像生成结果，如果不存在则返回 None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            text_hash = hashlib.md5(dream_text.encode('utf-8')).hexdigest()
            cache_key = f"dream:image_gen:{text_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                print(f"[缓存] 命中图像生成缓存: {cache_key[:50]}...")
                return result
            
            return None
            
        except Exception as e:
            print(f"[警告] 读取图像生成缓存失败: {e}")
            return None
    
    def set_image_generation_cache(
        self,
        dream_text: str,
        image_result: Dict[str, Any],
        ttl: int = 30 * 24 * 60 * 60  # 默认30天
    ) -> bool:
        """
        设置图像生成结果缓存
        Args:
            dream_text: 梦境文本
            image_result: 图像生成结果字典
            ttl: 缓存过期时间（秒），默认30天
        Returns:
            是否设置成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            text_hash = hashlib.md5(dream_text.encode('utf-8')).hexdigest()
            cache_key = f"dream:image_gen:{text_hash}"
            cache_data = json.dumps(image_result, ensure_ascii=False)
            self.redis_client.setex(cache_key, ttl, cache_data)
            print(f"[缓存] 已缓存图像生成结果: {cache_key[:50]}... (TTL: {ttl}秒)")
            return True
            
        except Exception as e:
            print(f"[警告] 写入图像生成缓存失败: {e}")
            return False
    
    # ========== 历史记录列表缓存 ==========
    
    def get_history_cache(self, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        获取历史记录列表缓存
        Args:
            limit: 记录数量限制
        Returns:
            缓存的历史记录列表，如果不存在则返回 None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = f"dream:history:{limit}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                print(f"[缓存] 命中历史记录缓存: limit={limit}")
                return result
            
            return None
            
        except Exception as e:
            print(f"[警告] 读取历史记录缓存失败: {e}")
            return None
    
    def set_history_cache(
        self,
        limit: int,
        history_data: Dict[str, Any],
        ttl: int = 5 * 60  # 默认5分钟
    ) -> bool:
        """
        设置历史记录列表缓存
        Args:
            limit: 记录数量限制
            history_data: 历史记录数据
            ttl: 缓存过期时间（秒），默认5分钟
        Returns:
            是否设置成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = f"dream:history:{limit}"
            cache_data = json.dumps(history_data, ensure_ascii=False)
            self.redis_client.setex(cache_key, ttl, cache_data)
            print(f"[缓存] 已缓存历史记录列表: limit={limit} (TTL: {ttl}秒)")
            return True
            
        except Exception as e:
            print(f"[警告] 写入历史记录缓存失败: {e}")
            return False
    
    def invalidate_history_cache(self, limit: Optional[int] = None) -> int:
        """
        使历史记录缓存失效
        Args:
            limit: 如果指定，只清除该 limit 的缓存；如果为 None，清除所有历史记录缓存
        Returns:
            删除的缓存数量
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            if limit is not None:
                cache_key = f"dream:history:{limit}"
                deleted = self.redis_client.delete(cache_key)
                if deleted:
                    print(f"[缓存] 已清除历史记录缓存: limit={limit}")
                return deleted
            else:
                # 清除所有历史记录缓存
                pattern = "dream:history:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    print(f"[缓存] 已清除 {deleted} 条历史记录缓存")
                    return deleted
            return 0
            
        except Exception as e:
            print(f"[警告] 清除历史记录缓存失败: {e}")
            return 0
    
    # ========== 梦境详情缓存 ==========
    
    def get_detail_cache(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        获取梦境详情缓存
        Args:
            entry_id: 记录ID
        Returns:
            缓存的梦境详情，如果不存在则返回 None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = f"dream:detail:{entry_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                print(f"[缓存] 命中梦境详情缓存: entry_id={entry_id}")
                return result
            
            return None
            
        except Exception as e:
            print(f"[警告] 读取梦境详情缓存失败: {e}")
            return None
    
    def set_detail_cache(
        self,
        entry_id: int,
        detail_data: Dict[str, Any],
        ttl: int = 60 * 60  # 默认1小时
    ) -> bool:
        """
        设置梦境详情缓存
        Args:
            entry_id: 记录ID
            detail_data: 详情数据
            ttl: 缓存过期时间（秒），默认1小时
        Returns:
            是否设置成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = f"dream:detail:{entry_id}"
            cache_data = json.dumps(detail_data, ensure_ascii=False)
            self.redis_client.setex(cache_key, ttl, cache_data)
            print(f"[缓存] 已缓存梦境详情: entry_id={entry_id} (TTL: {ttl}秒)")
            return True
            
        except Exception as e:
            print(f"[警告] 写入梦境详情缓存失败: {e}")
            return False
    
    def invalidate_detail_cache(self, entry_id: int) -> bool:
        """
        使梦境详情缓存失效
        Args:
            entry_id: 记录ID
        Returns:
            是否删除成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = f"dream:detail:{entry_id}"
            deleted = self.redis_client.delete(cache_key)
            if deleted:
                print(f"[缓存] 已清除梦境详情缓存: entry_id={entry_id}")
            return bool(deleted)
            
        except Exception as e:
            print(f"[警告] 清除梦境详情缓存失败: {e}")
            return False
    
    # ========== 综合分析结果缓存 ==========
    
    def get_comprehensive_analysis_cache(self, entry_ids: List[int]) -> Optional[Dict[str, Any]]:
        """
        获取综合分析结果缓存
        Args:
            entry_ids: 记录ID列表
        Returns:
            缓存的综合分析结果，如果不存在则返回 None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # 对 entry_ids 排序并生成哈希，确保相同组合的ID生成相同的键
            sorted_ids = sorted(entry_ids)
            ids_str = ','.join(map(str, sorted_ids))
            ids_hash = hashlib.md5(ids_str.encode('utf-8')).hexdigest()
            cache_key = f"dream:comprehensive:{ids_hash}"
            
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                print(f"[缓存] 命中综合分析缓存: {len(entry_ids)} 条记录")
                return result
            
            return None
            
        except Exception as e:
            print(f"[警告] 读取综合分析缓存失败: {e}")
            return None
    
    def set_comprehensive_analysis_cache(
        self,
        entry_ids: List[int],
        analysis_result: Dict[str, Any],
        ttl: int = 24 * 60 * 60  # 默认1天
    ) -> bool:
        """
        设置综合分析结果缓存
        Args:
            entry_ids: 记录ID列表
            analysis_result: 综合分析结果
            ttl: 缓存过期时间（秒），默认1天
        Returns:
            是否设置成功
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # 对 entry_ids 排序并生成哈希
            sorted_ids = sorted(entry_ids)
            ids_str = ','.join(map(str, sorted_ids))
            ids_hash = hashlib.md5(ids_str.encode('utf-8')).hexdigest()
            cache_key = f"dream:comprehensive:{ids_hash}"
            
            cache_data = json.dumps(analysis_result, ensure_ascii=False)
            self.redis_client.setex(cache_key, ttl, cache_data)
            print(f"[缓存] 已缓存综合分析结果: {len(entry_ids)} 条记录 (TTL: {ttl}秒)")
            return True
            
        except Exception as e:
            print(f"[警告] 写入综合分析缓存失败: {e}")
            return False
    
    def invalidate_comprehensive_analysis_cache(self, entry_id: Optional[int] = None) -> int:
        """
        使综合分析缓存失效
        Args:
            entry_id: 如果指定，清除包含该 entry_id 的所有综合分析缓存；如果为 None，清除所有
        Returns:
            删除的缓存数量
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            if entry_id is not None:
                # 查找所有包含该 entry_id 的缓存（需要遍历所有键）
                pattern = "dream:comprehensive:*"
                keys = self.redis_client.keys(pattern)
                deleted_count = 0
                for key in keys:
                    # 检查缓存值中是否包含该 entry_id
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        try:
                            data = json.loads(cached_data)
                            # 如果缓存的数据中包含该 entry_id，删除缓存
                            # 这里简化处理：清除所有综合分析缓存
                            deleted = self.redis_client.delete(key)
                            if deleted:
                                deleted_count += 1
                        except:
                            pass
                if deleted_count > 0:
                    print(f"[缓存] 已清除包含 entry_id={entry_id} 的综合分析缓存")
                return deleted_count
            else:
                # 清除所有综合分析缓存
                pattern = "dream:comprehensive:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    print(f"[缓存] 已清除 {deleted} 条综合分析缓存")
                    return deleted
            return 0
            
        except Exception as e:
            print(f"[警告] 清除综合分析缓存失败: {e}")
            return 0
    
    def is_available(self) -> bool:
        """
        检查 Redis 是否可用
        Returns:
            True 如果 Redis 可用，False 否则
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # 尝试 ping 操作
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取 Redis 缓存状态信息
        Returns:
            包含状态信息的字典
        """
        status = {
            "redis_library_installed": REDIS_AVAILABLE,
            "cache_enabled": self.enabled,
            "redis_available": False,
            "connection_info": None,
            "error": None
        }
        
        if not REDIS_AVAILABLE:
            status["error"] = "Redis Python 库未安装"
            return status
        
        if not self.enabled:
            status["error"] = "Redis 连接未启用"
            return status
        
        try:
            # 测试连接
            self.redis_client.ping()
            status["redis_available"] = True
            
            # 获取连接信息
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            redis_db = int(os.environ.get("REDIS_DB", 0))
            
            status["connection_info"] = {
                "host": redis_host,
                "port": redis_port,
                "db": redis_db
            }
            
            # 获取一些统计信息
            try:
                info = self.redis_client.info()
                status["redis_info"] = {
                    "version": info.get("redis_version", "unknown"),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0)
                }
            except Exception:
                pass
                
        except Exception as e:
            status["redis_available"] = False
            status["error"] = str(e)
        
        return status


# 全局缓存实例
_cache_instance = None

def get_cache() -> RedisCache:
    """获取全局缓存实例（单例模式）"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

