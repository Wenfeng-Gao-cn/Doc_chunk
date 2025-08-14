# app_service.py 
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from app_RAG_V3 import run_app
import yaml 
from typing import Optional
import uuid
import json
import time
import logging
import sys
import signal
from functools import partial
from Utils.logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)

# 优雅退出处理
def handle_exit(signum, frame, app):
    logger.info("Received exit signal, shutting down...")
    sys.exit(0)

# 固定API key
FIXED_API_KEY = "sk-test-1234567890abcdef1234567890abcdef"

async def verify_api_key(api_key: Optional[str] = Header(default=None)):
    if api_key != FIXED_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API Key"
        )

app = FastAPI()

# 新增：读取配置文件
def load_config():
    config_path = 'setup.yaml'
    customer_prompt_file = "custom_prompt_sample.md"
    default_config = {
        "batch_size": 10,
        "include_answers": False,
        "max_retries": 3,
        "customer_prompt": "",
        "stream_mode": True  # 新增默认流式模式
    }
    
    # 读取用户自定义提示词
    try:
        with open(customer_prompt_file, "r", encoding="utf-8") as f:
            default_config["customer_prompt"] = f.read()
    except FileNotFoundError:
        logger.warning(f"用户个性化提示词样例文件不存在: {customer_prompt_file}")
    except Exception as e:
        logger.error(f"读取用户提示词文件失败: {str(e)}")
    
    # 读取主配置文件
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                logger.error("配置文件格式错误: 不是有效的字典格式")
                return default_config
                
            match_config = data.get("Match_QA_config", {})
            if not isinstance(match_config, dict):
                logger.error("Match_QA_config格式错误: 不是有效的字典格式")
                return default_config
                
            # 合并配置，确保所有必需字段都有值
            return {
                "batch_size": match_config.get("batch_size", default_config["batch_size"]),
                "include_answers": match_config.get("include_answers", default_config["include_answers"]),
                "max_retries": match_config.get("max_retries", default_config["max_retries"]),
                "customer_prompt": default_config["customer_prompt"],
                "stream_mode": match_config.get("stream_mode", default_config["stream_mode"])  # 新增
            }
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_path}")
        return default_config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return default_config

# 新增：构建OpenAI兼容的响应块
def _build_openai_chunk(model: str, content: str, finish_reason: str = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    # 记录新请求
    request_id = str(uuid.uuid4())
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"New request {request_id} from {client_host}")
    
    # 验证API key
    api_key = authorization.split("Bearer ")[1] if authorization else None
    await verify_api_key(api_key)
    
    # 检查是否为Cherry Studio测试请求
    try:
        body = await request.json()
        if (body.get("model") == "FAQmatch" and 
            isinstance(body.get("messages"), list) and 
            len(body["messages"]) > 0 and 
            body["messages"][0].get("content") == "hi"):
            model = body.get("model", "FAQmatch")
            return StreamingResponse(
                iter([_build_openai_chunk(model, "hi", "stop")]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
    except:
        pass
    
    try:
        # 解析并验证请求体
        try:
            body = await request.json()
            if not isinstance(body, dict):
                logger.error(f"Invalid request body type: {type(body)}")
                raise ValueError("请求体必须是JSON对象")
                
            messages = body.get("messages")
            if not messages or not isinstance(messages, list):
                logger.error(f"Invalid messages field: {messages}")
                raise ValueError("messages字段必须是非空数组")
                
            model = body.get("model", "FAQmatch")
            if not isinstance(model, str):
                logger.error(f"Invalid model type: {type(model)}")
                raise ValueError("model字段必须是字符串")
            
            # 分离system提示和用户问题
            system_prompt = ""
            user_question = ""
            
            # 收集所有system提示
            for message in messages:
                if message.get("role") == "system" and message.get("content"):
                    system_prompt += message["content"] + "\n"
            
            # 获取最后一条用户消息
            for message in reversed(messages):
                if message.get("role") == "user" and message.get("content"):
                    user_question = message["content"]
                    break
            
            # 如果没有找到用户问题，尝试使用最后一条消息
            if not user_question and messages:
                last_message = messages[-1]
                if last_message.get("content"):
                    user_question = last_message["content"]
            
            # 清除多余的换行
            system_prompt = system_prompt.strip()
            
            if not user_question.strip():
                logger.error("No valid user question found")
                raise ValueError("必须提供有效的用户问题内容")
            
        except ValueError as ve:
            logger.error(f"Request validation failed: {str(ve)}")
            raise HTTPException(status_code=422, detail=str(ve))
      
        config = load_config()        
        # 构建graph_setup所需的输入
        # 不再需要inputs字典，直接使用user_question和system_prompt
        
        # 使用app_RAG.run_app生成流式响应
        async def event_stream():
            try:
                async for chunk in run_app(
                    question=user_question,
                    system_prompt=system_prompt
                ):
                    # 确保chunk是字符串
                    content = str(chunk) if chunk is not None else ""
                    # 使用OpenAI兼容格式返回内容
                    yield _build_openai_chunk(model, content)
                    
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield _build_openai_chunk(model, f"\n[处理错误: {str(e)}]")
            finally:
                # 发送结束标记
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
      
    except Exception as e:
        logger.error(
            f"Request {request_id} failed - Client: {request.client.host if request.client else 'unknown'} - "
            f"Path: {request.url.path} - Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=400,
            detail="请求处理失败，请稍后重试"
        )

if __name__ == "__main__":
    import uvicorn
    # 设置信号处理
    signal.signal(signal.SIGTERM, partial(handle_exit, app=app))
    signal.signal(signal.SIGINT, partial(handle_exit, app=app))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_config=None,
        timeout_keep_alive=60,
        access_log=False
    )
