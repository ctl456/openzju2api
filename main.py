import asyncio
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import httpx
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class ModelConfig:
    def __init__(self, appkey: str, model_name: str):
        self.appkey = appkey
        self.model_name = model_name
        self.hiagent_csrf = None
        self.x_csrf_token = None
        self.app_conversation_id = None
        self.conversation_id = None
        self.base_url = f"https://open.zju.edu.cn/product/llm/chat/{appkey}"


# 模型配置
MODELS = {
    "DeepSeek-V3-671B": ModelConfig("cu3m3jodetcjeetbl2f0", "DeepSeek-V3-671B"),
    "DeepSeek-R1-671B": ModelConfig("cumma920bha5qsc294vg", "DeepSeek-R1-671B")
}

# 通用请求头
BASE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0',
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'app-visitor-key': 'cuvd8r365lasv2ploq10',
    'Origin': 'https://open.zju.edu.cn',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Priority': 'u=0',
}


# 请求模型
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]


async def get_headers(model_config: ModelConfig, is_stream: bool = False) -> dict:
    """生成请求头"""
    headers = BASE_HEADERS.copy()
    headers.update({
        'x-csrf-token': model_config.x_csrf_token,
        'Content-Type': 'application/json; charset=utf-8',
        'Referer': model_config.base_url,
    })
    if is_stream:
        headers['Accept'] = 'application/json, text/event-stream'
        headers['timeout'] = '300000'
    return headers


async def get_cookies(model_config: ModelConfig) -> dict:
    """生成cookies"""
    return {
        'hiagent-csrf': model_config.hiagent_csrf,
        'x-csrf-token': model_config.x_csrf_token,
    }


async def get_cookie() -> None:
    """获取所有模型的cookie"""
    try:
        for model_config in MODELS.values():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                response = await client.get(model_config.base_url, headers=BASE_HEADERS)
                model_config.hiagent_csrf = response.cookies.get('hiagent-csrf')
                model_config.x_csrf_token = response.cookies.get('x-csrf-token')
                logger.info(f"获取 {model_config.model_name} cookie成功")
    except httpx.RequestError as e:
        logger.error(f"获取cookie错误: {e}")
        raise HTTPException(status_code=500, detail=f"获取cookie失败: {str(e)}")


async def get_app_conversation_id() -> None:
    """获取所有模型的会话ID"""
    try:
        for model_config in MODELS.values():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                headers = await get_headers(model_config)
                cookies = await get_cookies(model_config)
                json_data = {
                    'AppKey': model_config.appkey,
                    'Inputs': {},
                }
                response = await client.post(
                    'https://open.zju.edu.cn/api/proxy/chat/v2/create_conversation',
                    cookies=cookies,
                    headers=headers,
                    json=json_data
                )
                response.raise_for_status()
                model_config.app_conversation_id = json.loads(response.text)['Conversation']['AppConversationID']
                logger.info(f"获取 {model_config.model_name} AppConversationID成功")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"获取AppConversationID错误: {e}")
        raise HTTPException(status_code=500, detail=f"获取AppConversationID失败: {str(e)}")




async def process_response_stream(response: httpx.Response, model_config: ModelConfig) -> AsyncGenerator[str, None]:
    """处理流式响应"""
    full_content = ""
    is_first_chunk = True

    async for line in response.aiter_lines():
        if not line or line.strip() == "":
            continue

        if line.startswith("event:text"):
            continue

        if line.startswith("data:data:"):
            json_str = line[10:].strip()
            try:
                data = json.loads(json_str)

                if model_config.conversation_id is None:
                    model_config.conversation_id = data.get("conversation_id")

                if data.get("event") == "message":
                    # 将内容按字符分割，实现逐字输出
                    content = data.get("answer", "")
                    for char in content:
                        chunk = {
                            "id": f"chatcmpl-{data.get('id', '')}",
                            "object": "chat.completion.chunk",
                            "created": int(data.get("created_at", time.time())),
                            "model": model_config.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": char
                                } if not is_first_chunk else {
                                    "role": "assistant",
                                    "content": char
                                },
                                "finish_reason": None
                            }]
                        }
                        is_first_chunk = False
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        # 添加小延迟以实现更自然的流式效果
                        await asyncio.sleep(0.02)

                elif data.get("event") == "message_end":
                    end_chunk = {
                        "id": f"chatcmpl-{data.get('id', '')}",
                        "object": "chat.completion.chunk",
                        "created": int(data.get("created_at", time.time())),
                        "model": model_config.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                continue


async def generate_response(messages: List[dict], model: str, temperature: float, stream: bool,
                            max_tokens: Optional[int] = None, presence_penalty: float = 0,
                            frequency_penalty: float = 0, top_p: float = 1.0):
    """生成响应"""
    model_config = MODELS.get(model)
    if not model_config:
        raise HTTPException(status_code=400, detail="不支持的模型")

    headers = await get_headers(model_config, True)
    cookies = await get_cookies(model_config)

    json_data = {
        'Query': messages[-1]['content'],
        'AppConversationID': model_config.app_conversation_id,
        'AppKey': model_config.appkey,
        'QueryExtends': {'Files': []},
    }

    if model_config.conversation_id:
        json_data['ConversationID'] = model_config.conversation_id

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900)) as client:
            response = await client.post(
                'https://open.zju.edu.cn/api/proxy/chat/v2/chat_query',
                cookies=cookies,
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()

            async for chunk in process_response_stream(response, model_config):
                yield chunk

    except httpx.RequestError as e:
        logger.error(f"生成响应错误: {e}")
        raise HTTPException(status_code=500, detail=f"请求错误: {str(e)}")


async def get_full_response(messages: List[dict], model: str, temperature: float, stream: bool,
                            max_tokens: Optional[int] = None, presence_penalty: float = 0,
                            frequency_penalty: float = 0, top_p: float = 1.0) -> str:
    """获取完整的非流式响应"""
    full_response = ""
    async for chunk in generate_response(messages, model, temperature, stream, max_tokens,
                                         presence_penalty, frequency_penalty, top_p):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if "choices" in data and data["choices"]:
                    if "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                        full_response += data["choices"][0]["delta"]["content"]
            except json.JSONDecodeError:
                continue
    return full_response


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表"""
    current_time = int(time.time())
    models_data = []

    for model_id, config in MODELS.items():
        models_data.append(
            ModelData(
                id=model_id,
                created=current_time,
                owned_by="zju",
                root=model_id,
                permission=[{
                    "id": f"modelperm-{model_id}",
                    "object": "model_permission",
                    "created": current_time,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "zju",
                    "group": None,
                    "is_blocking": False
                }]
            )
        )

    return {"object": "list", "data": models_data}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """处理聊天完成请求"""
    messages = [msg.model_dump() for msg in request.messages]

    if request.stream:
        return StreamingResponse(
            generate_response(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                stream=request.stream,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                top_p=request.top_p
            ),
            media_type="text/event-stream"
        )
    else:
        response_text = await get_full_response(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            stream=request.stream,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            top_p=request.top_p
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    try:
        await get_cookie()
        await get_app_conversation_id()
    except Exception as e:
        logger.error(f"启动初始化错误: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    try:
        for model_config in MODELS.values():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                if model_config.app_conversation_id:
                    headers = await get_headers(model_config)
                    cookies = await get_cookies(model_config)
                    json_data = {
                        'AppKey': model_config.appkey,
                        'AppConversationID': model_config.app_conversation_id,
                    }
                    response = await client.post(
                        'https://open.zju.edu.cn/api/proxy/chat/v2/delete_conversation',
                        cookies=cookies,
                        headers=headers,
                        json=json_data,
                    )
                    response.raise_for_status()
                    logger.info(f"清理 {model_config.model_name} 会话成功")
    except Exception as e:
        logger.error(f"关闭清理错误: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
