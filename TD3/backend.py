"""
苹果配送机器人后端服务器 - WebSocket + HTTP服务
"""

import os
import sys
import json
import time
import asyncio
import threading
import base64
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol
import aiohttp
from aiohttp import web
import cv2

from config import Config
from robot_controller import AppleDeliveryRobotController, OperatingMode
from vision_recognizer import AppleVisionRecognizer


class RobotWebSocketServer:
    """机器人WebSocket服务器"""
    
    def __init__(self, robot_controller: AppleDeliveryRobotController):
        self.robot_controller = robot_controller
        self.clients: List[WebSocketServerProtocol] = []
        self.client_data: Dict[WebSocketServerProtocol, Dict] = {}
        
        # 状态更新任务
        self.status_update_task = None
        
        # 图像识别队列
        self.image_queue = asyncio.Queue()
        self.image_processing_task = None
        
        # 语音命令队列
        self.voice_command_queue = asyncio.Queue()
        
        print("✅ WebSocket服务器初始化完成")
    
    async def register_client(self, websocket: WebSocketServerProtocol):
        """注册客户端"""
        self.clients.append(websocket)
        self.client_data[websocket] = {
            'id': id(websocket),
            'connected_at': time.time(),
            'last_active': time.time()
        }
        print(f"📱 客户端连接: {len(self.clients)} 个客户端")
        
        # 发送欢迎消息
        welcome_msg = {
            'type': 'system',
            'message': '机器人系统连接成功',
            'timestamp': time.time(),
            'robot_status': self.get_robot_status_dict()
        }
        await websocket.send(json.dumps(welcome_msg))
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """注销客户端"""
        self.clients.remove(websocket)
        del self.client_data[websocket]
        print(f"📱 客户端断开: {len(self.clients)} 个客户端")
    
    def get_robot_status_dict(self) -> Dict:
        """获取机器人状态字典"""
        status = self.robot_controller.status
        return {
            'mode': status.mode.value,
            'position': {
                'x': float(status.position[0]),
                'y': float(status.position[1]),
                'theta': float(status.position[2])
            },
            'velocity': {
                'linear': float(status.velocity[0]),
                'angular': float(status.velocity[1])
            },
            'battery': float(status.battery_level),
            'current_task': status.current_task,
            'navigation_target': status.navigation_target,
            'navigation_progress': float(status.navigation_progress),
            'obstacles_detected': len(status.obstacles_detected),
            'last_update': status.last_update
        }
    
    async def broadcast_status(self):
        """广播机器人状态"""
        if not self.clients:
            return
        
        status_msg = {
            'type': 'status_update',
            'timestamp': time.time(),
            'data': self.get_robot_status_dict()
        }
        
        message = json.dumps(status_msg)
        tasks = [client.send(message) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            self.client_data[websocket]['last_active'] = time.time()
            
            if msg_type == 'command':
                await self.handle_command(websocket, data)
            elif msg_type == 'image_upload':
                await self.handle_image_upload(websocket, data)
            elif msg_type == 'training_start':
                await self.handle_training_start(websocket, data)
            elif msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
            
        except json.JSONDecodeError as e:
            error_msg = {'type': 'error', 'message': f'JSON解析错误: {str(e)}'}
            await websocket.send(json.dumps(error_msg))
        except Exception as e:
            error_msg = {'type': 'error', 'message': f'消息处理错误: {str(e)}'}
            await websocket.send(json.dumps(error_msg))
    
    async def handle_command(self, websocket: WebSocketServerProtocol, data: Dict):
        """处理控制命令"""
        command = data.get('command')
        params = data.get('params', {})
        
        print(f"🎮 收到命令: {command}, 参数: {params}")
        
        response = {'type': 'command_response', 'command': command, 'success': False}
        
        try:
            if command == 'start_voice_control':
                success = self.robot_controller.start_voice_control()
                response['success'] = success
                response['message'] = '语音控制启动成功' if success else '语音控制启动失败'
            
            elif command == 'stop_voice_control':
                self.robot_controller.stop_voice_control()
                response['success'] = True
                response['message'] = '语音控制已停止'
            
            elif command == 'navigate_to_shelf':
                shelf = params.get('shelf')
                if shelf:
                    success = self.robot_controller.navigate_to_shelf(shelf)
                    response['success'] = success
                    response['message'] = f'开始导航到{shelf}' if success else f'导航到{shelf}失败'
            
            elif command == 'return_to_start':
                success = self.robot_controller.return_to_start()
                response['success'] = success
                response['message'] = '开始返回起点' if success else '返回起点失败'
            
            elif command == 'go_to_charging':
                success = self.robot_controller.go_to_charging()
                response['success'] = success
                response['message'] = '开始前往充电站' if success else '前往充电站失败'
            
            elif command == 'stop_navigation':
                self.robot_controller.stop_navigation()
                response['success'] = True
                response['message'] = '导航已停止'
            
            elif command == 'recognize_image':
                image_data = params.get('image_data')
                if image_data:
                    await self.image_queue.put((websocket, image_data))
                    response['success'] = True
                    response['message'] = '图像已加入处理队列'
                else:
                    response['message'] = '缺少图像数据'
            
            elif command == 'start_training':
                episodes = params.get('episodes', 10)
                success = self.robot_controller.start_training(episodes)
                response['success'] = success
                response['message'] = f'开始训练 {episodes} 回合' if success else '训练启动失败'
            
            elif command == 'get_status':
                response.update({
                    'type': 'status_response',
                    'data': self.get_robot_status_dict(),
                    'success': True
                })
            
            else:
                response['message'] = f'未知命令: {command}'
            
        except Exception as e:
            response['message'] = f'命令执行错误: {str(e)}'
        
        await websocket.send(json.dumps(response))
    
    async def handle_image_upload(self, websocket: WebSocketServerProtocol, data: Dict):
        """处理图像上传"""
        try:
            image_data = data.get('image_data')
            if not image_data:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': '缺少图像数据'
                }))
                return
            
            # 解码base64图像
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': '图像解码失败'
                }))
                return
            
            # 保存临时图像文件
            temp_path = f"temp_image_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, image)
            
            # 识别苹果
            result = self.robot_controller.recognize_apple_from_image(temp_path)
            
            if result:
                apple_class, confidence, details = result
                
                # 读取处理后的图像
                processed_image_path = f"temp_processed_{int(time.time())}.jpg"
                if self.robot_controller.vision_recognizer:
                    processed_img = self.robot_controller.vision_recognizer.visualize_recognition(
                        temp_path, (apple_class, confidence, details)
                    )
                    cv2.imwrite(processed_image_path, processed_img)
                
                # 转换为base64
                with open(processed_image_path, 'rb') as f:
                    processed_image_data = base64.b64encode(f.read()).decode('utf-8')
                
                # 发送识别结果
                recognition_result = {
                    'type': 'recognition_result',
                    'apple_class': apple_class,
                    'confidence': float(confidence),
                    'num_detections': details.get('num_detections', 0),
                    'class_distribution': details.get('class_distribution', {}),
                    'processed_image': f"data:image/jpeg;base64,{processed_image_data}",
                    'shelf': Config.apple_to_shelf.get(apple_class) if apple_class in Config.apple_to_shelf else None,
                    'timestamp': time.time()
                }
                
                await websocket.send(json.dumps(recognition_result))
                
                # 清理临时文件
                os.remove(temp_path)
                if os.path.exists(processed_image_path):
                    os.remove(processed_image_path)
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'图像处理错误: {str(e)}'
            }))
    
    async def handle_training_start(self, websocket: WebSocketServerProtocol, data: Dict):
        """处理训练开始"""
        params = data.get('params', {})
        episodes = params.get('episodes', 10)
        
        # 在后台线程中启动训练
        def start_training():
            try:
                self.robot_controller.start_training(episodes)
            except Exception as e:
                print(f"训练启动错误: {e}")
        
        training_thread = threading.Thread(target=start_training, daemon=True)
        training_thread.start()
        
        await websocket.send(json.dumps({
            'type': 'training_started',
            'message': f'开始训练 {episodes} 回合',
            'episodes': episodes,
            'timestamp': time.time()
        }))
    
    async def start_status_updates(self):
        """开始状态更新"""
        while True:
            try:
                await self.broadcast_status()
                await asyncio.sleep(0.5)  # 每500ms更新一次
            except Exception as e:
                print(f"状态更新错误: {e}")
                await asyncio.sleep(1)
    
    async def handle_voice_commands(self):
        """处理语音命令"""
        while True:
            try:
                # 检查语音命令
                if hasattr(self.robot_controller, 'voice_control'):
                    voice_cmd = self.robot_controller.voice_control.get_command(timeout=0.1)
                    if voice_cmd:
                        keyword, cmd_type = voice_cmd
                        
                        # 广播语音命令
                        voice_msg = {
                            'type': 'voice_command',
                            'keyword': keyword,
                            'command_type': cmd_type,
                            'timestamp': time.time()
                        }
                        
                        message = json.dumps(voice_msg)
                        tasks = [client.send(message) for client in self.clients]
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"语音命令处理错误: {e}")
                await asyncio.sleep(1)
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """WebSocket处理函数"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.process_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print("客户端连接关闭")
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """启动WebSocket服务器"""
        print(f"🚀 启动WebSocket服务器: ws://{Config.web_host}:{Config.websocket_port}")
        
        # 启动状态更新任务
        self.status_update_task = asyncio.create_task(self.start_status_updates())
        
        # 启动语音命令处理任务
        voice_task = asyncio.create_task(self.handle_voice_commands())
        
        # 启动WebSocket服务器
        server = await websockets.serve(
            self.handler,
            Config.web_host,
            Config.websocket_port
        )
        
        await server.wait_closed()


class RobotHTTPServer:
    """机器人HTTP服务器"""
    
    def __init__(self, robot_controller: AppleDeliveryRobotController):
        self.robot_controller = robot_controller
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """设置路由"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/config', self.handle_config)
        self.app.router.add_post('/command', self.handle_command)
        self.app.router.add_post('/upload_image', self.handle_upload_image)
        self.app.router.add_static('/static', 'frontend')
        self.app.router.add_static('/', 'frontend')  # 根目录也指向前端
    
    async def handle_index(self, request):
        """处理首页请求"""
        return web.FileResponse('frontend/index.html')
    
    async def handle_status(self, request):
        """处理状态请求"""
        status = self.robot_controller.status
        status_dict = {
            'mode': status.mode.value,
            'position': status.position,
            'velocity': status.velocity,
            'battery': status.battery_level,
            'current_task': status.current_task,
            'navigation_target': status.navigation_target,
            'navigation_progress': status.navigation_progress,
            'obstacles_detected': len(status.obstacles_detected),
            'timestamp': time.time()
        }
        return web.json_response(status_dict)
    
    async def handle_config(self, request):
        """处理配置请求"""
        config_dict = {
            'apple_to_shelf': Config.apple_to_shelf,
            'shelf_locations': Config.shelf_locations,
            'apple_classes': Config.apple_classes,
            'max_linear_speed': Config.max_linear_speed,
            'max_angular_speed': Config.max_angular_speed
        }
        return web.json_response(config_dict)
    
    async def handle_command(self, request):
        """处理命令请求"""
        try:
            data = await request.json()
            command = data.get('command')
            params = data.get('params', {})
            
            response = {'success': False, 'message': ''}
            
            if command == 'navigate_to_shelf':
                shelf = params.get('shelf')
                if shelf:
                    success = self.robot_controller.navigate_to_shelf(shelf)
                    response['success'] = success
                    response['message'] = f'导航到{shelf}'
            
            elif command == 'return_to_start':
                success = self.robot_controller.return_to_start()
                response['success'] = success
                response['message'] = '返回起点'
            
            elif command == 'stop':
                self.robot_controller.stop_navigation()
                response['success'] = True
                response['message'] = '停止'
            
            elif command == 'start_voice':
                success = self.robot_controller.start_voice_control()
                response['success'] = success
                response['message'] = '启动语音控制'
            
            elif command == 'stop_voice':
                self.robot_controller.stop_voice_control()
                response['success'] = True
                response['message'] = '停止语音控制'
            
            else:
                response['message'] = f'未知命令: {command}'
            
            return web.json_response(response)
            
        except Exception as e:
            return web.json_response({'success': False, 'message': str(e)})
    
    async def handle_upload_image(self, request):
        """处理图像上传"""
        try:
            data = await request.post()
            image_file = data.get('image')
            
            if not image_file:
                return web.json_response({'success': False, 'message': '没有上传图像'})
            
            # 保存临时文件
            temp_path = f"temp_upload_{int(time.time())}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(image_file.file.read())
            
            # 识别苹果
            result = self.robot_controller.recognize_apple_from_image(temp_path)
            
            if result:
                apple_class, confidence, details = result
                
                # 处理图像
                processed_path = f"temp_processed_{int(time.time())}.jpg"
                if self.robot_controller.vision_recognizer:
                    processed_img = self.robot_controller.vision_recognizer.visualize_recognition(
                        temp_path, (apple_class, confidence, details)
                    )
                    cv2.imwrite(processed_path, processed_img)
                    
                    # 读取处理后的图像
                    with open(processed_path, 'rb') as f:
                        processed_data = base64.b64encode(f.read()).decode('utf-8')
                else:
                    processed_data = None
                
                # 清理临时文件
                os.remove(temp_path)
                if os.path.exists(processed_path):
                    os.remove(processed_path)
                
                response = {
                    'success': True,
                    'apple_class': apple_class,
                    'confidence': float(confidence),
                    'num_detections': details.get('num_detections', 0),
                    'class_distribution': details.get('class_distribution', {}),
                    'processed_image': processed_data,
                    'shelf': Config.apple_to_shelf.get(apple_class) if apple_class in Config.apple_to_shelf else None
                }
                
                return web.json_response(response)
            else:
                return web.json_response({'success': False, 'message': '识别失败'})
            
        except Exception as e:
            return web.json_response({'success': False, 'message': str(e)})
    
    async def start(self):
        """启动HTTP服务器"""
        print(f"🌐 启动HTTP服务器: http://{Config.web_host}:{Config.web_port}")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, Config.web_host, Config.web_port)
        await site.start()
        
        # 保持服务器运行
        await asyncio.Event().wait()


async def main():
    """主函数"""
    print("=" * 70)
    print("🍎 苹果配送机器人系统 - Web版本")
    print("=" * 70)
    
    # 创建机器人控制器
    robot_controller = AppleDeliveryRobotController()
    
    # 创建Web服务器
    http_server = RobotHTTPServer(robot_controller)
    websocket_server = RobotWebSocketServer(robot_controller)
    
    # 启动服务器
    await asyncio.gather(
        http_server.start(),
        websocket_server.start()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
