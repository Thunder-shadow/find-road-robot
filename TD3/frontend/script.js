/**
 * 苹果配送机器人前端JavaScript代码
 */

class RobotWebClient {
    constructor() {
        // WebSocket连接
        this.ws = null;
        this.reconnectTimeout = null;
        this.connected = false;
        
        // 状态数据
        this.robotStatus = {
            mode: '空闲',
            position: { x: 0, y: 0, theta: 0 },
            velocity: { linear: 0, angular: 0 },
            battery: 100,
            currentTask: '初始化',
            navigationTarget: null,
            navigationProgress: 0,
            obstaclesDetected: 0
        };
        
        // 训练数据
        this.trainingData = {
            rewards: [],
            losses: [],
            episodes: 0,
            successRate: 0
        };
        
        // 图表实例
        this.rewardChart = null;
        this.lossChart = null;
        this.laserChart = null;
        this.mapChart = null;
        
        // 配置
        this.config = {
            websocketUrl: `ws://${window.location.hostname}:8081`,
            statusUpdateInterval: 1000,
            maxLogLines: 1000
        };
        
        // 初始化
        this.init();
    }
    
    init() {
        // 设置事件监听器
        this.setupEventListeners();
        
        // 初始化图表
        this.initCharts();
        
        // 更新当前时间
        this.updateCurrentTime();
        setInterval(() => this.updateCurrentTime(), 1000);
        
        // 连接WebSocket
        this.connectWebSocket();
    }
    
    setupEventListeners() {
        // 系统控制
        document.getElementById('system-control').addEventListener('click', () => this.toggleSystem());
        
        // 视觉识别
        document.getElementById('btn-load-image').addEventListener('click', () => this.loadImage());
        document.getElementById('image-preview').addEventListener('click', () => this.loadImage());
        document.getElementById('btn-recognize').addEventListener('click', () => this.recognizeApple());
        
        // 语音控制
        document.getElementById('btn-start-voice').addEventListener('click', () => this.startVoiceControl());
        document.getElementById('btn-stop-voice').addEventListener('click', () => this.stopVoiceControl());
        
        // 导航控制
        document.getElementById('btn-go-shelf').addEventListener('click', () => this.goToShelf());
        document.getElementById('btn-return-home').addEventListener('click', () => this.returnHome());
        document.getElementById('btn-go-charge').addEventListener('click', () => this.goToCharge());
        document.getElementById('btn-stop-nav').addEventListener('click', () => this.stopNavigation());
        
        // 训练控制
        document.getElementById('btn-start-training').addEventListener('click', () => this.startTraining());
        document.getElementById('btn-stop-training').addEventListener('click', () => this.stopTraining());
        
        // 标签页切换
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // 日志控制
        document.getElementById('btn-clear-log').addEventListener('click', () => this.clearLog());
        document.getElementById('btn-save-log').addEventListener('click', () => this.saveLog());
        document.getElementById('log-level-filter').addEventListener('change', (e) => this.filterLog(e.target.value));
        
        // 地图控制
        document.getElementById('btn-refresh-map').addEventListener('click', () => this.refreshMap());
        document.getElementById('btn-save-map').addEventListener('click', () => this.saveMap());
        document.getElementById('map-zoom').addEventListener('input', (e) => this.updateZoom(e.target.value));
        
        // 设置控制
        document.getElementById('btn-save-settings').addEventListener('click', () => this.saveSettings());
        document.getElementById('btn-load-settings').addEventListener('click', () => this.loadSettings());
        document.getElementById('btn-reset-settings').addEventListener('click', () => this.resetSettings());
        document.getElementById('btn-browse-model').addEventListener('click', () => this.browseModel());
        
        // 模态对话框
        document.getElementById('modal-close').addEventListener('click', () => this.hideModal());
        document.getElementById('modal-cancel').addEventListener('click', () => this.hideModal());
        document.getElementById('modal-confirm').addEventListener('click', () => this.confirmModal());
        
        // 文件上传
        document.getElementById('file-input').addEventListener('change', (e) => this.handleImageUpload(e));
        
        // 拖放上传
        const imagePreview = document.getElementById('image-preview');
        imagePreview.addEventListener('dragover', (e) => {
            e.preventDefault();
            imagePreview.style.borderColor = '#667eea';
        });
        
        imagePreview.addEventListener('dragleave', (e) => {
            e.preventDefault();
            imagePreview.style.borderColor = '#ced4da';
        });
        
        imagePreview.addEventListener('drop', (e) => {
            e.preventDefault();
            imagePreview.style.borderColor = '#ced4da';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.loadImageFile(files[0]);
            }
        });
    }
    
    connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }
        
        this.updateConnectionStatus('正在连接...', 'warning');
        
        this.ws = new WebSocket(this.config.websocketUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket连接成功');
            this.connected = true;
            this.updateConnectionStatus('已连接', 'success');
            this.logMessage('WebSocket连接成功', 'info');
            
            // 请求初始状态
            this.sendCommand('get_status');
        };
        
        this.ws.onmessage = (event) => {
            this.handleWebSocketMessage(event.data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket连接关闭');
            this.connected = false;
            this.updateConnectionStatus('连接断开', 'error');
            
            // 尝试重新连接
            this.reconnectWebSocket();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.updateConnectionStatus('连接错误', 'error');
        };
    }
    
    reconnectWebSocket() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }
        
        this.reconnectTimeout = setTimeout(() => {
            console.log('尝试重新连接...');
            this.connectWebSocket();
        }, 3000);
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'status_update':
                    this.updateRobotStatus(message.data);
                    break;
                    
                case 'recognition_result':
                    this.handleRecognitionResult(message);
                    break;
                    
                case 'voice_command':
                    this.handleVoiceCommand(message);
                    break;
                    
                case 'training_started':
                    this.handleTrainingStarted(message);
                    break;
                    
                case 'command_response':
                    this.handleCommandResponse(message);
                    break;
                    
                case 'training_update':
                    this.handleTrainingUpdate(message);
                    break;
                    
                case 'error':
                    this.showError(message.message);
                    break;
                    
                case 'system':
                    this.logMessage(message.message, 'info');
                    break;
                    
                case 'pong':
                    // 心跳响应
                    break;
                    
                default:
                    console.log('未知消息类型:', message.type);
            }
        } catch (error) {
            console.error('消息解析错误:', error);
        }
    }
    
    sendCommand(command, params = {}) {
        if (!this.connected || !this.ws) {
            this.showError('未连接到服务器');
            return;
        }
        
        const message = {
            type: 'command',
            command: command,
            params: params,
            timestamp: Date.now()
        };
        
        this.ws.send(JSON.stringify(message));
    }
    
    updateRobotStatus(status) {
        this.robotStatus = status;
        
        // 更新界面显示
        document.getElementById('pos-x').textContent = `${status.position.x.toFixed(2)} m`;
        document.getElementById('pos-y').textContent = `${status.position.y.toFixed(2)} m`;
        document.getElementById('heading').textContent = `${(status.position.theta * 180 / Math.PI).toFixed(1)}°`;
        document.getElementById('linear-speed').textContent = `${status.velocity.linear.toFixed(2)} m/s`;
        document.getElementById('angular-speed').textContent = `${status.velocity.angular.toFixed(2)} rad/s`;
        document.getElementById('battery-level').textContent = `${status.battery.toFixed(1)}%`;
        document.getElementById('current-mode').textContent = status.mode;
        document.getElementById('current-task').textContent = status.currentTask;
        document.getElementById('obstacle-count').textContent = status.obstaclesDetected;
        
        // 更新导航进度
        const progress = Math.round(status.navigationProgress * 100);
        document.getElementById('navigation-progress').style.width = `${progress}%`;
        document.getElementById('progress-text').textContent = `${progress}%`;
        
        // 更新电池图标
        this.updateBatteryIcon(status.battery);
        
        // 更新激光雷达显示
        this.updateLaserDisplay();
        
        // 更新地图显示
        this.updateMapDisplay();
    }
    
    updateBatteryIcon(level) {
        const batteryEl = document.querySelector('.battery-status i');
        
        if (level > 75) {
            batteryEl.className = 'fas fa-battery-full';
        } else if (level > 50) {
            batteryEl.className = 'fas fa-battery-three-quarters';
        } else if (level > 25) {
            batteryEl.className = 'fas fa-battery-half';
        } else if (level > 10) {
            batteryEl.className = 'fas fa-battery-quarter';
        } else {
            batteryEl.className = 'fas fa-battery-empty';
        }
    }
    
    updateConnectionStatus(status, type) {
        const statusEl = document.getElementById('connection-status');
        const infoEl = document.getElementById('connection-info');
        
        statusEl.textContent = status;
        statusEl.className = `status-${type}`;
        infoEl.textContent = status;
        
        // 更新状态栏消息
        document.getElementById('status-message').textContent = status;
    }
    
    updateCurrentTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('zh-CN');
        document.getElementById('current-time').textContent = timeString;
    }
    
    // 视觉识别功能
    loadImage() {
        document.getElementById('file-input').click();
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadImageFile(file);
        }
    }
    
    loadImageFile(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const img = document.getElementById('preview-image');
            const placeholder = document.querySelector('.image-placeholder');
            
            img.src = e.target.result;
            img.classList.remove('hidden');
            placeholder.classList.add('hidden');
            
            // 启用识别按钮
            document.getElementById('btn-recognize').disabled = false;
        };
        
        reader.readAsDataURL(file);
    }
    
    recognizeApple() {
        const img = document.getElementById('preview-image');
        if (!img.src || img.classList.contains('hidden')) {
            this.showError('请先选择图像');
            return;
        }
        
        // 显示加载中
        this.showModal('识别中...', '请稍候，正在识别苹果...');
        
        // 发送识别命令
        this.sendCommand('recognize_image', {
            image_data: img.src
        });
    }
    
    handleRecognitionResult(result) {
        this.hideModal();
        
        // 更新识别结果
        document.getElementById('recognition-result').textContent = result.apple_class;
        document.getElementById('confidence-text').textContent = `${(result.confidence * 100).toFixed(1)}%`;
        document.getElementById('confidence-bar').style.width = `${result.confidence * 100}%`;
        document.getElementById('detection-count').textContent = result.num_detections;
        
        if (result.shelf) {
            document.getElementById('shelf-info').textContent = result.shelf;
        } else {
            document.getElementById('shelf-info').textContent = '无对应货架';
        }
        
        // 显示处理后的图像
        if (result.processed_image) {
            const img = document.getElementById('preview-image');
            img.src = result.processed_image;
        }
        
        // 记录识别日志
        this.logMessage(`识别结果: ${result.apple_class} (置信度: ${(result.confidence * 100).toFixed(1)}%, 检测到${result.num_detections}个)`, 'info');
        
        // 询问是否导航
        if (result.shelf) {
            this.showModal('识别成功', 
                `识别到${result.apple_class}，检测到${result.num_detections}个苹果\n\n是否导航到${result.shelf}？`,
                true,
                () => this.navigateToShelf(result.shelf, result.apple_class)
            );
        }
    }
    
    // 语音控制功能
    startVoiceControl() {
        this.sendCommand('start_voice_control');
        document.getElementById('btn-start-voice').disabled = true;
        document.getElementById('btn-stop-voice').disabled = false;
    }
    
    stopVoiceControl() {
        this.sendCommand('stop_voice_control');
        document.getElementById('btn-start-voice').disabled = false;
        document.getElementById('btn-stop-voice').disabled = true;
    }
    
    handleVoiceCommand(command) {
        const commandsList = document.getElementById('voice-commands-list');
        
        const commandItem = document.createElement('div');
        commandItem.className = 'command-item';
        commandItem.textContent = `${new Date(command.timestamp).toLocaleTimeString()} - ${command.keyword}`;
        
        commandsList.appendChild(commandItem);
        
        // 保持列表长度
        while (commandsList.children.length > 10) {
            commandsList.removeChild(commandsList.firstChild);
        }
        
        // 滚动到底部
        commandsList.scrollTop = commandsList.scrollHeight;
        
        this.logMessage(`语音命令: ${command.keyword}`, 'info');
    }
    
    // 导航控制功能
    goToShelf() {
        const shelfSelect = document.getElementById('shelf-select');
        const shelf = shelfSelect.value;
        
        this.navigateToShelf(shelf);
    }
    
    navigateToShelf(shelf, appleType = null) {
        this.hideModal();
        
        const params = { shelf: shelf };
        if (appleType) {
            params.apple = appleType;
        }
        
        this.sendCommand('navigate_to_shelf', params);
        this.logMessage(`开始导航到 ${shelf}`, 'info');
    }
    
    returnHome() {
        this.sendCommand('return_to_start');
        this.logMessage('开始返回起点', 'info');
    }
    
    goToCharge() {
        this.sendCommand('go_to_charging');
        this.logMessage('开始前往充电站', 'info');
    }
    
    stopNavigation() {
        this.sendCommand('stop_navigation');
        this.logMessage('停止导航', 'info');
    }
    
    // 训练控制功能
    startTraining() {
        const episodes = parseInt(document.getElementById('training-episodes').value);
        const learningRate = parseFloat(document.getElementById('learning-rate').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const gamma = parseFloat(document.getElementById('gamma').value);
        
        this.sendCommand('start_training', {
            episodes: episodes,
            learning_rate: learningRate,
            batch_size: batchSize,
            gamma: gamma
        });
        
        document.getElementById('btn-start-training').disabled = true;
        document.getElementById('btn-stop-training').disabled = false;
    }
    
    stopTraining() {
        // 这里需要实现停止训练的逻辑
        // 目前只是启用/禁用按钮
        document.getElementById('btn-start-training').disabled = false;
        document.getElementById('btn-stop-training').disabled = true;
        
        this.logMessage('训练已停止', 'info');
    }
    
    handleTrainingStarted(result) {
        this.logMessage(result.message, 'info');
    }
    
    handleTrainingUpdate(update) {
        // 更新训练数据
        this.trainingData.rewards.push(update.reward);
        this.trainingData.losses.push(update.loss);
        this.trainingData.episodes = update.episode;
        this.trainingData.successRate = update.success_rate;
        
        // 更新界面
        document.getElementById('current-episode').textContent = update.episode;
        document.getElementById('avg-reward').textContent = update.avg_reward.toFixed(2);
        document.getElementById('success-rate').textContent = `${(update.success_rate * 100).toFixed(1)}%`;
        document.getElementById('avg-loss').textContent = update.avg_loss.toFixed(4);
        
        // 更新图表
        this.updateTrainingCharts();
    }
    
    // 系统控制
    toggleSystem() {
        const btn = document.getElementById('system-control');
        
        if (btn.textContent.includes('启动')) {
            btn.innerHTML = '<i class="fas fa-power-off"></i> 停止系统';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-danger');
            this.logMessage('系统启动', 'info');
        } else {
            btn.innerHTML = '<i class="fas fa-power-off"></i> 启动系统';
            btn.classList.remove('btn-danger');
            btn.classList.add('btn-primary');
            this.logMessage('系统停止', 'info');
        }
    }
    
    // 标签页切换
    switchTab(tabId) {
        // 更新按钮状态
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
        
        // 显示对应标签页
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`tab-${tabId}`).classList.add('active');
    }
    
    // 日志功能
    logMessage(message, level = 'info') {
        const logOutput = document.getElementById('log-output');
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${level}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
        
        logOutput.appendChild(logEntry);
        
        // 保持日志长度
        while (logOutput.children.length > this.config.maxLogLines) {
            logOutput.removeChild(logOutput.firstChild);
        }
        
        // 滚动到底部
        logOutput.scrollTop = logOutput.scrollHeight;
    }
    
    clearLog() {
        document.getElementById('log-output').innerHTML = '';
        this.logMessage('日志已清空', 'info');
    }
    
    saveLog() {
        const logContent = document.getElementById('log-output').textContent;
        const blob = new Blob([logContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `robot_log_${new Date().toISOString().slice(0, 10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        this.logMessage('日志已保存', 'info');
    }
    
    filterLog(level) {
        const logEntries = document.querySelectorAll('.log-entry');
        
        logEntries.forEach(entry => {
            if (level === 'all' || entry.classList.contains(`log-${level}`)) {
                entry.style.display = 'block';
            } else {
                entry.style.display = 'none';
            }
        });
    }
    
    // 地图功能
    refreshMap() {
        // 这里可以实现地图刷新逻辑
        this.logMessage('地图已刷新', 'info');
    }
    
    saveMap() {
        const canvas = document.getElementById('map-canvas');
        const url = canvas.toDataURL('image/png');
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `robot_map_${new Date().toISOString().slice(0, 10)}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        this.logMessage('地图已保存', 'info');
    }
    
    updateZoom(value) {
        document.getElementById('zoom-value').textContent = `${value}%`;
        // 这里可以实现地图缩放逻辑
    }
    
    updateLaserDisplay() {
        const canvas = document.getElementById('laser-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2 - 10;
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 绘制网格
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        
        // 绘制同心圆
        for (let i = 1; i <= 4; i++) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius * i / 4, 0, Math.PI * 2);
            ctx.stroke();
        }
        
        // 绘制射线
        for (let i = 0; i < 8; i++) {
            const angle = (Math.PI * 2 * i) / 8;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(
                centerX + radius * Math.cos(angle),
                centerY + radius * Math.sin(angle)
            );
            ctx.stroke();
        }
        
        // 绘制模拟激光数据
        const numPoints = 180;
        const points = [];
        
        for (let i = 0; i < numPoints; i++) {
            const angle = (Math.PI * 2 * i) / numPoints;
            const distance = radius * (0.3 + 0.7 * Math.random());
            
            points.push({
                x: centerX + distance * Math.cos(angle),
                y: centerY + distance * Math.sin(angle)
            });
        }
        
        // 绘制点
        ctx.fillStyle = 'rgba(102, 126, 234, 0.7)';
        points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // 绘制机器人位置
        ctx.fillStyle = '#FF5252';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
        ctx.fill();
    }
    
    updateMapDisplay() {
        const canvas = document.getElementById('map-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 绘制背景
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);
        
        // 绘制网格
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        
        const gridSize = 50;
        for (let x = 0; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        for (let y = 0; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // 绘制货架位置
        const shelfPositions = {
            '1号货架': { x: width * 0.3, y: height * 0.2 },
            '2号货架': { x: width * 0.3, y: height * 0.4 },
            '3号货架': { x: width * 0.3, y: height * 0.6 },
            '4号货架': { x: width * 0.7, y: height * 0.3 },
            '5号货架': { x: width * 0.7, y: height * 0.5 },
            '充电站': { x: width * 0.1, y: height * 0.1 },
            '起点': { x: width * 0.1, y: height * 0.9 }
        };
        
        Object.entries(shelfPositions).forEach(([name, pos]) => {
            // 绘制货架
            ctx.fillStyle = name === '充电站' ? '#FF9800' : '#4CAF50';
            ctx.fillRect(pos.x - 15, pos.y - 15, 30, 30);
            
            // 绘制标签
            ctx.fillStyle = '#212529';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(name, pos.x, pos.y + 30);
        });
        
        // 绘制机器人位置
        const robotX = width * 0.5;
        const robotY = height * 0.5;
        
        // 绘制机器人
        ctx.fillStyle = '#FF5252';
        ctx.beginPath();
        ctx.arc(robotX, robotY, 10, 0, Math.PI * 2);
        ctx.fill();
        
        // 绘制方向箭头
        const heading = this.robotStatus.position.theta;
        const arrowLength = 20;
        const arrowX = robotX + arrowLength * Math.cos(heading);
        const arrowY = robotY + arrowLength * Math.sin(heading);
        
        ctx.strokeStyle = '#FF5252';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(robotX, robotY);
        ctx.lineTo(arrowX, arrowY);
        ctx.stroke();
        
        // 绘制机器人标签
        ctx.fillStyle = '#FF5252';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('机器人', robotX, robotY - 20);
        
        // 如果正在导航，绘制路径
        if (this.robotStatus.navigationTarget) {
            ctx.strokeStyle = 'rgba(102, 126, 234, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            
            ctx.beginPath();
            ctx.moveTo(robotX, robotY);
            
            // 绘制到目标位置的虚线
            const targetShelf = Object.keys(shelfPositions).find(
                name => name === this.robotStatus.navigationTarget
            );
            
            if (targetShelf) {
                const target = shelfPositions[targetShelf];
                ctx.lineTo(target.x, target.y);
                ctx.stroke();
            }
            
            ctx.setLineDash([]);
        }
    }
    
    // 设置功能
    saveSettings() {
        const settings = {
            ros_master: document.getElementById('ros-master').value,
            ros_ip: document.getElementById('ros-ip').value,
            max_linear_speed: parseFloat(document.getElementById('max-linear-speed').value),
            max_angular_speed: parseFloat(document.getElementById('max-angular-speed').value),
            safety_distance: parseFloat(document.getElementById('safety-distance').value),
            confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
            model_path: document.getElementById('model-path').value
        };
        
        // 保存到localStorage
        localStorage.setItem('robot_settings', JSON.stringify(settings));
        
        this.logMessage('设置已保存', 'info');
        this.showModal('设置保存', '设置已成功保存到本地存储');
    }
    
    loadSettings() {
        const savedSettings = localStorage.getItem('robot_settings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            document.getElementById('ros-master').value = settings.ros_master || 'http://localhost:11311';
            document.getElementById('ros-ip').value = settings.ros_ip || '127.0.0.1';
            document.getElementById('max-linear-speed').value = settings.max_linear_speed || 0.8;
            document.getElementById('max-angular-speed').value = settings.max_angular_speed || 1.5;
            document.getElementById('safety-distance').value = settings.safety_distance || 0.3;
            document.getElementById('confidence-threshold').value = settings.confidence_threshold || 0.5;
            document.getElementById('model-path').value = settings.model_path || 'models/yolov11_apple_best.pt';
            
            this.logMessage('设置已加载', 'info');
            this.showModal('设置加载', '设置已从本地存储加载');
        } else {
            this.showError('没有找到保存的设置');
        }
    }
    
    resetSettings() {
        this.showModal('重置设置', '确定要重置所有设置为默认值吗？', true, () => {
            document.getElementById('ros-master').value = 'http://localhost:11311';
            document.getElementById('ros-ip').value = '127.0.0.1';
            document.getElementById('max-linear-speed').value = 0.8;
            document.getElementById('max-angular-speed').value = 1.5;
            document.getElementById('safety-distance').value = 0.3;
            document.getElementById('confidence-threshold').value = 0.5;
            document.getElementById('model-path').value = 'models/yolov11_apple_best.pt';
            
            this.logMessage('设置已重置', 'info');
        });
    }
    
    browseModel() {
        // 这里可以实现文件浏览逻辑
        this.logMessage('模型浏览功能暂未实现', 'warning');
    }
    
    // 图表功能
    initCharts() {
        // 奖励图表
        const rewardCtx = document.getElementById('reward-chart').getContext('2d');
        this.rewardChart = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '奖励',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '奖励值'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '回合'
                        }
                    }
                }
            }
        });
        
        // 损失图表
        const lossCtx = document.getElementById('loss-chart').getContext('2d');
        this.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '损失',
                    data: [],
                    borderColor: '#FF5252',
                    backgroundColor: 'rgba(255, 82, 82, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '损失值'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '更新次数'
                        }
                    }
                }
            }
        });
    }
    
    updateTrainingCharts() {
        if (!this.rewardChart || !this.lossChart) return;
        
        // 更新奖励图表
        const rewardData = this.trainingData.rewards;
        this.rewardChart.data.labels = Array.from({length: rewardData.length}, (_, i) => i + 1);
        this.rewardChart.data.datasets[0].data = rewardData;
        this.rewardChart.update();
        
        // 更新损失图表
        const lossData = this.trainingData.losses;
        this.lossChart.data.labels = Array.from({length: lossData.length}, (_, i) => i + 1);
        this.lossChart.data.datasets[0].data = lossData;
        this.lossChart.update();
    }
    
    // 模态对话框
    showModal(title, message, showCancel = false, confirmCallback = null) {
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-message').textContent = message;
        
        const cancelBtn = document.getElementById('modal-cancel');
        cancelBtn.style.display = showCancel ? 'inline-block' : 'none';
        
        this.modalConfirmCallback = confirmCallback;
        
        document.getElementById('modal-overlay').classList.remove('hidden');
    }
    
    hideModal() {
        document.getElementById('modal-overlay').classList.add('hidden');
        this.modalConfirmCallback = null;
    }
    
    confirmModal() {
        if (this.modalConfirmCallback) {
            this.modalConfirmCallback();
        }
        this.hideModal();
    }
    
    showError(message) {
        this.showModal('错误', message);
        this.logMessage(message, 'error');
    }
    
    handleCommandResponse(response) {
        if (response.success) {
            this.logMessage(response.message || '命令执行成功', 'info');
        } else {
            this.showError(response.message || '命令执行失败');
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.robotClient = new RobotWebClient();
});
