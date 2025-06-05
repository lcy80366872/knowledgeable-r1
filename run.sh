



# 初始化全局变量
ERROR_FLAG=0
CURRENT_SESSION_ID=""
export NCCL_P2P_LEVEL=LOC
#export DS_SKIP_CUDA_CHECK=1
source activate vllm-cuda118
# 清理函数
cleanup() {
    echo "[清理] 终止所有相关进程..."
    
    # 终止API服务
    if kill -0 $API_PID 2>/dev/null; then
        kill -9 $API_PID && echo "已终止API进程(PID: $API_PID)"
    fi
    
    # 终止客户端进程树
    if [ -n "$CURRENT_SESSION_ID" ]; then
        echo "正在终止会话 $CURRENT_SESSION_ID 的所有进程..."
        pkill -s $CURRENT_SESSION_ID && echo "已终止会话进程"
        # 双重确认清理
        pgrep -s $CURRENT_SESSION_ID | xargs -r kill -9
    fi
}

# 注册信号捕获
trap 'cleanup; exit' INT TERM EXIT

# 启动API服务（指定GPU7）
CUDA_VISIBLE_DEVICES=7 python ref_server.py &
API_PID=$!
echo "[主进程] API服务已启动(PID: $API_PID)，分配GPU0"

# 延时启动客户端函数
delayed_client_start() {
    echo "[主进程] 等待20秒后启动客户端..."
    for i in {1..20}; do
        printf "\r倒计时: %2d/20秒" $i
        sleep 1
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "\n[错误] API进程异常退出!"
            return 1
        fi
    done
    
    echo -e "\n[主进程] 启动客户端程序..."
    
    # 使用独立会话启动客户端，记录会话ID
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 setsid  deepspeed --include localhost:1,2,3,4,5,6 grpo_program.py &  # --include localhost:1,2,3,4,5,6
    CLIENT_PID=$!
    CURRENT_SESSION_ID=$(ps -o sid= -p $CLIENT_PID | tr -d ' ')
    echo "[主进程] 客户端已启动(PID: $CLIENT_PID, SID: $CURRENT_SESSION_ID)，分配GPU1,2,3,4,5,6"

    # 监控循环
    while true; do
        sleep 5
        # 检测step_20文件夹
        if [ -d "save_model/final_model" ]; then
            echo "[主进程] 检测到final_mode文件夹，等待客户端进程结束..."
            ERROR_FLAG=1

            # 等待客户端退出（最多2分钟）
            TIMEOUT=120
            START_TIME=$(date +%s)
            while ps -s $CURRENT_SESSION_ID >/dev/null; do
                CURRENT_TIME=$(date +%s)
                ELAPSED=$((CURRENT_TIME - START_TIME))
                if [ $ELAPSED -ge $TIMEOUT ]; then
                    echo "[警告] 等待客户端超时，强制终止..."
                    break
                fi
                printf "\r已等待 %3d 秒（剩余 %3d 秒）" $ELAPSED $((TIMEOUT - ELAPSED))
                sleep 10
            done

            # 同步Wandb数据
            # echo -e "\n[主进程] 启动Wandb数据同步..."
            # wandb sync ./step_1000 --sync-all
            # echo "[主进程] Wandb数据同步完成"

            cleanup
            return 1
        fi
        
        # 检查客户端会话是否存在活跃进程
        if ! ps -s $CURRENT_SESSION_ID >/dev/null; then
            wait $CLIENT_PID
            CLIENT_EXIT_CODE=$?
            [ $CLIENT_EXIT_CODE -ne 0 ] && ERROR_FLAG=1
            break
        fi
    done
}


# 最终状态检查
if [ $ERROR_FLAG -eq 0 ]; then
    echo "[完成] 所有进程正常退出"
    exit 0
else
    echo "[完成] 存在异常退出进程"
    exit 1
fi
