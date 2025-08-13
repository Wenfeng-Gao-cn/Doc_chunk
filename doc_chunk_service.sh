#!/bin/bash

# 服务管理脚本
SERVICE_NAME="doc_chunk_service"
PYTHON_SCRIPT="Write_k_b_from_folder.py"
LOG_DIR="logs"
PID_FILE="${SERVICE_NAME}.pid"
LOG_FILE="${LOG_DIR}/${SERVICE_NAME}_$(date +%Y%m%d).log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 启动服务
start() {
    if [ -f "${PID_FILE}" ]; then
        echo "服务已在运行中 (PID: $(cat ${PID_FILE}))"
        return 1
    fi

    echo "启动 ${SERVICE_NAME}..."
    nohup python3 ${PYTHON_SCRIPT} >> ${LOG_FILE} 2>&1 &
    echo $! > ${PID_FILE}
    echo "服务已启动 (PID: $(cat ${PID_FILE}))"
    echo "日志文件: ${LOG_FILE}"
}

# 停止服务
stop() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "服务未运行"
        return 1
    fi

    PID=$(cat ${PID_FILE})
    echo "停止 ${SERVICE_NAME} (PID: ${PID})..."
    kill ${PID}
    rm -f ${PID_FILE}
    echo "服务已停止"
}

# 重启服务
restart() {
    stop
    sleep 2
    start
}

# 查看服务状态
status() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null; then
            echo "${SERVICE_NAME} 正在运行 (PID: ${PID})"
            echo "日志文件: ${LOG_FILE}"
            tail -n 10 ${LOG_FILE}
            return 0
        else
            echo "${SERVICE_NAME} PID文件存在但进程不存在"
            return 2
        fi
    else
        echo "${SERVICE_NAME} 未运行"
        return 3
    fi
}

# 查看实时日志
logs() {
    if [ ! -f "${LOG_FILE}" ]; then
        echo "日志文件不存在: ${LOG_FILE}"
        return 1
    fi
    tail -f ${LOG_FILE}
}

# 主逻辑
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|status|logs}"
        exit 1
esac

exit 0
