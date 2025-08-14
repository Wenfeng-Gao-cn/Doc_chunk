#!/bin/bash

# 服务管理脚本
SERVICE_NAME="app_gen_chunks.py"
PYTHON_SCRIPT="app_gen_chunks.py"
LOG_DIR="logs"
PID_FILE="${SERVICE_NAME}.pid"
LOG_FILE="${LOG_DIR}/${SERVICE_NAME}_$(date +%Y%m%d).log"

# 确保在脚本所在目录执行
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

# 创建日志目录
mkdir -p ${LOG_DIR}

# 启动服务
start() {
    if [ -f "${PID_FILE}" ]; then
        echo "服务已在运行中 (PID: $(cat ${PID_FILE}))"
        return 1
    fi

    # 获取知识库目录
    read -p "请输入需要导入知识库的目录(默认: sample_doc): " DOC_DIR
    DOC_DIR=${DOC_DIR:-sample_doc}

    # 检查Python解释器
    PYTHON_PATH=$(which python3)
    if [ -z "$PYTHON_PATH" ]; then
        echo "错误: 未找到python3解释器"
        return 1
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') 启动 ${SERVICE_NAME}..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') 使用Python路径: $PYTHON_PATH"
    nohup "$PYTHON_PATH" "${PYTHON_SCRIPT}" --doc_dir "${DOC_DIR}" >> "${LOG_FILE}" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S') 服务进程已启动 (PID: $!)"
    echo $! > ${PID_FILE}
    echo "服务已启动 (PID: $(cat ${PID_FILE}))"
    echo "知识库目录: ${DOC_DIR}"
    echo "日志文件: ${LOG_FILE}"
}

# 停止服务
stop() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "服务未运行"
        return 1
    fi

    PID=$(cat ${PID_FILE})
    echo "$(date '+%Y-%m-%d %H:%M:%S') 停止 ${SERVICE_NAME} (PID: ${PID})..."
    kill ${PID}
    rm -f ${PID_FILE}
    echo "$(date '+%Y-%m-%d %H:%M:%S') 服务已停止"
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
            echo "$(date '+%Y-%m-%d %H:%M:%S') ${SERVICE_NAME} 正在运行 (PID: ${PID})"
            echo "$(date '+%Y-%m-%d %H:%M:%S') 日志文件: ${LOG_FILE}"
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
