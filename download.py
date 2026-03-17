import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
MS_REPO_ID = "chuanjun/Seismic-AI-Data"
TARGET_FOLDER = "AQ2009GM"

# 本地保存根目录 (确保指向你的 F 盘)
LOCAL_SAVE_ROOT = "/mnt/f/AI_Seismic_Data"

# 并发下载数
MAX_WORKERS = 2
# ===========================================

# 核心：导入你指定的 ModelScope 数据集单文件下载 API
from modelscope.hub.file_download import dataset_file_download

def get_remote_file_list(repo_id, folder_path):
    """
    通过底层网络请求获取 ModelScope 目录下的文件名单。
    这样做 100% 规避了 SDK 版本不同导致的 list_repo_tree 参数报错问题。
    """
    url = f"https://modelscope.cn/api/v1/datasets/{repo_id}/repo/tree"
    params = {"Revision": "master", "Root": folder_path}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        file_paths = []
        if "Data" in data and "Files" in data["Data"]:
            for item in data["Data"]["Files"]:
                if item["Type"] == "tree":
                    # 如果有子文件夹，递归获取
                    file_paths.extend(get_remote_file_list(repo_id, item["Path"]))
                elif item["Type"] == "blob":
                    # 拿到具体的文件路径
                    file_paths.append(item["Path"])
        return file_paths
    except Exception as e:
        print(f"⚠️ 获取远端目录清单失败: {e}")
        return []

def process_download_task(file_path):
    """
    单个文件的检查与下载逻辑
    """
    # 预判目标文件在本地的绝对路径
    target_local_path = os.path.join(LOCAL_SAVE_ROOT, file_path)
    
    # --- [增量检查] ---
    # 如果本地已经有这个文件了，直接跳过，不调用 API，节省时间和流量
    if os.path.exists(target_local_path):
        return f"⏭️ [已存在-跳过] {file_path}"
        
    try:
        # --- [官方接口下载] ---
        # 调用你指定的 dataset_file_download
        # local_dir 参数会自动帮我们在本地创建对应的 AQ2009GM/... 目录结构
        dataset_file_download(
            dataset_id=MS_REPO_ID,
            file_path=file_path,
            local_dir=LOCAL_SAVE_ROOT
        )
        return f"✅ [下载成功] {file_path}"
    except Exception as e:
        return f"❌ [下载失败] {file_path}: {e}"

def incremental_sync():
    print(f"🚀 启动增量下载: ModelScope({MS_REPO_ID}) -> {LOCAL_SAVE_ROOT}")
    print(f"📂 目标文件夹: {TARGET_FOLDER}")
    
    # 1. 获取名单
    print("\n🔍 正在获取远端文件清单...")
    remote_files = get_remote_file_list(MS_REPO_ID, TARGET_FOLDER)
    
    if not remote_files:
        print(f"⚠️ 未找到 {TARGET_FOLDER} 目录下的文件。")
        return
        
    print(f"📦 远端共发现 {len(remote_files)} 个文件，开始比对与下载...")
    
    success = 0
    skipped = 0
    failed = 0
    
    # 2. 多线程比对并下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_download_task, f): f for f in remote_files}
        
        # 进度条
        with tqdm(total=len(remote_files), unit='file', desc="同步进度") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                pbar.update(1)
                
                if "⏭️" in result:
                    skipped += 1
                elif "✅" in result:
                    success += 1
                else:
                    print(result) # 打印失败原因
                    failed += 1
                    
    print(f"\n🏁 任务结束 | 新增下载: {success} | 已跳过(本地存在): {skipped} | 失败: {failed}")
    if failed > 0:
        print("💡 提示: 有少量失败文件，重新运行脚本会自动跳过已完成的文件并重试失败项。")

if __name__ == "__main__":
    incremental_sync()