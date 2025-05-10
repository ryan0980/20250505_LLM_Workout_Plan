from huggingface_hub import HfApi, Repository
import os

# 1. 配置
HF_TOKEN = os.getenv("HF_TOKEN", "")  # 建议通过环境变量 HF_TOKEN 注入
REPO_ID = "tusrau/Lawbench_split"                      # 你的用户名/ 数据集名
LOCAL_DIR = "lawbench3"                                # 本地目录

# 2. 初始化 API 并创建（如果已存在可跳过）数据集仓库
api = HfApi()
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True,       # 如果仓库已存在则不会报错
    token=HF_TOKEN
)

# 3. 克隆到本地（如果你之前没有 git clone）
repo = Repository(
    local_dir=LOCAL_DIR,
    clone_from=REPO_ID,
    repo_type="dataset",
    use_auth_token=HF_TOKEN
)

# 4. 把所有文件添加到 git
repo.git_add(pattern="*")

# 5. 提交并推送到 Hugging Face
repo.git_commit("Initial upload of LawBench splits")
repo.git_push()

print("上传完成 ✅")
