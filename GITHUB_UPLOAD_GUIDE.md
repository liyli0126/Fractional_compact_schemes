# GitHub 上传指南

本指南将帮助您将代码上传到 GitHub，同时排除 Mac 系统文件（如 `.DS_Store`）和 Python 缓存文件（如 `__pycache__`）。

## 快速开始

### 方法一：使用自动化脚本（推荐）

```bash
cd "/Users/yingli/Desktop/Fractional_compact"
./setup_git.sh
git add .
git commit -m "Initial commit"
git remote add origin <你的GitHub仓库URL>
git push -u origin main
```

### 方法二：手动操作

#### 步骤 1: 清理暂存区中的不需要文件

```bash
cd "/Users/yingli/Desktop/Fractional_compact"

# 从暂存区移除所有 .DS_Store 文件
find . -name ".DS_Store" -exec git rm --cached {} \;

# 从暂存区移除所有 __pycache__ 目录
find . -name "__pycache__" -type d -exec git rm -r --cached {} \;
```

#### 步骤 2: 添加 .gitignore 文件

`.gitignore` 文件已经创建好了，它会自动排除：
- `.DS_Store` 和其他 Mac 系统文件
- `__pycache__/` 和 Python 缓存文件
- 虚拟环境目录
- IDE 配置文件

```bash
git add .gitignore
```

#### 步骤 3: 添加所有代码文件

```bash
git add .
```

现在 Git 会自动忽略 `.gitignore` 中列出的文件。

#### 步骤 4: 提交代码

```bash
git commit -m "Initial commit: Fractional compact schemes (MATLAB and Python)"
```

#### 步骤 5: 在 GitHub 上创建仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" → "New repository"
3. 输入仓库名称（例如：`Fractional_compact`）
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

#### 步骤 6: 连接本地仓库到 GitHub

```bash
# 添加远程仓库（替换为你的实际URL）
git remote add origin https://github.com/你的用户名/你的仓库名.git

# 或者使用 SSH（如果已配置）
git remote add origin git@github.com:你的用户名/你的仓库名.git
```

#### 步骤 7: 推送代码到 GitHub

```bash
git branch -M main
git push -u origin main
```

## 验证

上传后，在 GitHub 网页上检查：
- ✅ 不应该看到任何 `.DS_Store` 文件
- ✅ 不应该看到任何 `__pycache__` 目录
- ✅ 只应该看到 `.m` 和 `.py` 源代码文件

## 如果已经提交了不需要的文件

如果之前已经提交了 `.DS_Store` 或 `__pycache__`，需要从 Git 历史中完全移除：

```bash
# 从所有提交中移除 .DS_Store
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch **/.DS_Store" \
  --prune-empty --tag-name-filter cat -- --all

# 从所有提交中移除 __pycache__
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch **/__pycache__" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送（警告：这会重写历史）
git push origin --force --all
```

## .gitignore 说明

`.gitignore` 文件已配置为排除：

1. **Mac 系统文件**
   - `.DS_Store` - Finder 元数据
   - `.AppleDouble`, `.LSOverride` - 其他 Mac 系统文件

2. **Python 文件**
   - `__pycache__/` - Python 字节码缓存
   - `*.pyc`, `*.pyo` - 编译后的 Python 文件
   - `*.egg-info/` - 包信息

3. **虚拟环境**
   - `venv/`, `env/`, `.venv/` - Python 虚拟环境

4. **IDE 配置**
   - `.vscode/`, `.idea/` - 编辑器配置

## 常见问题

**Q: 为什么我的 `.DS_Store` 文件还在暂存区？**
A: 运行 `git rm --cached` 命令后，文件会从暂存区移除，但本地文件仍然存在（这是正常的）。

**Q: 如何防止以后再次添加这些文件？**
A: `.gitignore` 文件已经配置好了，Git 会自动忽略这些文件。你也可以全局配置 Git：

```bash
# 全局忽略 .DS_Store
git config --global core.excludesfile ~/.gitignore_global
echo ".DS_Store" >> ~/.gitignore_global
```

**Q: 如何检查哪些文件会被上传？**
A: 运行 `git status` 查看暂存的文件，运行 `git ls-files` 查看所有被追踪的文件。
