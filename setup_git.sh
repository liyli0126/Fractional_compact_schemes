#!/bin/bash
# 设置 Git 仓库并清理不需要的文件

cd "$(dirname "$0")"

echo "=== 步骤 1: 添加 .gitignore ==="
git add .gitignore

echo ""
echo "=== 步骤 2: 从暂存区移除 .DS_Store 文件 ==="
git rm --cached .DS_Store 2>/dev/null
find . -name ".DS_Store" -exec git rm --cached {} \; 2>/dev/null

echo ""
echo "=== 步骤 3: 从暂存区移除 __pycache__ 目录 ==="
find . -name "__pycache__" -type d -exec git rm -r --cached {} \; 2>/dev/null

echo ""
echo "=== 步骤 4: 查看当前暂存状态 ==="
echo "暂存的文件数量:"
git status --short | wc -l

echo ""
echo "=== 完成！现在可以运行以下命令： ==="
echo "1. git add .                    # 添加所有文件（.gitignore 会自动排除不需要的文件）"
echo "2. git commit -m 'Initial commit'"
echo "3. git remote add origin <your-github-repo-url>"
echo "4. git push -u origin main"
