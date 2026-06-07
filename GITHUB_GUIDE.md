# GitHub 上传指南

## 第1步：在GitHub创建仓库

1. 打开 https://github.com/new
2. 仓库名填写：`random-number-test`
3. 设为 Public（公开）或 Private（私有）都可以
4. **不要勾选** "Add a README file"、"Add .gitignore" 等选项（因为本地已有）
5. 点击 "Create repository"

## 第2步：推送本地代码

创建好仓库后，在终端执行：

```bash
cd F:\random-number-test
git push -u origin master
```

如果遇到 SSL 错误，先执行：
```bash
git config http.schannelCheckRevoke false
git push -u origin master
```

如果提示输入用户名密码，使用 GitHub 的 Personal Access Token（个人访问令牌）替代密码。
