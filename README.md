# 专业课程资料辅助学习系统

本版本为课题三多页面版，已经去掉“报告草稿生成”和“答辩话术”等与系统功能无关的内容。

## 运行方式

双击根目录：

```bat
run-local.bat
```

浏览器打开：

```text
http://127.0.0.1:8010
```

本版本默认使用 8010 端口，避免你电脑上旧版 8000 进程没有关闭时仍然打开旧页面。

## 页面

- 总览
- 资料库
- 智能问答
- 摘要笔记
- 自测训练
- 检索溯源
- 复习计划
- 学习历史
- 系统设置

## API Key

复制 `backend/.env.example` 为 `backend/.env`，填写：

```env
LLM_API_KEY=你的 DeepSeek Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

保存后重启服务，页面会显示“API 已连接：真实大模型”。
