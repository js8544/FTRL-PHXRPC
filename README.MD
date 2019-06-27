# 基于PHXRpc的FTRL算法工程实现

## 文件

该项目有三个可执行文件，分别为

1. ftrl：FTRL模型，业务逻辑
2. ftrl_main：服务器接口
3. ftrl_tool_main：客户端接口

## 用法

首先运行业务逻辑进程，用来维护FTRL模型
`./ftrl <data> <model_location>`
然后运行服务器接口
`./ftrl_main -c ftrl_server.conf`

环境搭建完毕后使用客户端接口调用FTRL模型请求广告
`./ftrl_tool_main -c ftrl_client.conf -f FTRL -u <user_id>`

得到返回的广告id后，需要反馈结果（是否点击该广告）
`./ftrl_tool_main -c ftrl_client.conf -f Feedback -u <user_id> -a <ad_id> -r <result>`
