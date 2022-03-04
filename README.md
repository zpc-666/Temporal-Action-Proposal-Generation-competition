# 基于飞桨实现乒乓球时序动作定位大赛 ：B榜第2名方案

## 项目描述
基于飞桨实现乒乓球时序动作定位大赛 ：B榜第2名方案相关的全部代码及方案解说。

## 项目结构
```
-|PaddleVideo # 我们修改之后的训练验证预测套件
-|generate_data_for_training.py # 生成训练验证视频切片及信息json文件
-|generate_data_for_testing.py # 生成测试视频切片及信息json文件
-|utils.py # 数据集预处理及提案后处理工具箱
-README.MD
-main.ipynb # 方案介绍以及展现训练验证测试流程
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/3548768?contributionType=1)  
B：下载下来，按照main.ipynb一步步运行即可，当然数据集要在AiStudio上下载，本地安装好Paddle，GPU至少16GB显存，不过你可以调整batch size来适应
