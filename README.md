
## 开始使用

### 环境准备
在开始之前，您需要确保已安装所有依赖项，以便正确运行MDMIN。使用以下命令安装依赖项：

```bash
pip install -r requirements.txt
```

### 数据下载
为了训练模型，您需要下载相应的数据集。我们推荐使用从[Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)提供的数据集。请按以下步骤操作：

1. 从上述链接下载所有必需的CSV数据文件。
2. 在项目根目录下创建一个名为`dataset`的文件夹，并将下载的CSV文件放在这个目录中。

### 模型训练
所有训练脚本均存放在`scripts`目录下。您可以运行特定的脚本来训练模型或获取预测结果。例如，要获取天气数据集的多变量预测结果，请运行：

```bash
sh ./scripts/MDMIN/weather.sh
```

训练完成后，可以在`result.txt`文件中查看输出结果。

