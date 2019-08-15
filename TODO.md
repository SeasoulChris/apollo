# Apollo Fuel TODO Items

对任何项目感兴趣，请直接联系Owner，然后把自己加进去。对于无主项目请直接领取并把自己列为Owner。欢迎任何新idea，请发proposal到 usa-data@baidu.com 或直接在pysparker群里讨论。

## 2019 Q3

### 感知“数据标准”制定，收集数据X万帧
1. 涉及到Lidar、Camera、Lane Detection等，需整合数据需求；但如果因此产生极大delay，就一项一项做。
1. PM：Natasha
1. 感知：Kawai（Camera），Guang（Lidar），Yuliang（Lane Detection）
1. 数据：Longtao，Xiangquan

### 传感器标定
1. PM：Natasha
1. 感知：Guang
1. 数据：Longtao，Xiangquan

### 支持Azure Blob存储
1. PM：Oliver
1. 数据：Longtao，Xiangquan

### 通用Profiling管线
1. 需提供便捷的对各模块、各类指标的统计信息，包括平均值、最大值、最小值、std_dev， 10分位、90分位， 99分位，对异常信息进行邮件报警，在Dashboard上作图展示。
1. 可参考pandas的 DataFrame.describe()

## 候选项目

### 开放服务的Web前端
1. 现有的命令行工具的等价替代：https://github.com/ApolloAuto/apollo/blob/master/modules/tools/fuel_proxy/submit_job.py
1. 需要至少一个最简化的合作伙伴管理系统，可以登录和提交任务
1. 需部署于Baidu BAE Pro应用引擎

### 数据流水线性能的量化度量
1. 可自研白盒度量标准和工具
1. 可调研Kubernetes、Spark其他的黑盒度量工具

### 核心环境更新升级
1. Ubuntu、Conda、Python 2.7等依赖于上游Apollo环境的升级，持续跟踪
1. Kubernetes集群依赖于百度云的升级，持续跟踪
1. PySpark、Python3环境可自由升级，但依赖于新版本对上述环境的依赖和冲突，已知问题：
1. PySpark 2.4.3与Cyber的Numpy 1.8.2冲突，需维持2.4.0并等待上游升级
1. 模型训练所需的torchvision库暂不支持Python 3.7，需维持3.6并等待上游升级

### Driving Path噪音消除
1. 我们会在ETL流水线里对车辆定位信息进行采样：https://github.com/ApolloAuto/apollo-fuel/blob/master/fueling/data/record_parser.py#L160
1. 然后在Dashboard上绘制，例如：http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:warehouse-service:8000/proxy/task/mnt/bos/small-records/2019/2019-08-06/2019-08-06-20-04-00
1. 但是因为定位的抖动，尤其是冷启动阶段，会导致我们目前naive的采样逻辑得到了少量噪点，在Dashboard上产生错误：http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:warehouse-service:8000/proxy/task/mnt/bos/small-records/2019/2019-08-06/2019-08-06-12-24-46 同时我们对里程的统计也是错误的，需要修正。
1. 因此我们需要改进采样方法，丢弃明显错误的定位点

## 持续调研和学习

1. Spark和K8S都在持续迭代，即时关注新特性有助于我们更好地优化框架。
1. AI计算需求持续增加，业余持续学习PyTorch、TensorFlow和PaddlePaddle等框架，有助于提出更好的支持方案。
1. 列式数据库仍有可能是我们存储数据的终极形态，国内也有一些尝试。我们可以拿small-records来做实验。但资源消耗仍然巨大，收益不明显，且Cyber、ROS的工具也都是针对原始文件的。数据库可以选择Cassandra，云端可以选择Baidu TableStorage：https://cloud.baidu.com/product/bts.html
1. 时序数据库也是新兴的物联网解决方案，可能适合无人车场景，但对于海量数据性能未知。开源产品可选择InfluxDB，云端可以选择Baidu TSDB：https://cloud.baidu.com/product/tsdb.html 书籍已有《OpenTSDB技术内幕》
1. 云端多机GPU训练短期内是一个OverKill，但持续关注业界。除百度云的Infinite进展以外，可关注华为Volcano（支持Spark、PyTorch、TF），字节跳动BytePS（支持TF、Keras、PyTorch）。