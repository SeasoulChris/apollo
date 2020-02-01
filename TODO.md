# Apollo Fuel TODO Items

对任何项目感兴趣，请直接联系Owner，然后把自己加进去。对于无主项目请直接领取并把自己列为Owner。欢迎任何新idea，请发
proposal到 usa-data@baidu.com 或直接在pysparker群里讨论。

## 通用Profiling管线
1. 需提供便捷的对各模块、各类指标的统计信息，包括平均值、最大值、最小值、std_dev， 10分位、90分位， 99分位，对异常信
   息进行邮件报警，在Dashboard上作图展示。
1. 可参考pandas的 DataFrame.describe()

## 数据流水线性能的量化度量
1. 可自研白盒度量标准和工具
1. 可调研Kubernetes、Spark其他的黑盒度量工具

# 持续调研和学习
1. Spark和K8S都在持续迭代，即时关注新特性有助于我们更好地优化框架
1. PyTorch 1.4 支持分布式训练
1. Kubeflow 原生支持Tensorflow分布式训练
1. 列式数据库仍有可能是我们存储数据的终极形态，国内也有一些尝试。我们可以拿small-records来做实验。但资源消耗仍然巨大，
   收益不明显，且Cyber、ROS的工具也都是针对原始文件的。数据库可以选择Cassandra，云端可以选择Baidu TableStorage：
   https://cloud.baidu.com/product/bts.html
1. 时序数据库也是新兴的物联网解决方案，可能适合无人车场景，但对于海量数据性能未知。开源产品可选择InfluxDB，云端可以选
   择Baidu TSDB：https://cloud.baidu.com/product/tsdb.html 书籍已有《OpenTSDB技术内幕》
