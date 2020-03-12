### 目的

本文档定义模型数据内容及格式

- 保证Plannning learning data 的pb binary 包含所有除labeling外的数据内容
- 保证Planning labeling 的内容和格式正确
- 规范语义地图的数据内容及格式
- 规范模型所需数据内容及格式



### 数据内容及格式



#### 主车当前信息

主车当前信息来源于localination和chassis，应对齐其感知的某一帧数据时间点。 localization 和chassis 相较于感知的msg的前后时间差不应超过【30ms】。

##### 包含信息

- 位置
- Heading
- 速度
- 加速度
- 加加速度



#### 主车历史信息

主车过去【1秒】内的历史信息，可以为空。

##### 包含信息

- 位置
- Heading
- 速度
- 加速度
- 加加速度



#### 障碍物信息

障碍物当前信息，来源于perception

##### 包含信息

- 位置
- 形状大小
- heading
- 速度



#### 障碍物预测信息

障碍物预测信息，未来【8秒】，来源于prediction

##### 包含信息

- 位置
- heading
- 速度



#### Routing信息

routing 信息来源于routing_reponse 或者 routing_response_history

##### 包含信息

- lane id
- lane的中心线轨迹点



#### 红绿灯信息

红绿灯的位置及状态，信息来源于 traffic light status 和地图

##### 包含信息

- 红绿灯状态
- 红绿灯位置
- 红绿灯停止线位置



#### 地图信息

来源于地图或者训练数据

##### 包含信息

- 地图名称
  - 比如sunnyvale， sunnyvale_big_loop
- 地图版本
  - 比如1.0



#### 主车未来真实轨迹点

来源于localization 和chassis

##### 包含信息

- 位置
- Heading
- 速度
- 加速度
- 加加速度