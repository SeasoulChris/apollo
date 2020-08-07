# replace the spark_submitter config to dreamland environment

# replace the kubectl proxy host
sed -i 's/10.99.197.153/10.90.254.232/g' ../client.py
# replace the k8s host, which can be found in ~/.kube/config
sed -i 's/180.76.60.99/180.76.244.207/g' ../utils.py
# replace the mongodb host
sed -i 's/auPpfadGy.mongodb.bj.baidubce.com/Rknf3Gadg.mongodb.bj.baidubce.com/g' ../../../../fueling/common/mongo_utils.py
sed -i 's/auPpfaZwf.mongodb.bj.baidubce.com/Rknf3Gyce.mongodb.bj.baidubce.com/g' ../../../../fueling/common/mongo_utils.py
# replace the redis host
sed -i 's/192.168.32.22/redis.fyqpoktasipw.scs.bj.baidubce.com/g' ../../../../fueling/common/redis_utils.py

# change the ~/.kube/config file to dreamland version manually
