set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

source /usr/local/miniconda/bin/activate py27_adbsdk

# wiki: http://wiki.baidu.com/pages/viewpage.action?pageId=1040599879
# VISIT http://agile.baidu.com/#/release/baidu/adu/adbsdk
# COPY and REPLACE the TOKEN/ADBSDK_URL
TOKEN=""
ADBSDK_URL=""
# install adbsdk
wget -O adbsdk.tar.gz --no-check-certificate --header "IREPO-TOKEN:${TOKEN}" "${ADBSDK_URL}"

mkdir adbsdk_tmp
tar zxf adbsdk.tar.gz -C adbsdk_tmp
tar zxf adbsdk_tmp/output/adbsdk*.tar.gz
rm -r adbsdk_tmp
rm adbsdk.tar.gz
mv adbsdk /adbsdk
sudo chmod 777 /adbsdk
cd /adbsdk
python setup.py install
