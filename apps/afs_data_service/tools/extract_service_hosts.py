"""extract adb service name to ip"""
import json
import socket
import urllib3


def extract_service(serviceName):
    """extract internal hostname & ip from service name
    return: [(host1, ip1, port1), (host2, ip2, port2,), ...]
    """
    hostinfos = []
    http = urllib3.PoolManager()
    r = http.request('GET',
                     F'http://bns.noah.baidu.com/webfoot/index.php?'
                     F'r=webfoot/GetInstanceInfo&serviceName={serviceName}')
    if r.status == 200:
        data = json.loads(r.data)
        for item in data['instanceInfo']:
            hostName = item['hostName']
            status = item['status']
            port = json.loads(item['port'])['main']
            if status == '0':
                ip = socket.gethostbyname(hostName)
                hostinfos.append((hostName, ip, port))
    return hostinfos


def get_ip_list(serviceName):
    """extract internal ip list from service name"""
    # [(host1, ip1, port1), (host2, ip2, port2,), ...]
    hostinfos = extract_service(serviceName)
    # ip1:port1,ip2:port2,...
    iplist = ','.join(map(lambda h: F'{h[1]}:{h[2]}', hostinfos))
    return iplist


if __name__ == '__main__':
    adb_server_hosts = 'adb-server-online.IDG.yq'
    adb_export_server_hosts = 'export-server.AD-EAP.all'

    adb_server_hosts_ips = get_ip_list(adb_server_hosts)
    print(F'adb.server.hosts: {adb_server_hosts_ips}')

    adb_export_server_hosts_ips = get_ip_list(adb_export_server_hosts)
    print(F'adb.export_server.hosts: {adb_export_server_hosts_ips}')

