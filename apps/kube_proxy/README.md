# kubernetes service proxy  
## Build and Run
Build and run an docker image running kubernetes proxy:  

    bash build_on_host.sh  

## Access your service
Format and visit the url below:

    http://{host-ip}:{host-port}/api/v1/namespaces/default/services/http:{service-name}:{service-port}/proxy/
