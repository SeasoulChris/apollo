# Web Portal

## Debug

```bash
conda install flask gunicorn
bazel run //apps/web_portal:index -- --debug
```

## Deploy

```bash
deploy/0_...sh
deploy/1_...sh
```
