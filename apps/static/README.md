# Pre compiled tools

## Upload a single file to BOS

```bash
bos_fstool --src=path/to/local/file.txt --dst=path/to/bos/file.txt
```

When finished, you'll see the file at /mnt/bos/path/to/bos/file.txt, if your BOS
is mounted at /mnt/bos.
