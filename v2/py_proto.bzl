def py_proto(name, src):
    src_path = "$(location {})".format(src)
    out_file = src.replace(".proto", "_pb2.py")
    out_path = "$$(dirname {})/{}".format(src_path, out_file)
    native.genrule(
        name = name + "_rule",
        srcs = [src],
        outs = [out_file],
        cmd = "protoc $< -I/fuel -I/apollo --python_out=. && mv {} $@".format(out_path),
    )
    native.py_library(
        name = name,
        srcs = [out_file]
    )
