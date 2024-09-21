---
title: Multi-threaded Brotli compression tool
date: 2024-07-13 10:52:52
tags: [devtools, algorithms]
---

## Brotli compression is slow
[Brotli](https://en.wikipedia.org/wiki/Brotli) is an extremely efficient compression format (alternative to _gzip_, _zstd_, etc.), with [google/brotli](https://github.com/google/brotli) being sort of its reference implementation. What always bugged me with Brotli is computation time, though. As a consequence of yielding such good compression results, Brotli-compressing a file is also an extremely slow process (orders of magnitude slower compared to _gzip_). That fact that the "official" `brotli` command-line tool (written in C) is single-threaded doesn't particularly help with this either.

## Multi-threaded implementation
Recently came across [this article](https://dropbox.tech/infrastructure/-broccoli--syncing-faster-by-syncing-less) on Dropbox's tech blog (super interesting, btw.!), though, where they describe how they switched to Brotli for file downloads and uploads. In that context, they came up with their own Rust impementation of Brotli. What's especially interesting about this is the fact that their library support multi-threaded compression. In my understanding, this was enabled by "simply" compressing individual chunks in parallel and then concatenating the binary output.

Their tool is mainly a Rust library and comes with C bindings in addition. Enabled through those, they also provide Python- and Go bindings. What's more, they [release](https://github.com/dropbox/rust-brotli/releases) CLI executables in addition, however, apparently only for Windows (?). Perhaps it's a drop-in replacement for Google's `brotli` command, but I didn't get to try it, as I didn't have easy access to a Windows machine at the time. Building the Rust project produces an executable, but (without having read through the code) it seems to be single-threaded, only.

## Getting the Go version running

Instead, I dove a bit deeper into the code and realized their [Go example](https://github.com/dropbox/rust-brotli/blob/master/c/go/main.go) is also a more or less ready-to-use, standalone program to perform multi-threaded (de-) compression (without a proper command-line interface, though). I had to apply a couple of changes to get it working. Specifically, I did:

- Fix the library dependency version
- Update the hard-coded, relative dependency path to my local home dir
- Make it use as many threads as available CPU cores by default
- Link the `brotli_ffi` library statically into the Go executable

You can find patch files for my changes below.

<details>
<summary><code>my-patch.go</code></summary>

```
diff --git a/c/go/brotli/brotli.go b/c/go/brotli/brotli.go
index ce24370..3294664 100644
--- a/c/go/brotli/brotli.go
+++ b/c/go/brotli/brotli.go
@@ -2,7 +2,7 @@ package brotli
 
 /*
 #cgo CFLAGS: -I. -I../../..
-#cgo LDFLAGS: -L../../../target/release -L../target/release -L../../target/release -lbrotli_ffi -lm -ldl
+#cgo LDFLAGS: -L../../../target/release -L../target/release -L../../target/release -Wl,-Bstatic -lbrotli_ffi -Wl,-Bdynamic -lm -ldl -lc -lmvec
 #include "brotli/encode.h"
 #include "brotli/decode.h"
 #include "brotli/broccoli.h"
diff --git a/c/go/go.mod b/c/go/go.mod
index d159bcd..683a633 100644
--- a/c/go/go.mod
+++ b/c/go/go.mod
@@ -2,6 +2,6 @@ go 1.18
 
 module main
 
-require github.com/dropbox/rust-brotli/c/go/brotli v0.0.0-20220217093550-f3a32293f213
+require github.com/dropbox/rust-brotli/c/go/brotli v0.0.0-20240527152928-37d403b437c3
 
-replace github.com/dropbox/rust-brotli/c/go/brotli => /home/danielrh/dev/rust-brotli/c/go/brotli
+replace github.com/dropbox/rust-brotli/c/go/brotli => /home/ferdinand/dev/rust-brotli/c/go/brotli
diff --git a/c/go/main.go b/c/go/main.go
index b0955c5..9b87e92 100644
--- a/c/go/main.go
+++ b/c/go/main.go
@@ -5,12 +5,13 @@ import (
 	"io"
 	"io/ioutil"
 	"os"
+	"runtime"
 )
 
 func main() {
 	decompress := false
 	options := brotli.CompressionOptions{
-		NumThreads: 1,
+		NumThreads: runtime.NumCPU(),
 		Quality:    9.5,
 		Catable:    true,
 		Appendable: true,

```
</details>

After applying these changes, building the executable was straightforward:

```bash
make
cd c/go
go build -o brotli-rust
```

Eventually, I was able to Brotli-compress a file while utilizing 100 % of my CPU 🙌:

```bash
cat random.txt | ./brotli-rust -w > random.txt.br
```

## Benchmark

I ran a few quick test to compare performance on a 128 MB high-entropy text file and these are the results (on a 12-core CPU):

```bash
time brotli random.txt                                # 197.83 s
time cat random.txt | brotli-rust -w > random.txt.br  # 7.62 s
time gzip random.txt                                  # 4.91 s
time pigz random.txt                                  # 0.35 s
```

In this case, the multi-threaded version is 26x faster than the original one. For comparison, I ran _gzip_ as well (single- and multi-threaded, the latter using `pigz`). However, the comparison is unfair, of course, as gzip compression rate is much worse on average.