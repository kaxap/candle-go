An example on using [huggingface/candle](https://github.com/huggingface/candle) with Golang. For educational purposes only.

## Usage
1. Compile this project with Rust:
```bash
cargo build --release
```
2. Navigate to `go/` directory and build the Go binary there:
```bash
go build
```
The cgo code references Rust library in a relative path:
```
#cgo LDFLAGS: -L../target/release -ltxtvec
```
So the relative path should be the same for `go build` to work.
3. Now you have a single fat binary that can download and cache [e5-large-multilingual](https://huggingface.co/intfloat/multilingual-e5-large) model and serve the text embeddings:
```bash
curl -X POST -H "Content-Type: application/json" -d '["hello", "world"]' http://localhost:8080/embeddings
```
This requires 2.4 GB of RAM to serve the model.

