An example on using [huggingface/candle](https://github.com/huggingface/candle) with Golang. For educational purposes only.
The implementation is thread-safe and uses [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) model.

On of the use cases for text embeddings is to provide semantic search:
```
> go test -run TestEmbeddings
top3 for 'query: burnouts and how to deal with them':
dist: 0.212324  'query: overworking leads to depression'
dist: 0.228119  'query: protein shakes and other stuff'
dist: 0.231076  'query: the cause of bad behaviour'
top3 for 'query: feline anatomy':
dist: 0.111701  'query: cat's body'
dist: 0.241103  'query: protein shakes and other stuff'
dist: 0.245675  'query: the works of Francis Bacon'
top3 for 'query: 16th century philosophers':
dist: 0.092926  'query: 18th century philosophers'
dist: 0.230078  'query: Critique of Pure Reason'
dist: 0.236497  'query: the works of Francis Bacon'
top3 for 'query: overworking leads to depression':
dist: 0.186588  'query: the cause of bad behaviour'
dist: 0.212324  'query: burnouts and how to deal with them'
dist: 0.239519  'query: what the reason for being not nice'
top3 for 'query: Critique of Pure Reason':
dist: 0.181644  'query: the cause of bad behaviour'
dist: 0.225016  'query: 18th century philosophers'
dist: 0.226892  'query: what the reason for being not nice'
top3 for 'query: the books of Immanuel Kant':
dist: 0.188276  'query: the works of Francis Bacon'
dist: 0.250316  'query: Critique of Pure Reason'
dist: 0.252901  'query: 18th century philosophers'
...
```

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

