package main

/*
#cgo LDFLAGS: -L../target/release -ltxtvec
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    float* data;
    uintptr_t len;
} FloatArray;

typedef struct {
    FloatArray* data;
    uintptr_t len;
} FloatArrayArray;

FloatArrayArray process_strings(const char** strings, const uintptr_t* lengths, uintptr_t count);
void free_float_array_array(FloatArrayArray data);
*/
import "C"
import (
	"unsafe"
)

// CreateEmbeddings creates embeddings for the given input strings.
func CreateEmbeddings(input []string) [][]float32 {
	cStrings := make([]*C.char, len(input))
	defer func() {
		// Free the C strings
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	lengths := make([]C.uintptr_t, len(input))
	for i, s := range input {
		cStrings[i] = C.CString(s)
		lengths[i] = C.uintptr_t(len(s))
	}

	result := C.process_strings(&cStrings[0], &lengths[0], C.uintptr_t(len(input)))
	defer func() {
		C.free_float_array_array(result)
	}()

	// Convert C result to Go slices
	goResult := make([][]float32, result.len)
	for i := 0; i < int(result.len); i++ {
		floatArray := (*C.FloatArray)(unsafe.Pointer(uintptr(unsafe.Pointer(result.data)) + uintptr(i)*unsafe.Sizeof(C.FloatArray{})))
		goSlice := (*[1 << 30]float32)(unsafe.Pointer(floatArray.data))[:floatArray.len:floatArray.len]
		goResult[i] = make([]float32, len(goSlice))
		copy(goResult[i], goSlice)
	}
	return goResult
}

type EmbeddingsZeroCopy struct {
	Result [][]float32
	mem    *C.FloatArrayArray
}

func (e *EmbeddingsZeroCopy) Free() {
	C.free_float_array_array(*e.mem)
}

// CreateEmbeddingsZeroCopy creates embeddings for the given input strings. The embeddings are allocated in C memory and
// are not copied to Go memory. The caller must call Free() on the returned object to free the C memory.
func CreateEmbeddingsZeroCopy(input []string) *EmbeddingsZeroCopy {
	cStrings := make([]*C.char, len(input))
	defer func() {
		// Free the C strings
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	lengths := make([]C.uintptr_t, len(input))
	for i, s := range input {
		cStrings[i] = C.CString(s)
		lengths[i] = C.uintptr_t(len(s))
	}

	result := C.process_strings(&cStrings[0], &lengths[0], C.uintptr_t(len(input)))

	// Convert C result to Go slices
	goResult := make([][]float32, result.len)
	for i := 0; i < int(result.len); i++ {
		floatArray := (*C.FloatArray)(unsafe.Pointer(uintptr(unsafe.Pointer(result.data)) + uintptr(i)*unsafe.Sizeof(C.FloatArray{})))
		goSlice := (*[1 << 30]float32)(unsafe.Pointer(floatArray.data))[:floatArray.len:floatArray.len]
		goResult[i] = goSlice
	}
	return &EmbeddingsZeroCopy{
		Result: goResult,
		mem:    &result,
	}
}
