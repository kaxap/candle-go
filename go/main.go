package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	log.Println("loading the model...")
	CreateEmbeddings([]string{"hello"})
	log.Println("model loaded. Serving :8080/embeddings")
	http.HandleFunc("/embeddings", processHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func processHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var inputStrings []string
	if err := json.NewDecoder(r.Body).Decode(&inputStrings); err != nil {
		http.Error(w, "Failed to decode input", http.StatusBadRequest)
		return
	}

	embeddings := CreateEmbeddingsZeroCopy(inputStrings)
	defer embeddings.Free()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(embeddings.Result); err != nil {
		http.Error(w, "Failed to encode output", http.StatusInternalServerError)
	}
}
