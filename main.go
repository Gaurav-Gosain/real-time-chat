package main

// simple file server to serve static files in the wav folder

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		file, err := os.Open(filepath.Join("wav", r.URL.Path))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer file.Close()

		// Serve the file
		if r.Method == "GET" {
			w.Header().Set("Content-Type", "audio/wav")
			w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", r.URL.Path))
			io.Copy(w, file)
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	log.Fatal(http.ListenAndServe(":9000", nil))
}
