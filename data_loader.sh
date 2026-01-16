#!/bin/bash
directory="./data"
upload_url="http://localhost:8081/chunks"

found_files=false

for file in "$directory"/*.pdf; do
  if [ -f "$file" ]; then
    found_files=true
    echo "Uploading $file..."

    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
      "$upload_url?return_chunks=false&image_model=qwen2.5vl:3b&text_model=llama3.2:3b" \
      -H "accept: application/json" \
      -F "file=@$file;type=application/pdf")

    if [ "$response" -eq 200 ]; then
      echo "Successfully uploaded $file"
    else
      echo "Failed to upload $file. HTTP Status Code: $response"
    fi
  fi
done

if [ "$found_files" = false ]; then
  echo "No PDF files found in the directory."
fi

echo "All files have been processed."
