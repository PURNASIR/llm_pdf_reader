Test in Postman
🔹 1️⃣ Upload PDF + Create Chroma DB
POST → http://127.0.0.1:8000/upload_and_embed

| KEY       | VALUE                                                                      | TYPE |
| --------- | -------------------------------------------------------------------------- | ---- |
| file      | *(choose PDF)*                                                             | File |
| json_data | `{"pdf_folder": "E:/AIMODEL/input", "output_folder": "E:/AIMODEL/output"}` | Text |


2️⃣ Query Existing Chroma DB

POST → http://127.0.0.1:8000/query_chroma

Body → form-data:

KEY	VALUE	TYPE
output_folder	E:/AIMODEL/output	Text
question	What is this PDF about?	Text
