The pdf(History_of_Bangladeh.pdf) is a doc (later saved as pdf) taken from Bangladesh goverment's official website of Bangladesh Freedom Fighters Wellfare Trusts(bffwt)
link: https://bffwt.portal.gov.bd/sites/default/files/files/bffwt.portal.gov.bd/page/e2b55969_0e0c_4337_bb4e_ddcc801df7db/History%20of%20Bangladesh%20(4).docx
Date of downloading the document: 15th February, 2025 (11:40 PM)

# Practical_RAG
This repo shows different RAG apps utilizing different approaches. 

## Instructions

1. Create a `.env` file with the following content:
    ```bash
    google_api_key = "your_api_key_here"
    ```

2. Create a virtual environment:
    ```bash
    python3.11 -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the program:
    ```bash
    streamlit run NAIVE_RAG_streamlit.py
    ```
    you can also run individual rag file (HyDRAG for example) by running
    ```
    python HyDRAG.py
    ```

## Requirements
- Python 3.11.8
