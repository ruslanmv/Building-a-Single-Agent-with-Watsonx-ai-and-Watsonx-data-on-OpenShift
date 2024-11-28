# Building a Single Agent with Watsonx.ai and Watsonx.data on OpenShift  
 

Welcome to a complete guide to building a single-agent system integrating **Watsonx.ai** and **Watsonx.data**. This blog outlines a four-step process for creating a robust application capable of querying Watsonx.data, generating responses with Watsonx.ai, and deploying on OpenShift with a Gradio-based chatbot interface.  

---

## 📋 Overview  

This system is built in four parts:  
1. **Setup of Watsonx.data**: Configure Watsonx.data for storing and retrieving structured data.  
2. **Setup of Watsonx.ai**: Create embeddings and use Watsonx.ai for generating responses.  
3. **Setup of the Agent Backend with Langraph**: Design the backend logic to connect data and language models.  
4. **Setup of the Chatbot Frontend**: Build a Gradio interface for user interactions and deploy the system.  

---

## 🛠️ Prerequisites  

### Tools and Services  
- **Python 3.8+**  
- **IBM Cloud Account**  
- **Watsonx.ai API Key**  
- **Watsonx.data Instance**  
- **OpenShift Cluster**  

### Environment Setup  
- Python virtual environment  
- Docker/OpenShift CLI  

---

## 1. Setup of Watsonx.data  

### 1.1 Load Wikipedia Data into Watsonx.data  

#### Install Required Libraries  

```bash
!pip install ipython-sql==0.4.1
!pip install sqlalchemy==1.4.46
!pip install "pyhive[presto]"
!pip install python-dotenv
!pip install wikipedia
```  

#### Fetch Wikipedia Articles  

```python
import wikipedia

articles = {
    'Climate change': None, 
    'Climate change mitigation': None
}

for title in articles.keys():
    article = wikipedia.page(title)
    articles[title] = article.content
    print(f"Successfully fetched: {title}")
```  

#### Chunk and Insert Data  

```python
def split_into_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

split_articles = {title: split_into_chunks(content, 225) for title, content in articles.items()}
```  

#### Insert Data into Watsonx.data  

```python
from sqlalchemy import create_engine

# Connection details
LH_HOST_NAME = 'localhost'
LH_PORT = 8443
LH_USER = 'admin'
LH_PW = 'password'
LH_CATALOG = 'tpch'
LH_SCHEMA = 'tiny'

engine = create_engine(
    f"presto://{LH_USER}:{LH_PW}@{LH_HOST_NAME}:{LH_PORT}/{LH_CATALOG}/{LH_SCHEMA}",
    connect_args={"protocol": "https", "requests_kwargs": {"verify": False}}
)

# Insert data
for title, chunks in split_articles.items():
    for i, chunk in enumerate(chunks):
        escaped_chunk = chunk.replace("'", "''").replace("%", "%%")
        insert_stmt = f"INSERT INTO hive_data.watsonxai.wikipedia VALUES ('{i+1}', '{escaped_chunk}', '{title}')"
        with engine.connect() as connection:
            connection.execute(insert_stmt)
    print(f"Inserted: {title}")
```  

---

## 2. Setup of Watsonx.ai  

### 2.1 Install Milvus and Dependencies  

```bash
!pip install grpcio==1.60.0
!pip install pymilvus
!pip install sentence_transformers
```  

### 2.2 Create Milvus Collection  

```python
from pymilvus import connections, FieldSchema, DataType, Collection, CollectionSchema

connections.connect(alias='default', host='localhost', port='19530')

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="article_text", dtype=DataType.VARCHAR, max_length=2500),
    FieldSchema(name="article_title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
]

schema = CollectionSchema(fields, "Schema for Wikipedia articles")
wiki_collection = Collection(name="wiki_articles", schema=schema)

index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
wiki_collection.create_index(field_name="vector", index_params=index_params)
```  

### 2.3 Insert Data into Milvus  

```python
from sentence_transformers import SentenceTransformer

articles_df = pd.read_sql_query("SELECT * FROM hive_data.watsonxai.wikipedia", engine)
model = SentenceTransformer('sentence-transformers/all-minilm-l12-v2')

passages = articles_df['text'].tolist()
titles = articles_df['title'].tolist()
vectors = model.encode(passages)

data = [passages, titles, vectors]
wiki_collection.insert(data)
wiki_collection.flush()
```  

### 2.4 Query Milvus and Generate Responses  

#### Query Function  

```python
def query_milvus(query, num_results=5):
    query_vector = model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = wiki_collection.search(data=query_vector, anns_field="vector", param=search_params, limit=num_results)
    return [result.entity.get("article_text") for result in results[0]]
```  

#### Generate Answers with Watsonx.ai  

```python
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

def generate_answer(context, question):
    prompt = f"{context}\n\nPlease answer the question: {question}"
    params = {GenParams.DECODING_METHOD: "greedy", GenParams.MAX_NEW_TOKENS: 300}
    model = Model(model_id="ibm/granite-13b-chat-v2", params=params, credentials={"apikey": "your-api-key", "url": "cloud.ibm.com"})
    return model.generate_text(prompt)
```  

---

## 3. Setup of the Agent Backend with Langraph  

### Define the Agent  

```python
from ibm_watsonx_ai.foundation_models import WatsonxLLM  
from sqlalchemy.engine import create_engine  

class WatsonxAgent:  
    def __init__(self, ai_key, data_url):  
        self.llm = WatsonxLLM(  
            model_id="meta-llama/llama-3-70b-instruct",  
            apikey=ai_key  
        )  
        self.data_engine = create_engine(data_url)  

    def query_data(self, query):  
        with self.data_engine.connect() as conn:  
            result = conn.execute(query)  
            return [row for row in result]  

    def generate_response(self, user_query):  
        return self.llm.generate_response(prompt=user_query)  

    def process_query(self, user_query):  
        data_query = f"SELECT * FROM hive_data.watsonxai.wikipedia WHERE text LIKE '%{user_query}%'"  
        data_result = self.query_data(data_query)  

        if data_result:  
            context = "\n".join([row['text'] for row in data_result])  
            ai_query = f"Using this context: {context}, answer: {user_query}"  
        else:  
            ai_query = f"No relevant data found. Answer: {user_query}"  

        return self.generate_response(ai_query)  
```  

---

## 4. Setup of the Chatbot Frontend  

### Build the Chatbot Interface  

```python
import gradio as gr  

def chatbot_interface(user_input):  
    agent = WatsonxAgent(ai_key="your_watsonx_ai_api_key", data_url="your_watsonx_data_url")  
    return agent.process_query(user_input)  

interface = gr.Interface(  
    fn=chatbot_interface,  
    inputs="text",  
    outputs="text",  
    title="Watsonx.ai and Watsonx.data Chatbot"  
)  

if __name__ == "__main__":  
    interface.launch(server_name="0.0.0.0", server_port=7860)  
```  

---

## Conclusion  

This guide walked you through creating a single-agent system with **Watsonx.ai** and **Watsonx.data**, storing data, generating responses, and deploying it with a Gradio-based chatbot interface. By leveraging these tools, you’ve built a foundation for advanced applications.  

Want to scale up? Add multiple agents, integrate additional datasets, or refine your workflow!  

