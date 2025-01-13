# Cheat Sheet: Fundamentals of Building AI Agents using RAG and Lang Chain

## Packages and Methods

### Generate Text
Generates text sequences based on input without computing gradients.

```python
# Generate text
output_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50,
    num_return_sequences=1
)

# Alternative
with torch.no_grad():
    outputs = model(**inputs)
outputs
```

---

### Formatting Prompts Function
Generates formatted text prompts from a dataset.

```python
def formatting_prompts_func(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"###\nInstruction:\n{mydataset['instruction'][i]}"
            f"\n\n###\nResponse:\n{mydataset['output'][i]}</s>"
        )
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_no_response(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"###\nInstruction:\n{mydataset['instruction'][i]}"
            f"\n\n### Response:\n"
        )
        output_texts.append(text)
    return output_texts
```

---

### `torch.no_grad()`
Generates text sequences with gradient computations disabled for optimized performance and memory usage.

```python
with torch.no_grad():
    # Due to resource limitation, only apply the function on 3 records using "instructions_torch[:10]"
    pipeline_iterator = gen_pipeline(
        instructions_torch[:3],
        max_length=50,  # Increase if using GPU
        num_beams=5,
        early_stopping=True,
    )

    generated_outputs_lora = []
    for text in pipeline_iterator:
        generated_outputs_lora.append(text[0]["generated_text"])
```

---

### `mixtral-8x7b-instruct-v01`
Adjusts parameters to optimize creativity and response length.

```python
model_id = 'mistralai/mixtral-8x7b-instruct-v01'
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}
project_id = "skills-network"
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
```

---

### String Prompt Templates
Formats a single string, typically for simpler inputs.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}
prompt.invoke(input_)
```

---

### Chat Prompt Templates
Formats a list of messages for complex conversational structures.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

input_ = {"topic": "cats"}
prompt.invoke(input_)
```

---

### Messages Placeholder
Allows dynamic insertion of user-provided message lists.

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]} 
prompt.invoke(input_)
```

---

### Example Selectors
Selects examples dynamically for Few-Shot learning.

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```

---

### JSON Parser
Parses output into JSON format with a specified schema.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

output_parser = JsonOutputParser(pydantic_object=Joke)

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | mixtral_llm | output_parser

chain.invoke({"query": "Tell me a joke."})
```

---

### Comma Separated List Parser
Used to return a list of comma-separated items.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | mixtral_llm | output_parser
```

---

### Document Object
Contains information about some data in LangChain.

```python
from langchain_core.documents import Document

Document(
    page_content="""Python is an interpreted high-level general-purpose programming language. 
    Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
    metadata={
        'my_document_id': 234234,
        'my_document_source': "About Python",
        'my_document_create_time': 1680013019
    }
)
```

---

### Text Splitter
Splits text into semantically meaningful chunks.

```python
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)
print(len(chunks))
```

---

### Embedding Models
Generates a vector representation for a given text.

```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

from langchain_ibm import WatsonxEmbeddings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
```

---

### Vector Store-Backed Retriever
A retriever that uses a vector store for querying.

```python
retriever = docsearch.as_retriever()

docs = retriever.invoke("Langchain")
```

---

### ChatMessageHistory Class
Stores and manages AI-human message exchanges.

```python
from langchain.memory import ChatMessageHistory

chat = mixtral_llm

history = ChatMessageHistory()

history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")
```

---

### LangChain Chains
Creates chains for generating recommendations or other tasks.

```python
from langchain.chains import LLMChain

template = """Your job is to come up with a classic dish from the area that the users suggest.\n{location}\nYOUR RESPONSE:\n"""

prompt_template = PromptTemplate(template=template, input_variables=['location'])

location_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='meal')

location_chain.invoke(input={'location': 'China'})
```

---

*Further details continue with specific task-based LangChain examples. Let me know if the entire remaining content is needed!*
