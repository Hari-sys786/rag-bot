from langchain_ollama import ChatOllama
import time
llm = ChatOllama(model="phi3")
start = time.time()
print(llm.invoke("Say hello").content)
print("LLM time:", time.time() - start)