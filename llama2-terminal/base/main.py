import LlamaBase

agent = LlamaBase.build_llama_agent()
while True:
    query = input("Query: ")
    print(LlamaBase.get_prediction(agent,query))
    