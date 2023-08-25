from llama2terminal.base.agents import LlamaConversationalAgent

agent = LlamaConversationalAgent()
while True:
    query = input("Query: ")
    if query == "exit":
        break
    print(agent.get_prediction(query))
    