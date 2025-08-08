import gradio as gr
from rag_bot.retrieval.retriever import answer_question

def rag_chatbot(message, history):
    answer, sources = answer_question(message, top_k=5)
    sources_str = "\n".join(sources)
    output = f"{answer}\n\n**Sources:**\n{sources_str}"
    return output

with gr.Blocks() as demo:
    gr.Markdown("## :robot_face: IT Infra RAG Chatbot Demo")
    chatbot = gr.ChatInterface(
        fn=rag_chatbot,
        chatbot=gr.Chatbot(),
        title="IT Infra Assistant"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=True)
