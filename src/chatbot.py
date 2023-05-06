import gradio as gr
from core import ToolMatrix

def add_text(text, history):
    history = history + [(text, None)]
    return history, ""

def add_file(file, history):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    response = "**That's cool!**"
    history[-1][1] = response
    return history

if __name__ == "__main__":
    tool_matrix = ToolMatrix()
    tool_matrix.init_all()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=550)
        state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=0.85):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("📁")

        txt.submit(tool_matrix.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(tool_matrix.run_img, [btn, state], [chatbot, state])
        clear.click(tool_matrix.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch()
