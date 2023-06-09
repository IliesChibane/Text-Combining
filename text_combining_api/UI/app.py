import gradio as gr

textfields = []
visible_textfields = 10

def combine(*text):
    text = " ".join(text)
    return text
num_textfields = 10


with gr.Blocks() as demo:

    def show_fn():
        global visible_textfields
        global textfields

        if visible_textfields < 10:
            visible_textfields += 1
            textfields[visible_textfields-1].update(visible=True) 

    def hide_fn():
        global visible_textfields
        global textfields

        if visible_textfields > 0:
            visible_textfields -= 1
            textfields[visible_textfields].update(visible=False) 

    def submit_fn():
        text_values = [textfield.value for textfield in textfields]
        combined_text = combine(*text_values)
        output_textbox.value = combined_text

    def clear_fn():
        for textfield in textfields:
            textfield.value = ""

    with gr.Row():
        with gr.Column(scale=1):
            
            for i in range(3):
                textfield = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte {i+1}')
                textfields.append(textfield)
            
            clean_btn = gr.Button("Nettoyer")
            submit_btn = gr.Button("Combiner")

        with gr.Column(scale=2):
            output_textbox = gr.outputs.Textbox(label="Texte Combiné")

            plus_btn = gr.Button("+")
            minus_btn = gr.Button("-")

    clean_btn.click(clear_fn)
    submit_btn.click(submit_fn)
    plus_btn.click(show_fn)
    minus_btn.click(hide_fn)

demo.launch()
