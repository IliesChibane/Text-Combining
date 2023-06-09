import gradio as gr

textfields = []
visible_textfields = 10

for i in range(10):
    textfield = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte {i+1}')
    textfields.append(textfield)

def combine(*text):
    text = " ".join(text)
    return text

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
        textfields[visible_textfields-1].update(visible=False)
        visible_textfields -= 1

with gr.Interface(
    fn=combine,
    inputs=textfields,
    outputs=gr.outputs.Textbox(label="Texte Combiné"),
    title="Text Combiner"
) as iface:
    iface.launch()
