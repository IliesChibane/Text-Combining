import gradio as gr
from transformers import pipeline

'''generator = pipeline('text-generation', model='gpt2')'''

def combine(text):
    '''result = generator(text, max_length=30, num_return_sequences=1)
    return result[0]["generated_text"]'''
    return 'yey'

examples = [
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

demo = gr.Interface(
    fn=combine,
    inputs=[gr.Textbox(lines=4, placeholder="Votre texte...",label='Texte 1'), gr.Textbox(lines=4, placeholder="Votre texte...",label='Texte 2')],
    outputs=gr.outputs.Textbox(label="Texte Combine"),
    examples=examples
)

demo.launch()