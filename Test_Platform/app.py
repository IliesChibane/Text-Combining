import gradio as gr

visible = 0

def combine(texts):
    return " ".join(texts)

def show_fn(text3, text4, text5, text6, text7, text8, text9, text10):
    global visible
    if visible == 0:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=False),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 1:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 2:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 3:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 4:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 5:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 6:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=True),
                textfield10: gr.update(visible=False),
        }
    elif visible == 7:
        visible += 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=True),
                textfield10: gr.update(visible=True),
        }
    else:
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=True),
                textfield10: gr.update(visible=True),
        }

def hide_fn(text3, text4, text5, text6, text7, text8, text9, text10):
    global visible

    if visible == 9:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=True),
                textfield10: gr.update(visible=True),
        }
    if visible == 8:
            visible -= 1
            return {textfield3: gr.update(visible=True),
                    textfield4: gr.update(visible=True),
                    textfield5: gr.update(visible=True),
                    textfield6: gr.update(visible=True),
                    textfield7: gr.update(visible=True),
                    textfield8: gr.update(visible=True),
                    textfield9: gr.update(visible=True),
                    textfield10: gr.update(visible=False),
            }
    if visible == 7:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=True),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    if visible == 6:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=True),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    if visible == 5:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=True),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    if visible == 4:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=True),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    if visible == 3:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=True),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 2:
        visible -= 1
        return {textfield3: gr.update(visible=True),
                textfield4: gr.update(visible=False),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    elif visible == 1:
        visible -= 1
        return {textfield3: gr.update(visible=False),
                textfield4: gr.update(visible=False),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }
    else:
        return {textfield3: gr.update(visible=False),
                textfield4: gr.update(visible=False),
                textfield5: gr.update(visible=False),
                textfield6: gr.update(visible=False),
                textfield7: gr.update(visible=False),
                textfield8: gr.update(visible=False),
                textfield9: gr.update(visible=False),
                textfield10: gr.update(visible=False),
        }

def submit_fn(text1, text2, text3, text4, text5, text6, text7, text8, text9, text10):
    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]
    return combine(texts)

def clear_fn(text1, text2, text3, text4, text5, text6, text7, text8, text9, text10):
    return {textfield1: gr.update(value=""),
            textfield2: gr.update(value=""),
            textfield3: gr.update(value=""),
            textfield4: gr.update(value=""),
            textfield5: gr.update(value=""),
            textfield6: gr.update(value=""),
            textfield7: gr.update(value=""),
            textfield8: gr.update(value=""),
            textfield9: gr.update(value=""),
            textfield10: gr.update(value=""),
    }


with gr.Blocks() as demo:

    gr.Markdown("<center><h1>Text Combining Test Platform</h1></center> <center><h2>End of the Year Project</h2></center>")

    with gr.Row():
        with gr.Column(scale=1):

            textfield1 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 1')
            textfield2 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 2')
            textfield3 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 3', visible=False)
            textfield4 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 4', visible=False)
            textfield5 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 5', visible=False)
            textfield6 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 6', visible=False)
            textfield7 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 7', visible=False)
            textfield8 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 8', visible=False)
            textfield9 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 9', visible=False)
            textfield10 = gr.Textbox(lines=4, placeholder="Votre texte...", label=f'Texte 10', visible=False)
            
            plus_btn = gr.Button("+")
            minus_btn = gr.Button("-")

            plus_btn.click(show_fn, 
                            [textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10],
                            [textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10])

            minus_btn.click(hide_fn, 
                            [textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10], 
                            [textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10])


        with gr.Column(scale=2):
            output_textbox = gr.Textbox(label="Texte Combiné")

            clean_btn = gr.Button("Nettoyer")
            submit_btn = gr.Button("Combiner")

            clean_btn.click(clear_fn, 
                            [textfield1, textfield2, textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10], 
                            [textfield1, textfield2, textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10])
            submit_btn.click(submit_fn,
                            [textfield1, textfield2, textfield3, textfield4, textfield5, textfield6, textfield7, textfield8, textfield9, textfield10],
                            output_textbox)

if __name__ == "__main__":
    demo.launch()
