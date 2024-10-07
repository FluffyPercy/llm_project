import gradio as gr
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import PIL.Image
import time
import pandas as pd


genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="You are a vaccine data expert. Be caring but profesional and focus your discussion on vaccines and their possible side effects",
    )



chat = model.start_chat(history=[])



def image_assistant(input_image):
    instruction_prompt = 'If the image is of the skin of a human body part, produce a JSON summary with the following fields: position, estimated_size, shape, color, texture, abnomality. Put "unsure" as the value if unsure.'
    example_prompt = 'Example output: {"position": "neck","estimated_size": "10cm", "shape": "circular","color": "brown", "texture": "smooth", "abnomality": false}'
    result = model.generate_content([input_image, '\n\n', instruction_prompt+example_prompt])
    return result.text



def response(inputs, history):
    message = [inputs["text"]]
    if len(inputs["files"]) != 0:
        try:
            sample_image = PIL.Image.open(inputs["files"][0]["path"])
            message.append(sample_image)
            info = image_assistant(sample_image)
            history.append(info)
            print(info)
            if len(message[0]) == 0:
                message[0] = f'Please summarise the input information using the following information: {info}'
        except Exception:
            print('Only allow image upload.')
            yield '[File not supported]'
            return None
    else:
        pass
    response = chat.send_message(message, stream=True, safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE})
    full_response = ""
    for partial_response in response:
        time.sleep(0.2)
        try:
            full_response += partial_response.text
        except Exception:
            print('Safety filter triggered.')
            chat.rewind()
            yield '[Safety filter triggered]'
            return None
        yield full_response



with gr.Blocks(fill_height=True, fill_width=True) as demo:
    chatbot = gr.ChatInterface(
        fn=response,
        title="VAERS Vaccine Database Assitant",
        multimodal=True,
        description='An assistant for accessing the VAERS data from the past 10 years. **Warning: Not suitable for actual medical practice.**',
        examples=[{'text':"Are children or seniors more susceptible to COVID vaccine side effects?"}, {'text':"Print a list of all reports on flu."}]
    )




if "demo" in locals() and demo.is_running:
    print('yes')
    demo.close()


if __name__ == "__main__":
    demo.launch()



# problem1: must terminate manually
# problem2: gradio textbox is partially blank