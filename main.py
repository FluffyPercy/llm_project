import gradio as gr
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import PIL.Image
import time
import pandas as pd
import json

########################## Load data ###########################
# Load vaccine code
with open('./vaccine_code.txt') as f:
    vaccine_code = f.read()

vaccine_code = json.loads(vaccine_code)   #convert to dict
print('------------------------- vaccine code loaded -------------------------')


# Load vaccine data
dataset = {}
for year in range(2014, 2024):
    dataset[f'{year}data'] = pd.read_csv(f'./data/{year}VAERSData/{year}VAERSDATA.csv', encoding='latin1')
    dataset[f'{year}symp'] = pd.read_csv(f'./data/{year}VAERSData/{year}VAERSSYMPTOMS.csv', encoding='latin1')
    dataset[f'{year}vax'] = pd.read_csv(f'./data/{year}VAERSData/{year}VAERSVAX.csv', encoding='latin1')
print('------------------------- vaccine data loaded -------------------------')



########################## Initiate LLM ###########################
genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="You are a vaccine data assistant. You will have access to some VAERS datasets on human vaccine adverse event reports from the US when needed.\
        The datasets is not from a rigirous epidemiology study, do not contain general information of each vaccine, \
        and do not reflect the outcome of all vaccine recipients, but only on those that voluntarily reported adverse events.\
        Be caring and profesional. Give concise answers. Focus your main discussion on vaccines data and the information you can infer from them.\
        Provide additional information with caution. Do NOT give medical advises, instead refer the user to a doctor.",
    )

chat = model.start_chat(history=[])




########################## Define different task performers ###########################

# image analysis
def image_assistant(input_image):
    instruction_prompt = 'If the image is of the skin of a human body part, produce a JSON summary with the following fields: \
        position, estimated_size, shape, color, texture, abnomality. Put "unsure" as the value if unsure for one field.'
    example_prompt = 'Example output: {"position": "neck","estimated_size": "10cm", "shape": "circular","color": "brown", "texture": "smooth", "abnomality": false}.\
        Example output:{"position": "unsure","estimated_size": "1cm-5cm", "shape": "irregular","color": "brown", "texture": "unsure", "abnomality": unsure}.'
    result = model.generate_content([input_image, '\n\n', instruction_prompt+example_prompt])
    return result.text


# determine if access data
def data_reqeust(input_text):
    instruction_prompt = "Instruction: You are to determine if the given input is relevant to the VAERS datasets. Respond '0' for false and '1' for true.\
        This includes inquiries on general statistics on adverse events, i.e. side effects, not limited to the US. The datasets only concern humans."
    example_prompt = 'Example input-output: "What are the most common symptoms caused by COVID vaccines?", "1".\
        Example input: "Should I get the TBE vaccine"; output:"0".\
        Example input: "What are the most popular vaccines in the US?"; output:"0".\
        Example input: "Are old people more suseptible to vaccine side effect?"; output:"1".\
        Example input: "Can I eat vaccine?"; output:"0".\
        Example input: "Should my dog get vaccinated?"; output:"0".'
    result = model.generate_content([input_text, '\n\n', instruction_prompt+example_prompt])
    try:
        return int(result.text)
    except:
        print('Data request failed.')


# determin relevant data
def data_assistant(input_text):
    instruction_prompt = "Instruction: Your task is to write Python code to access the VAERS datasets.\
        You are given a dataset in the format of a Python dictionary, called 'dataset'.\
        The dataset consists of passive adverse event reports related to vaccines in the US between year 2014 to year 2023.\
        The dataset dictionary has keys of the following format: \
        '{year}data', '{year}symp', '{year}vax', etc, and the values are panda dataframes.\
        where you can replacing {year} with year numbers between 2014 to 2023 (incl.). All files have headers.\
        All files from the same year contain the same patients, identified by their VAERS_ID.\
        The '{year}data' files contain ganeral information of the patients,\
        the '{year}symp' files contain discriptions of their symptoms,\
        the '{year}vax' files contain discriptions of the vaccines they recieved.\
        In addition, you are given a Python dictionary called 'vaccine_code', where ther keys are the vaccine codes that you see in '{year}vax',\
        and the values are the actual names of the vaccines.\
        You need to analyse the data needed from the request, write ... " ##########??? find how to access pd dataframe
    example_prompt = 'Example output: ' #############??? give examples, can use examples in data_request
    result = model.generate_content([input_text, '\n\n', instruction_prompt+example_prompt])
    return result.text

# user facing component
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



########################## Assemble chatbot ###########################

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
