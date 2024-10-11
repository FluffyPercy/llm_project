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
    system_instruction="You are a (human) vaccine data expert. \
        You will begiven some VAERS datasets on human vaccine adverse event reports from 2014 to 2013 (inlc) from the US when needed.\
        The datasets include detailed information including specific vaccine type, e.g. plague vaccine, covid vaccine, etc.\
        You may use the datasets as examples if no specific data is required. \
        You may use the datasets as examples even when the question is not limited to the US.\
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



# data retrieval
def data_assistant(input_text, vc = vaccine_code, ds = dataset):
    # find the relevant information in the input
    instruction_condense = 'Instruction: Find the relevant components in the input and generate a JSON file with possible fields of\
         "year", "ID", "vaccine", "disease", "symptoms", "age", "sex", "died" or other info that you deem relevant, where one ID represents a patient or person or report.\
        If the input has a field requiring specific information, e.g. vaccine name, symptoms and age, return in the format\
            {"vaccine": "(vaccine name)", "symptoms": "(symptom name)", "age": (int) or string discription (young, old, etc.)}.\
        If the input has a field requiring general information, e.g. all symptoms concerning COVID vaccine, return in the format\
            {"symptoms": "all", "vaccine": "COVID"}.\
        The "year" key can only have value of an integer between 2014 to 2023 (incl.) or a list of such integers; "recent years" count as the last three years;\
            do not include "year" key if unsure of year(s).\
        If both women and men are mentioned, set value "all" for "sex" key.\
        If the input does not concern the datasets/reports, vaccine side effects or vaccine receivers, respond "None". \n\n'
    # example prompt
    instruction_condense += 'Example input-output: "What are the most common symptoms caused by corona vaccines?", "{"symptoms": "all", "vaccine": "COVID"}".\n'
    instruction_condense += 'Example input-output: "Should I get TBE vaccine?", "{"vaccine": "TBE"}".\n'
    instruction_condense += 'Example input-output: "What are the most popular vaccines in the US?", "None".\n'
    instruction_condense += 'Example input-output: "Give me all the reports in recent years", "{"years": [2021, 2022, 2023]}".\n'
    instruction_condense += 'Example input-output: "Give me all the reports on TBE vaccines in recent years", "{"vaccine": "TBE", "years": [2021, 2022, 2023]}".\n'
    instruction_condense += 'Example input-output: "Are old people more suseptible to vaccine side effect?", "{"age": "old"}".\n'
    instruction_condense += 'Example input-output: "Should my dog get vaccinated?", "None".\n'
    instruction_condense += 'Example input-output: "Do women or men get more side effects from COVID vaccines?", "{"vaccine": "COVID", "sex": "all"}".\n'
    instruction_condense += 'Example input-output: "How many people reported in year 2018 and 2019?", "{"ID": "all", "year": [2018, 2019]}".\n'
    instruction_condense += 'Example input-output: "What is the trend of adverse event reports for polio vaccine?", "{"vaccine": "polio"}".\n'
    instruction_condense += 'IMPORTANT: only respond with above JSON format or "None".'
    # generated condensed input
    input_condensed = model.generate_content([input_text, '\n\n', instruction_condense]).text
    print('Data request:', input_condensed)
    if 'None' in input_condensed:
        return None
    else:
        pass
    # instrcution prompt
    instructions = f"INSTRUCTION: You are assisting a programmer on data retrieval. \
        Follow the following instruction to give information for accessing the relevant parts of the VAERS datasets,\
        based on the information given in the input.\n\n"
    instructions += "DATASET: The intended Python code will perform on datasets in the format of a Python dictionary, called 'dataset'. \
        The dataset consists of passive adverse event reports related to vaccines in the US between year 2014 to year 2023.\
        The dataset dictionary has keys of the following format: \
        '{year}data', '{year}symp', '{year}vax', where you can replacing {year} with year numbers between 2014 to 2023 (incl.), \
        and the values are panda dataframes.\
        All files from the same year contain the same patients, identified by their VAERS_ID.\
        The '{year}data' files contain ganeral information of the patients,\
        the '{year}symp' files contain discriptions of their symptoms,\
        the '{year}vax' files contain discriptions of the vaccines they recieved.\n"
    instructions += f'''STEPS: Follow these steps for the task:\n\
        1. Example dataframe: Take a look at the first few lines of the files from 2014 and use this as reference for the data structure,\n\
            '2014data':\n{ds['2014data'].head()}\n\
            '2014symp':\n{ds['2014symp'].head()}\n\
            '2014vax':\n{ds['2014vax'].head()}\n\n\
        2. Use the input dictionary to determine the objective: 'filenamn', 'filter'('column', 'trait'), 'info'.\
            For 'filename', if there is a 'year' key, consider only and all file names containing those years;\
            each key with 'all' as value should be mapped to a column name in a file and be considered as 'info';\
            all other keys are consider 'filter', where each one is mapped to a column name and its value considered as 'trait';
            if applicable, consider a few synonyms for 'trait' for the next step.\
            If a vaccine or disease is mentioned, refer to the dict below of all vaccines in the dataset, which includes their codes and names,\n\
                VAX_CODE: \n{vc}.\n\
            If the requied vaccine or decease has only a vague description, e.g. corona or flu, you should take all relevant vaccines in to consideration.\
            If year is not specified, consider all years (2014-2013 incl.).\
            If a symptom is mentioned, you should consider all similar symptoms, e.g. 'sick' and 'nausea';\n'''
    instructions += '3. Output a list of dictionaries of the format:\
            [{"filename_1": filename, "filter": {"trait_column_1": ["trait"], "trait_column_2": ["trait",..., "trait"]}, "info": ["info_column", "info_coumn"]}, ..., \
                {"filename_n": filename, "filter": {"trait_column_1": ["trait",..., "trait"]}, "info": ["info_column",..., "info_column"]}, \
                sub_VAX_CODE].\n\
            All trait_column and info_column should be actual columns in a file with filename. All traits should be possible values under the column, infered from input.\
                "VAX_TYPE" traits should be vaccine codes given as keys in VAX_CODE.\
                If need trait_column and info_column from different files, draw these files seperately.\n\
            The purpose of this output is to later perform i)finding the correct files: i)for each "filename_k", \n\
                file_df = ds[filename];\
            ii) filtering of VAERS_ID numbers accoridng to traits: for each "trait_column_m",\n\
                ID_m = file_df.loc[file_df["trait_column_m"] in ["trait",..., "trait"], "VAERS_ID"].tolist(); \n\
            iii) let ID_filtered be the intersection of ID_1,..., ID_m, then use it to generate a filtered-dataframe from the same file or a different file of the same year with these traits:\n\
                filtered_df = ds[filename].loc[df["VAERS_ID"].isin(ID_filtered)]; and\n\
            iv) return filtered dataframe, or further extract only relevant columns:\n\
                sub_df["info_column", ..., "info_column"].\n\
            v) after all above is done for all files involved, the a sub-dict of VAX_CODE, denoted by sub_VAX_CODE, is used for further referecne,\
                which contains all vaccines relevant to what is meantioned in the input (as vaccine or disease). Retrun empty dict if no specific vaccine or disease is concerned.\
            Explanation of the output:\
                The list contains information of all files needed. For each file, filename, relevant trait filters and relevant columns are given.\
                    You may call each file at most once.\
                    All filter_columns and info_columns must be present in the file, as shown in the example dataframe in step 1. \
                    There can be duplicated "trait_column" if different traits are needed; retrun empty dict for "filter" if no "trait_column"-"trait" pair is present.\
                    If multiple string traits are given, they should be less than 5 and they should be synomyms, e.g. "sick" and "nausea".\
                    Return empty list as value of "info" if no specific column is needed.\
                The dictionary contains relevant vaccine codes and names for later reference; return empty disctionary if non specific vaccine is relevant.\
                The string is a concise discription of what the code will do, once inserted the values from the above list.\n\n'
    instructions += "ADDITIONAL INFO: All COVID related requests only concern year 2020 and onwards"
    # example prompt
    instructions += 'Example input: "{"symptoms": "all", "vaccine": "COVID"}";\
        output: [{"filename_1": "2020vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_2": "2020symp", "filter": {}, "info": ["SYMPTOM1", "SYMPTOMVERSION1", "SYMPTOM2", "SYMPTOMVERSION2", "SYMPTOM3", "SYMPTOMVERSION3", "SYMPTOM4", "SYMPTOMVERSION4", "SYMPTOM5", "SYMPTOMVERSION5"]}\
                {"filename_3": "2021vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_4": "2021symp", "filter": {}, "info": ["SYMPTOM1", "SYMPTOMVERSION1", "SYMPTOM2", "SYMPTOMVERSION2", "SYMPTOM3", "SYMPTOMVERSION3", "SYMPTOM4", "SYMPTOMVERSION4", "SYMPTOM5", "SYMPTOMVERSION5"]}\
                {"filename_5": "2022vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_6": "2022symp", "filter": {}, "info": ["SYMPTOM1", "SYMPTOMVERSION1", "SYMPTOM2", "SYMPTOMVERSION2", "SYMPTOM3", "SYMPTOMVERSION3", "SYMPTOM4", "SYMPTOMVERSION4", "SYMPTOM5", "SYMPTOMVERSION5"]}\
                {"filename_7": "2023vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_8": "2023symp", "filter": {}, "info": ["SYMPTOM1", "SYMPTOMVERSION1", "SYMPTOM2", "SYMPTOMVERSION2", "SYMPTOM3", "SYMPTOMVERSION3", "SYMPTOM4", "SYMPTOMVERSION4", "SYMPTOM5", "SYMPTOMVERSION5"]}\
                {"COVID19": "Coronavirus 2019 vaccine", "COVID19-2": "Coronavirus 2019 vaccine, bivalent"}].\n\n'
    instructions += 'Example input: "{"age": "old", "year": "2021, 2022, 2023"}";\
        output: [{"filename_1": "2021data", "filter": {"AGE_YRS", [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]}, "info": []},\
                {"filename_2": "2022data", "filter": {"AGE_YRS", [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]}, "info": []},\
                {"filename_3": "2023data", "filter": {"AGE_YRS", [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]}, "info": []},\
                {}].\n\n'
    instructions += 'Example input: "{"vaccine": "COVID", "sex": "all"}";\
        output: [{"filename_1": "2020vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_2": "2020data", "filter": {}, "info": ["SEX"]}\
                {"filename_3": "2021vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_4": "2021data", "filter": {}, "info": ["SEX"]}\
                {"filename_5": "2022vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_6": "2022data", "filter": {}, "info": ["SEX"]}\
                {"filename_7": "2023vax", "filter": {"VAX_TYPE": ["COVID19", "COVID19_2"]}, "info": []},\
                {"filename_8": "2023data", "filter": {}, "info": ["SEX"]}\
                {"COVID19": "Coronavirus 2019 vaccine", "COVID19-2": "Coronavirus 2019 vaccine, bivalent"}].\n\n'


    instructions += "**IMPORTANT**: ONLY REPLY WITH A LIST OF THE ABOVE FORMAT OR 'None'."
    result = model.generate_content(['Please perform the task for: '+ input_condensed,'\n\n', instructions]).text
    
    ## test
    # print('test, all text output: ', result)
    # print('test, try JSON: ', result.split('\n')[1])
    # test_result = json.loads('[' + result.split('\n')[0].replace("'", '"') + ']')
    # print('test, after JSON', test_result)
    ## test end

    try:
        result = json.loads('[' + result.split('\n')[1].replace("'", '"') + ']')
        print('Data request succeeded: \n', result)
        return result
    except:
        print('Data request failed: \n', result)
        return None

## test
# print('test data_assistant: ', data_assistant('What symptoms are most common among plague vaccine receivers?'))
## test end



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
# problem3: cannot perform statistics
