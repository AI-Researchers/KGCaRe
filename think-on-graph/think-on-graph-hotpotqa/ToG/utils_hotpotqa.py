from prompt_list_hotpotqa import *
import json
import time
# import openai
import re
from langchain import PromptTemplate #,  LLMChain
from langchain.prompts.chat import ChatPromptTemplate
# from auto_gptq import exllama_set_max_input_length
# import torch
# from langchain_community.llms import VLLM
# from langchain_ollama.llms import OllamaLLM
from langchain.chat_models import ChatOpenAI
import string

llm = None
def initialise_llama3(temperature, max_tokens):
    print("--initialising llama3")
    global llm
    llm = VLLM(
            model = "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_llama2(temperature, max_tokens):
    print("--initialising llama2")
    global llm
    llm = VLLM(
            model = "TheBloke/Llama-2-70B-Chat-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_mixtral(temperature, max_tokens):
    print("--initialising Mixtral")
    global llm
    # llm = VLLM(
    #         model = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    #         trust_remote_code=True,  
    #         dtype="float16",
    #         max_new_tokens=max_tokens,
    #         top_k=-1,
    #         top_p=0.95,
    #         temperature=temperature,
    #         repetition_penalty=1.1,
    #         return_full_text=False
    #     )
    # llm = OllamaLLM(model="mixtral:8x7b-text-v0.1-fp16")
    inference_server_url = "http://localhost:8000/v1"
    llm = ChatOpenAI(
        model="mistralai/Mixtral-8X7B-Instruct-v0.1",
        openai_api_key="token-abc123",
        openai_api_base=inference_server_url,
        max_tokens=256,
        temperature=0,
    )

def initialise_mistral(temperature, max_tokens):
    print("--initialising Mixtral")
    global llm
    # llm = OllamaLLM(model="mistral:7b-instruct")
    inference_server_url = "http://0.0.0.0:8000/v1"
    llm = ChatOpenAI(
    model = "mistralai/Mistral-7B-Instruct-v0.2",
    openai_api_key="token-abc123",
    openai_api_base=inference_server_url,
    max_tokens=256,
    temperature=0,
)


def initialise_gpt(temperature, max_tokens, opeani_api_keys, model_name):
    print("--initialising gpt")
    global llm
    llm = ChatOpenAI(
    model=model_name,
    api_key=opeani_api_keys,
    temperature=temperature,
    max_tokens=max_tokens,
    request_timeout=30,
    max_retries=3,
    timeout=60 * 3,
)


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llama3(prompt_input):

    template="""You are an AI assistant that helps people find information.
    {prompt_input}
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_llama2(prompt_input):

    template="""[INST] <<SYS>> You are an AI assistant that helps people find information.<</SYS>>
    {prompt_input}[/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_mixtral(prompt_input):

    template="""<s>[INST] You are an AI assistant that helps people find information.
    {prompt_input} [/INST]
    """
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm
    # print(prompt.format(prompt_input=prompt_input))
    result = llm_chain.invoke({"prompt_input": prompt_input})
    
    return result.content

def run_mistral(prompt_input):

    template="""<s>[INST] You are an AI assistant that helps people find information.
    {prompt_input} [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    # print(prompt_input)
    result = llm_chain.invoke(prompt_input)
    # print("$$$ Mistral LLM output: ", result.content)

    return result.content

def run_gpt(prompt_input):

    template="""You are an AI assistant that helps people find information.
    {prompt_input}
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    result = llm_chain.invoke({"prompt_input": prompt_input})
    # print("$$ LLM OUTPUT $$: ", result.content)
    return result.content

def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        # print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, ref_answers, answer, cluster_chain_of_entities, file_name, ans_from, qid, question_type, prompt_template, intermediate_answer, output_file):
  
    current_answer = {
        "id": qid,
        "question": question,
        "gold_answer": ref_answers,
        "generated_answer": answer,
        "reasoning_chains": cluster_chain_of_entities,
        "ans_from": ans_from , 
        "prompt_template": prompt_template,
        "intermediate_answer": intermediate_answer
    }
    print("CURRENT ANSWER: ", current_answer)
    with open(output_file, "a") as f:
        json.dump(current_answer, f)
        f.write('\n')


    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args, question_type):
   
    prompt = "generate_directly"+ "\n\nQuestion: " + question + "\nAnswer:"
    prompt_template = "generate_directly"

    if (args.LLM_type).startswith("gpt"):
        response = run_gpt(prompt)
    elif args.LLM_type == "llama3":
        response = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        response = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        response = run_mixtral(prompt)
    elif args.LLM_type == "mistral":
        response = run_mistral(prompt)
    # print("!! FINAL LLM Answer\n", response)
    return prompt_template, response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst

def classify_single_question(references):
    """Function to classify a single question type into 'yes/no', 'yes/no_conditional', 'span', or 'span_conditional'."""
    
    if not references:
        return "unanswerable"
    
    # Check for yes/no answers
    if any(ans[0] in ["yes", "no"] for ans in references):
        if any(ans[1] for ans in references):
            # print("Yes/no - conditions")
            return "Yes/no - conditions"
        else:
            # print("Yes/no")
            return "Yes/no"
    
    # Check for span answers
    else:
        if any(ans[1] for ans in references):
            # print("span - conditions")
            return "span - conditions"
        else:
            # print("span")
            return "span"


def prepare_dataset_condqa(dataset_name):
    if dataset_name == 'conditionalqa':
        with open('../data/conditional_qa_dev_entities_v3_gpt3.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'context_question'
    
    else:
        print("dataset not found, you should pick from available dataset: conditionalqa.")
        exit(-1)
    return datas, question_string

def prepare_dataset_hotpotqa(dataset_name):
    if dataset_name == 'hotpotqa500':
        with open('../data/stratified_hotpotqa_500sample_with_tag.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    
    else:
        print("dataset not found, available dataset: hotpotqa500.")
        exit(-1)
    return datas, question_string

def classify_hotpotqa_question(self, gold_ans):
        if gold_ans in ["yes","no"]:
            return "yes"
        else:
            return "span" 
        
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))