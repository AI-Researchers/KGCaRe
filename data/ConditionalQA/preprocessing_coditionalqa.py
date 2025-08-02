from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import json
import sys


llm = None

def initialise_model(openai_api_key):
    global llm
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
        temperature=0.0,
        request_timeout=30,
        max_retries=3,
        timeout=60 * 3,
    )


def extract_entities(question):
    
    template = """Please extract keywords from the following text. \
    Include both explicitly mentioned keywords and any implied keywords that are relevant to the context. \
    Ensure the output only contains the keywords, separated by commas, and nothing else. \n
    **Question:**
    {question}\n
    **Keywords:**
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"question": question})
    return output_all.content


def preprocess_conditionalqa(input_condqa_dev_file, new_condqa_dev_with_entities_file, openai_api_key):
    with open(input_condqa_dev_file,encoding='utf-8') as f:
        datas = json.load(f)
    
    initialise_model(openai_api_key)

    condqa_entities = []
    for j, data in enumerate(datas):
        # print("\n************ for question no: ", j+1)
        data['context_question'] = data['scenario'] + ' ' + data['question']
        context_q = data['context_question']
        # print(context_q)
        topic_entities = extract_entities(context_q)
        try:
            topic_entities_list = topic_entities.split('\n')[0]
        except:
            topic_entities_list = topic_entities
        topic_entities_list = topic_entities_list.split(',')
        topic_entity_cleaned = [word.strip() for word in topic_entities_list]
        data['topic_entity'] = topic_entity_cleaned
        # print(topic_entity_cleaned)
        condqa_entities.append(data)

    with open(new_condqa_dev_with_entities_file, 'w') as file:
        json.dump(condqa_entities, file, indent=4)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python preprocessing_conditionalqa.py <dev.json> <dev_with_entities_gpt3.json> <openai_api_key>")
        sys.exit(1)
    preprocess_conditionalqa(sys.argv[1], sys.argv[2], sys.argv[3])
