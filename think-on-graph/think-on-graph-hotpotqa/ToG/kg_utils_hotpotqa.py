from utils_hotpotqa import *
from prompt_list_hotpotqa import *
# import json
import time
# import openai
import re
# from neo4j import GraphDatabase
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.error

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def execute_entity_search_query(cypher_query, entity_id, relation_name):
    result = session.run(cypher_query, parameters={"entity_id":entity_id, "relation_name": relation_name})
    return [record["entity_found"] for record in result]

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    
def clean_relations(string, entity_id, head_relations):
    # pattern = r"(?P<relation>[A-Z_]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)"
    pattern = r"\{\s*(?P<relation>.+?)\s+\(Score:\s+(?P<score>[0-9.]+)\)\}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
        
    if not relations:
        return False, []
    return True, relations

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

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_wiki % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        
def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def get_qid(entity_name, max_retries=3, base_delay=2):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item rdfs:label "{entity_name}"@en.
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            if results["results"]["bindings"]:
              return results["results"]["bindings"][0]["item"]["value"].split("/")[-1]
            else:
                return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(e)
                continue
    # raise Exception("Too many retries — still getting rate limited. - get qid")
    print(f"Max retries reached for entity '{entity_name}'. Moving on.")
    return None
    


def get_head_relations(qid, entity, max_retries=3, base_delay=2):

    query = f"""
    SELECT ?relationLabel
    WHERE {{
    wd:{qid} ?prop ?x .
    ?relation wikibase:directClaim ?prop .

    SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
    }}
    }}
    LIMIT 100
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    for attempt in range(max_retries):
        try:
            response = sparql.query()
            try:
                results = response.convert()
            except json.JSONDecodeError as e:
                wait = base_delay * (2 ** attempt)
                print(f"JSON decode error: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                continue
            ent_relation = {}
            for result in results["results"]["bindings"]:
                relation = result["relationLabel"]["value"]
                if entity in ent_relation.keys():
                    ent_relation[entity].append(relation)
                else: ent_relation[entity] = [relation] 
                
            return ent_relation
        except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"Rate limited (429). Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(e)
                    continue
    print("Too many retries — still getting rate limited. - get head relations")
    return {}

    

def get_tail_relations(qid, entity, max_retries=3, base_delay=2):

    query = f"""
    SELECT ?relationLabel
    WHERE {{
    ?x ?prop wd:{qid} .
    ?relation wikibase:directClaim ?prop .

    SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
    }}
    }}
    LIMIT 100
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    for attempt in range(max_retries):
        try:
            response = sparql.query()
            try:
                results = response.convert()
            except json.JSONDecodeError as e:
                wait = base_delay * (2 ** attempt)
                print(f"JSON decode error: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                continue

            ent_relation = {}
            for result in results["results"]["bindings"]:
                relation = result["relationLabel"]["value"]
                if entity in ent_relation.keys():
                    ent_relation[entity].append(relation)
                else: ent_relation[entity] = [relation] 
                
            return ent_relation
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(e)
                continue
    print("Too many retries — still getting rate limited. - get tail relations")
    return {}

 
def relation_search_prune(entity_id, pre_relations, pre_head, question, args):
    qid = get_qid(entity_id)
    if qid:
        head_entity_relations = get_head_relations(qid, entity_id)
        tail_entity_relations = get_tail_relations(qid, entity_id)
    else:
        head_entity_relations = {}
        tail_entity_relations = {}

   
    total_entity_relations = head_entity_relations | tail_entity_relations
    flag = False
    retrieve_relations_with_scores = []
    for entity_name, relations in total_entity_relations.items():
        total_relations = relations
        if len(total_relations) > args.width: 
            if args.prune_tools == "llm":
                print("--Pruning relation output--")
                prompt = construct_relation_prune_prompt(question, entity_name, total_relations[:100], args)
                if (args.LLM_type).startswith("gpt"):
                    result = run_gpt(prompt)
                elif args.LLM_type == "llama3":
                    result = run_llama3(prompt)
                elif args.LLM_type == "llama2":
                    result = run_llama2(prompt)
                elif args.LLM_type == "mixtral":
                    result = run_mixtral(prompt)
                elif args.LLM_type == "mistral":
                    result = run_mistral(prompt)
                flag, relations_with_scores = clean_relations(result, entity_name, total_relations) 
                retrieve_relations_with_scores.append(relations_with_scores)
   
    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length

def get_property_id(relation_label, max_retries=3, base_delay=2):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?property
    WHERE {{
        ?property rdfs:label "{relation_label}"@en .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["property"]["value"].split("/")[-1]  
            else:
                return None  
        except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"Rate limited (429). Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(e)
                    continue
    print("Too many retries — still getting rate limited. - get property id")
    return None


def get_tail_entities(qid, relation, max_retries=3, base_delay=2):
    relation_id = get_property_id(relation)

    if relation_id is None:
        print(f"No property found for relation label: {relation}")
        return []
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT  ?tailEntity ?tailEntityLabel
    WHERE {{
        wd:{qid} wdt:{relation_id} ?tailEntity .
        
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
            ?tailEntity rdfs:label ?tailEntityLabel .
        }}
        FILTER(BOUND(?tailEntityLabel))
    }}
    LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            tail_ent = []
            for result in results.get("results", {}).get("bindings", []):
                label_data = result.get("tailEntityLabel")
                if label_data and label_data.get("type") == "literal" and label_data.get("xml:lang")=="en":
                    tail_ent.append(label_data["value"])
            
            return tail_ent
        except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"Rate limited (429). Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(e)
                    continue
    print("Too many retries — still getting rate limited. - get tail entities")
    return []


def get_head_entities(qid, relation, max_retries=3, base_delay=2):

    relation_id = get_property_id(relation)
    
    if relation_id is None:
        print(f"No property found for relation label: {relation}")
        return []
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT  ?headEntity ?headEntityLabel
    WHERE {{
        ?headEntity wdt:{relation_id} wd:{qid} .
        
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
            ?headEntity rdfs:label ?headEntityLabel .
        }}
        FILTER(BOUND(?headEntityLabel))
    }}
    LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            head_ent = []
            for result in results.get("results", {}).get("bindings", []):
                label_data = result.get("headEntityLabel")
                if label_data and label_data.get("type") == "literal" and label_data.get("xml:lang")=="en":
                    head_ent.append(label_data["value"])
                
            return head_ent
        except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"Rate limited (429). Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(e)
                    continue
    print("Too many retries — still getting rate limited. - get head entities")
    return []



def entity_search(entity, relation, head=True):
    qid = get_qid(entity)
    if qid:
        if head:
            entities = get_tail_entities(qid, relation)
        else:
            entities = get_head_entities(qid, relation)
    else:
        print("qid not found - entity search")

    return entities

def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = entity_candidates_id
    
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        print("!! pruning triggered inside entity score")
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        if (args.LLM_type).startswith("gpt"):
            result = run_gpt(prompt)
        elif args.LLM_type == "llama3":
            result = run_llama3(prompt)
        elif args.LLM_type == "llama2":
            result = run_llama2(prompt)
        elif args.LLM_type == "mixtral":
            result = run_mixtral(prompt)
        elif args.LLM_type == "mistral":
            result = run_mistral(prompt)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id
   
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head

def half_stop(question, ref_answers, cluster_chain_of_entities, depth, args, qid, question_type, intermediate_answer):
    prompt_template, answer = generate_answer(question, cluster_chain_of_entities, args, question_type)
    save_2_jsonl(question, ref_answers, answer, cluster_chain_of_entities, file_name=args.dataset, ans_from="LLM+KG-half_stop", qid=qid, question_type=question_type, prompt_template=prompt_template, intermediate_answer=intermediate_answer, output_file=args.output_file)

def generate_answer(question, cluster_chain_of_entities, args, question_type): 
   
    prompt = answer_prompt_wiki + "\n"+ question +'\n'
    prompt_template = "answer_prompt_wiki"

    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'Answer: '

    if (args.LLM_type).startswith("gpt"):
        result = run_gpt(prompt)
    elif args.LLM_type == "llama3":
        result = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        result = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        result = run_mixtral(prompt)
    elif args.LLM_type == "mistral":
        result = run_mistral(prompt)
    return prompt_template, result

def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads

def reasoning(question, cluster_chain_of_entities, args):
    print("--Running LLM Reasoning--")
    prompt = prompt_evaluate_wiki + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
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
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    



