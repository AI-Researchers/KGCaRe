from tqdm import tqdm
import argparse
from utils_hotpotqa import *
from kg_utils_hotpotqa import *
import random
import time


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="hotpotqa500", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.1, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.1, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--openai_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--output_file", type=str,
                        default="./output/ToG_output.json", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()
    if args.LLM_type == "llama3":
        initialise_llama3(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "mixtral":
        initialise_mixtral(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "llama2":
        initialise_llama2(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "mistral":
        initialise_mistral(args.temperature_reasoning, args.max_length)
    elif args.LLM_type.startswith("gpt"):
        initialise_gpt(args.temperature_reasoning, args.max_length, args.openai_api_keys, args.LLM_type)

    print("output_file = ", args.output_file)

    datas, question_string = prepare_dataset_hotpotqa(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    
    for j, data in tqdm(enumerate(datas)):
        if j+1 >= 0:
            
            print("\n\n************ for question no: ", j+1)
            qid = data['_id']
            # print(qid)
            question = data[question_string]
            # print('Q: ', question)
            topic_entity = data['entities']
            # print('topic entity: ', topic_entity)
            question_type = []
            question_type = data['qtype']
            # print("qtype = ",question_type)
            ref_answers = data['answer']
            
            cluster_chain_of_entities = []
            intermediate_answer = {}
            inter_answer_current_entity_list = []
            total_candidates_list = []
            
            pre_relations = []
            pre_heads= [-1] * len(topic_entity)
            flag_printed = False
            pre_heads_exists = False
            for depth in range(1, args.depth+1):
                print("\nFOR DEPTH: ", depth)
                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    entity = remove_punctuation(entity)
                    print("\n--for Entity: ", entity)
                    if entity!="[FINISH_ID]": 
                        retrieve_relations_with_scores = relation_search_prune(entity, pre_relations, pre_heads_exists, question, args)  
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                        
                    i+=1
                total_candidates = []
                total_scores = []
                total_relations = []
                total_entities_id = []
                total_topic_entities = []
                total_head = []
                inter_answer_current_entity_list.append(current_entity_relations_list)
                for entity_list in current_entity_relations_list:
                    for entity in entity_list:
                        try:
                            if entity['head']:
                                entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                            else:
                                entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                            if args.prune_tools == "llm":
                                if len(entity_candidates_id) >=20:
                                    entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                            if len(entity_candidates_id) ==0:
                                continue
                            scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                            
                            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                        except Exception as e:
                            print("ERROR occured in current entity relations list!!")
                            print(e)
                            continue
                total_candidates_list.append(total_candidates)
                current_depth_candidates = 'Depth_'+ str(depth) + '_Candidates'
                intermediate_answer[current_depth_candidates] = total_candidates_list
                if len(total_candidates) ==0:
                    half_stop(question, ref_answers, cluster_chain_of_entities, depth, args, qid, question_type, intermediate_answer)
                    flag_printed = True
                    break
                    
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                if len(chain_of_entities)>0:
                    chain_of_entities = list(set(chain_of_entities[0]))
                
                cluster_chain_of_entities.append(chain_of_entities)
                current_depth_chain = 'Depth_'+ str(depth) + '_chain_of_entities'
                intermediate_answer[current_depth_chain] = chain_of_entities
                if flag:
                    stop, results = reasoning(question, cluster_chain_of_entities, args)
                    print("\nRESULT OF REASONING:: ", stop, results)
                    if stop:
                        print("ToG stoped at depth %d." % depth)
                        half_stop(question, ref_answers, cluster_chain_of_entities, depth, args, qid, question_type, intermediate_answer)
                        flag_printed = True
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                        entities_id = list(set(entities_id))
                        topic_entity = entities_id
                        continue
                else:
                    half_stop(question, ref_answers, cluster_chain_of_entities, depth, args, qid, question_type, intermediate_answer)
                    flag_printed = True
        
            if not flag_printed:
                prompt_template, results = generate_without_explored_paths(question, args, question_type)
                save_2_jsonl(question, ref_answers, results, [], file_name=args.dataset, ans_from="LLM", qid=qid, question_type=question_type, prompt_template="last_", intermediate_answer=intermediate_answer, output_file=args.output_file)
                
              
        else: continue
    
    end_time = time.time()
    exc_time_in_min = round((end_time - start_time) / float(60), 4)
    print('Program executed in %s mins', exc_time_in_min)
