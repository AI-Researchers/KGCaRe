import json
import random

# # Load the JSON dataset
# with open('/home/simsam/conditionalqa_rag_pipeline/data/train_FS_sampled_triples.json', 'r') as f:
#     data = json.load(f)

# Function to filter questions by type and randomly select n examples
def select_random_few_shots(data,question_type, n=5):
    selected_data = [item for item in data if item['questionType'] == question_type]
    
    # Randomly select n examples (or less if not enough data)
    selected_examples = random.sample(selected_data, min(len(selected_data), n))
    
    # Prepare structured output
    formatted_examples = ""
    for example in selected_examples:
        # Formatting each example in the desired structure
        evidence = ', '.join(example['evidences'])
        if question_type in ['span_conditional', 'yes/no_conditional']:
            conditions = "\n".join(example['answers'][0][1])
            answer = example['answers'][0][0] + "\n Conditions: " + conditions
        else:    
            answer = example['answers'][0][0]
        formatted_example = (
            # f"Context information: {evidence} \n"
            # f"{example['knowledge_triples']} \n"
            f"Question: {example['scenario']} {example['question']} \n"
            f"Answer: {answer} \n"
            f"---------------------\n"
        )
        formatted_examples += formatted_example
    
    return formatted_examples

# Example usage: Select 5 random examples from the 'yes/no' questionType
# question_type = 'span_conditional'  # You can change this to any other questionType ('yes/no_conditional', 'span', etc.)
# n = 5  # Number of examples to select
# formatted_output = select_and_format(question_type, n)

# # Print the formatted output
# print(formatted_output)
