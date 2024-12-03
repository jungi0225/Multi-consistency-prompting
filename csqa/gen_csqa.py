import openai
import json
import numpy as np
import random
import os

def construct_message(agents, question,label,text, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, choose a label among following labels and texts. labels:{}, texts:{}. Don't forget that your final answer should be a single label (A-E), in the form \\boxed{{label}}.".format(label,text)}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the problem? \n The original problem is {}. You should choose one label among following labels and texts. labels:{}, texts:{}. Dont forget that your final answer should be a single label (A-E), in the form \\boxed{{label}}, at the end of your response.""".format(question,label, text)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):  
    with open(path) as fh:  
        return [json.loads(line) for line in fh.readlines() if line]  

if __name__ == "__main__":   
    #agents = 3     
    #rounds = 3     
    num_tests = 50
    
    openai.api_key = os.environ.get("OPENAI_API_KEY")  
    questions = read_jsonl("data/csqa/csqa_data.jsonl")  
    idx = 0
    for seed_num in range(1,11):
        idx += 1
        random.seed(seed_num)
        random.shuffle(questions)
        generated_description = {}
        for agents in range(1,2):
            for rounds in range(1,2): 
                ques_num = 1
                for data in questions[:num_tests]:  
                    
                    question = data['question']
                    answer = data['answerKey']
                    choice = data['choices']
                    label = choice['label']
                    text = choice['text']

                    agent_contexts = [[{"role": "user", "content": """Can you solve the following problem? {} Explain your reasoning. You should choose a single label among following labels and texts. labels:{}, texts:{}. Don't forgent that your final answer should be a single label (A-E), in the form \\boxed{{label}}, at the end of your response. """.format(question,label, text)}] for agent in range(agents)]


                    for round in range(rounds):
                        for i, agent_context in enumerate(agent_contexts):

                            if round != 0:
                                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                                message = construct_message(agent_contexts_other, question,label,text,2*round - 1)
                                agent_context.append(message)

                            completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=agent_context,
                                    n=1)


                            assistant_message = construct_assistant_message(completion)
                            agent_context.append(assistant_message)

                        print("round {} Done!".format(round))
                    
                    
                    generated_description[question] = (agent_contexts,label,text, answer)
                    
                    print("question {} of Test set {} Complete! \n".format(ques_num,idx))
                    ques_num += 1

                json.dump(generated_description, open("output/csqa/turbo/csqa_agent_{}_round_{}_test_{}_turbo_{}.json".format(agents, rounds,num_tests,idx), "w"))


