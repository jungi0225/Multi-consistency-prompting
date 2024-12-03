import openai
import json
import numpy as np
import random
import os

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
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
    questions = read_jsonl("data/gsm8k/gsm8k_data.jsonl")  
    for seed_num in range(11,21):
        random.seed(seed_num)
        random.shuffle(questions)
        generated_description = {}
        for agents in range(5,6):
            for rounds in range(3,4): 
                ques_num = 1
                for data in questions[:num_tests]:  
                    
                    question = data['question']
                    answer = data['answer']

                    agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]


                    for round in range(rounds):
                        for i, agent_context in enumerate(agent_contexts):

                            if round != 0:
                                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                                message = construct_message(agent_contexts_other, question, 2*round - 1)
                                agent_context.append(message)

                            completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=agent_context,
                                    n=1)


                            assistant_message = construct_assistant_message(completion)
                            agent_context.append(assistant_message)

                        print("round {} Done!".format(round))
                    
                    
                    generated_description[question] = (agent_contexts, answer)
                    
                    print("question {} of Test set {} Complete! \n".format(ques_num,seed_num-10))
                    ques_num += 1

                json.dump(generated_description, open("output/gsm/turbo/gsm_agent_{}_round_{}_test_{}_turbo_{}.json".format(agents, rounds,num_tests,seed_num-10), "w"))


