import json
import openai
import numpy as np
import time
import re

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):                         #string에 가장 마지막에 등장하는 숫자 parsing
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):                                #{}로 둘러쌓인 숫자 parsing
    pattern = r"\{([^}]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    if solution == "":
        return None
    else:
        return solution

def parse_answer_gt(input_str):
    solution = input_str.split('####')[1].strip()
    solution = solution.replace(',', '')
    return solution


def compute_accuracy(gt, pred_solutions, num_agent, num_round,MC):
    answers = parse_answer_gt(gt)

    if answers is None:
        return None

    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            pred_answers.append(pred_answer)
        if MC == 1:
            pred_answer = mc(pred_answers,num_agent, num_round)
        else:
            pred_answer = ma(pred_answers,num_agent, num_round)

    if pred_answer is None:
        return 0
    # try:
    if float(answers) == float(pred_answer):
        return 1
    else:
        return 0
    # except:
    #     import pdb
    #     pdb.set_trace()
    #     print(pred_solution)


def ma(answers,num_agent, num_round):
    last_round_response = []
    for i in range(len(answers)):
        if i % num_round == (num_round-1):
            last_round_response.append(answers[i])

    last_round_response_clean = [x for x in last_round_response if x is not None]

    num = last_round_response_clean[0] 
    cnt = last_round_response_clean.count(num) 
    for i in last_round_response_clean: 
        if last_round_response_clean.count(i)>cnt : 
            num = i 
            cnt = last_round_response_clean.count(i)
    return num

def mc(List,num_agent, num_round):
    counter = 0
    num = List[0]

    same_list =[]
    last_round_response = []
    same = 0 

    for i in range(len(List)):
        if i % num_round == (num_round-1):
            last_round_response.append(List[i])

    cleaned_list = [x for x in List if x is not None]
    cleaned_last_round_response = [x for x in last_round_response if x is not None]
        
    for i in cleaned_list:
        current_frequency = cleaned_list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    
    same_list.append(num)
    
    for i in cleaned_list: 
        current_frequency = cleaned_list.count(i)
        if current_frequency == counter and i not in same_list:
            same_list.append(i)
            same = 1

    counter =0 
    if same == 1:
        for i in same_list: 
            current_frequency = cleaned_last_round_response.count(i)
            if current_frequency > counter:
                counter = current_frequency
                num = i

    return num


if __name__ == "__main__":
    for i in range(1,11):
        file_name = "output/gsm/turbo/gsm_agent_12_round_1_test_50_turbo_{}.json".format(i)

        MC = 1

        parts = file_name.split('_')
        num_agent = int(parts[2])  
        num_round = int(parts[4])

        response_dict = json.load(open(file_name, "r"))
        questions = list(response_dict.keys())

        accuracies = []

       
        for question in questions:
            responses, gt = response_dict[question]
            pred_solutions = []

            for response in responses:
                '''
                if MC == 0:
                    pred_solution = response[-1]['content']              #에이전트가 가장 마지막 라운드에 내놓은 대답 
                    pred_solutions.append(pred_solution)
                else:
                '''
                for i in range(len(response)):
                    if response[i]['role'] == 'assistant':
                        pred_solution = response[i]['content']
                        pred_solutions.append(pred_solution)
                    else:
                        continue
                
                
            accurate = compute_accuracy(gt, pred_solutions ,num_agent, num_round,MC)
            
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                import pdb
                pdb.set_trace()
                print(gt)


        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))

