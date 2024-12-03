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



def parse_answer_csqa(input_str,labels,texts):
    # 정규 표현식을 사용하여 중괄호 {}로 둘러싸인 문자열을 추출하는 패턴 정의
    pattern = r"\{([^}]*)\}"
    matches = re.findall(pattern, input_str)

    # matches 리스트에서 마지막으로 유효한 A-E만 남김
    for match_str in reversed(matches):
        # A부터 E 사이의 문자만 남기고 나머지 문자 제거
        cleaned_str = re.sub(r"[^A-E]", "", match_str)
        if cleaned_str:
            return cleaned_str
        else: 
            no_space_str = match_str.replace(" ", "")
            for idx in range(5):
                if no_space_str == texts[idx]:
                    return labels[idx]

    return None  # 유효한 A-E 문자열을 찾지 못한 경우 None 반환

def compute_accuracy_csqa(gt, pred_solutions, num_agent, num_round,labels,texts):
    answers = gt

    if answers is None:
        return None

    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer_csqa(pred_solution,labels,texts)
            pred_answers.append(pred_answer)

        pred_answer = most_frequent(pred_answers,num_agent, num_round)

    if answers == pred_answer :
        return 1
    else:
        return 0


def most_frequent(List,num_agent, num_round):
    counter = 0
    num = List[0]

    same_list =[]
    last_round_response = []
    same = 0 

    for i in range(len(List)):
        if i % num_round == (num_round-1):
            last_round_response.append(List[i])
        
    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    
    same_list.append(num)
    
    for i in List: 
        current_frequency = List.count(i)
        if current_frequency == counter and i not in same_list:
            same_list.append(i)
            same = 1

    counter =0 
    if same == 1:
        for i in same_list: 
            current_frequency = last_round_response.count(i)
            if current_frequency > counter:
                counter = current_frequency
                num = i

    return num

if __name__ == "__main__":
    
    for i in range(1,11):
        file_name = "output/csqa/turbo/csqa_agent_1_round_1_test_50_turbo_{}.json".format(i)

        MC = 0

        parts = file_name.split('_')
        num_agent = int(parts[2])  
        num_round = int(parts[4])

        response_dict = json.load(open(file_name, "r"))
        questions = list(response_dict.keys())

        accuracies = []

       
        for question in questions:
            responses, labels, texts ,gt = response_dict[question]
            pred_solutions = []

            for response in responses:
                if MC == 0:
                    pred_solution = response[-1]['content']              #에이전트가 가장 마지막 라운드에 내놓은 대답 
                    pred_solutions.append(pred_solution)

                else:
                    for i in range(len(response)):
                        if response[i]['role'] == 'assistant':
                            pred_solution = response[i]['content']
                            pred_solutions.append(pred_solution)
                        else:
                            continue
                
            accurate = compute_accuracy_csqa(gt, pred_solutions ,num_agent, num_round,labels,texts)
            
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                import pdb
                pdb.set_trace()
                print(gt)


        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))

