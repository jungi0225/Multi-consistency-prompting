import openai
import json
import numpy as np
import random
import os
import re



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

def parse_answer_gt(input_str):
    solution = input_str.split('####')[1].strip()
    solution = solution.replace(',', '')
    return solution

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


for i in range(1,11):

    agent_num = 3
    round_num = 2
    test_set = i

    method = "gsm"
    if method == "csqa":
        csqa = 1
    else:
        csqa = 0

    # JSON 파일 로드
    file_name = "output/{}/turbo/{}_agent_{}_round_{}_test_50_turbo_{}.json".format(method, method, agent_num, round_num,test_set)
    with open(file_name, 'r') as file:
        data = json.load(file)


    # 결과를 저장할 딕셔너리 초기화
    result = {}

    # 데이터셋의 각 질문 처리
    for idx, (question, responses) in enumerate(data.items()):
        # 실제 정답과 라운드별 응답 추출
        if csqa:
            actual_answer = responses[-1]
            response = responses[0]
            labels = responses[1]
            texts = responses[2]
        else:
            actual_answer = parse_answer_gt(responses[-1])

        


        
        round_responses = [[] for _ in range(agent_num)]

        response_list = []

        for agent in range(agent_num):
            answer_list = []
            response = responses[0][agent]
            for round in range(len(response)): 

                if response[round]['role'] == 'assistant': 
                    pred_solution = response[round]['content'] 
                    if csqa:
                        pred_answer = parse_answer_csqa(pred_solution,labels,texts)
                    else:
                        pred_answer = parse_answer(pred_solution)   
                    
                    answer_list.append(pred_answer)
                    response_list.append(pred_answer)
                else:
                    continue
            round_responses[agent].append(answer_list)
        
        mc_answer = mc(response_list,agent_num,round_num)
        ma_answer = ma(response_list,agent_num,round_num)
        

        # 현재 질문에 대한 엔트리 구성
        result[idx] = [actual_answer, round_responses,mc_answer,ma_answer]

    # 결과를 JSON 형식으로 저장
    output_file_name = "transformed_data_{}_{}_{}.json".format(agent_num, round_num, test_set)
    with open(output_file_name, 'w') as output_file:
        json.dump(result, output_file, indent=4, ensure_ascii=False)

    print(f"변환된 데이터가 {output_file_name}에 저장되었습니다.")
    for j in range(50):
        if (result[j][2] != result[j][3]):
            print(j, result[j])

