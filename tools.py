stopword = [line.split()[0] for line in open("stopword.txt")]

def GetFar(question, answers):
    
    question = set(question)
    sametokens = [sum(1 for token in set(answer) if token in question) for answer in answers]

    return sametokens.index(min(sametokens)), sametokens


if __name__ == "__main__":
    
    question = [ '那', '要去', '上班', '的', '時候', '就算', '了', '一下', '好像', '沒有', '辦法', '藉由', '這', '個', '方式', '，', '一般', '的', '職業', '去', '償還', '這', '債務', '所以', '我們', '就', '嘗試', '就', '有', '藉由', '創業', '這', '個', '方向', '然後', '去接', '案子', '然', '，', '用', '更多', '時間', '去', '換取', '就是', '償還', '這', '債務', '的', '機會', '這樣子']
    
    answers = [
	['拜訪', '一', '個', '家庭'],
	['一般', '的', '職業'],
	['這', '個', '方向'],
	['債務', '的', '機會']
    ]
    
    answer_index, scores = GetFar(question, answers)
    print(answer_index)
    print(scores)
    
    