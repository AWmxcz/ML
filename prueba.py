def minion_game(string):
    vowels = ['A', 'E', 'I', 'O', 'U']
    stuart_words = []
    kevin_words = []

    # Iterar sobre todas las subcadenas posibles
    for i in range(len(string)):
        for j in range(i, len(string)):
            sub = string[i:j+1]
            print(sub)

            if sub[0] in vowels:
                kevin_words.append(sub)
            else:
                stuart_words.append(string.count(sub))

    stuart_score = sum(string.count(word) for word in stuart_words)

    
    kevin_score = sum(string.count(word) for word in kevin_words)

    
    if stuart_score > kevin_score:
        print("Stuart", stuart_score)
    elif kevin_score > stuart_score:
        print("Kevin", kevin_score)
    else:
        print("Draw")
minion_game('Bannanas')