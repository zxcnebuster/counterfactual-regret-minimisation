import itertools

def calculate_information_sets(game_def):
    num_players = game_def['numPlayers']
    num_rounds = game_def['numRounds']
    num_suits = game_def['numSuits']
    num_ranks = game_def['numRanks']
    num_hole_cards = game_def['numHoleCards']
    num_board_cards = game_def['numBoardCards']
    max_raises = game_def['maxRaises']
    raise_size = game_def['raiseSize']
    
    num_hole_card_combinations = num_suits * num_ranks

    board_card_combinations = []
    for i in range(num_rounds):
        board_card_combinations.append(itertools.combinations(range(num_suits * num_ranks), num_board_cards[i]))
    
    def betting_sequences(max_raises, num_players):
        sequences = []
        for raises in range(max_raises + 1):
            sequences.extend(itertools.product(range(num_players), repeat=raises))
        return sequences
    
    betting_sequences_per_round = [betting_sequences(max_raises[i], num_players) for i in range(num_rounds)]
    
    total_information_sets = 0
    for round_index in range(num_rounds):
        for board in board_card_combinations[round_index]:
            for sequence in betting_sequences_per_round[round_index]:
                total_information_sets += num_hole_card_combinations ** num_players
    
    return total_information_sets

game_def = {
    'numPlayers': 2, 
    'numRounds': 3,
    'numSuits': 4,
    'numRanks': 3,
    'numHoleCards': 1,
    'numBoardCards': [0, 1, 1],
    'maxRaises': [1, 1,1],
    'raiseSize': [2, 4],
    'stack': 100
}

num_info_sets = calculate_information_sets(game_def)
print("Number of information sets:", num_info_sets)
