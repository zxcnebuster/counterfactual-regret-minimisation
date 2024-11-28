import itertools

def calculate_game_tree_size(game_def):
    num_players = game_def['numPlayers']
    num_rounds = game_def['numRounds']
    num_suits = game_def['numSuits']
    num_ranks = game_def['numRanks']
    num_hole_cards = game_def['numHoleCards']
    num_board_cards = game_def['numBoardCards']
    max_raises = game_def['maxRaises']
    raise_size = game_def['raiseSize']
    first_player = game_def['firstPlayer']
    stack = game_def['stack']

    # Calculate the number of hole card combinations
    num_hole_card_combinations = num_suits * num_ranks

    # Calculate the number of board card combinations per round
    def board_card_combinations(num_board_cards, num_suits, num_ranks):
        if num_board_cards == 0:
            return 1
        return len(list(itertools.combinations(range(num_suits * num_ranks), num_board_cards)))

    total_board_combinations = 1
    for num in num_board_cards:
        total_board_combinations *= board_card_combinations(num, num_suits, num_ranks)

    # Calculate the number of possible actions per round
    def possible_actions(max_raises):
        return 3 + 2 * max_raises  # Call, fold, and raise (up to max_raises times)

    total_possible_actions = 1
    for i in range(num_rounds):
        total_possible_actions *= possible_actions(max_raises[i])

    # Calculate the total game tree size
    game_tree_size = (num_hole_card_combinations ** num_players) * total_board_combinations * total_possible_actions
    
    return game_tree_size

game_def = {
    'numPlayers': 2,
    'numRounds': 3,
    'numSuits': 4,
    'numRanks': 13,
    'numHoleCards': 1,
    'numBoardCards': [0, 1, 1],
    'maxRaises': [3, 3, 3],
    'raiseSize': [10, 20, 20],
    'firstPlayer': [1, 2, 1],
    'stack': 100
}
game_tree_size = calculate_game_tree_size(game_def)
print("Size of the game tree:", game_tree_size)
