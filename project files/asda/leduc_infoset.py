from math import factorial

def calculate_information_sets(gamedef):
    """Calculates info sets for a generalized poker game."""

    num_information_sets = 0

    def recursive_calculate(round_num, history, pot_size, raise_count, cards_dealt):
        """Recursively calculates info sets."""
        nonlocal num_information_sets
        if round_num == gamedef['numRounds']:
            return 1  

        num_actions = 1  
        if raise_count < gamedef['maxRaises'][round_num]:
            num_actions += 1 

        for action_num in range(num_actions):
            new_history = history
            new_raise_count = raise_count
            new_pot = pot_size
            if action_num == 0:
                new_history += 'c'
                new_pot += 1  
            else:
                new_history += 'r'
                new_raise_count += 1
                new_pot += 2  

            if (len(new_history) % 2) == 0:
                num_information_sets += recursive_calculate(
                    round_num + 1, '', 0, 0, cards_dealt + gamedef['numBoardCards'][round_num]
                )
            else:
                num_information_sets += recursive_calculate(
                    round_num, new_history, new_pot, new_raise_count, cards_dealt
                )

        return num_information_sets

    num_information_sets = recursive_calculate(0, '', 0, 0, 0)

    num_cards = gamedef['numSuits'] * gamedef['numRanks']

    # Calculate hand combinations (for 2 players)
    hand_combinations = (num_cards * (num_cards - 1)) // 2  

    # Calculate board combinations
    board_combinations = 1
    cards_left = num_cards - 2  # Cards remaining after dealing hole cards
    for num_board in gamedef['numBoardCards']:
        board_combinations *= factorial(cards_left) // (
            factorial(num_board) * factorial(cards_left - num_board)
        )
        cards_left -= num_board

    return num_information_sets * hand_combinations * board_combinations

leduc_gamedef = {
    'numRounds': 2,
    'numSuits': 2,
    'numRanks': 3, 
    'numBoardCards': [0, 1],  # One board card (after the first round)
    'maxRaises': [1, 1]  # One raise allowed per round in Leduc
}

print(calculate_information_sets(leduc_gamedef))