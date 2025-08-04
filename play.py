

#3 Card GOPS

import random
# Nash Equilibrium Solver for 3-Card GOPS
import itertools
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class GOPS:
    def __init__(self):
        self.cards = [1, 2, 3]
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.player1_cards = self.cards.copy()
        self.player2_cards = self.cards.copy()
        self.prize_cards = self.cards.copy()
        random.shuffle(self.prize_cards)
        
        self.player1_score = 0
        self.player2_score = 0
        self.round_num = 0
        
        self.history = {
            'player1_played': [],
            'player2_played': [],
            'prizes_won': [],
            'prize_order': self.prize_cards.copy()
        }
    
    def get_current_state(self):
        """Get current game state"""
        return {
            'round': self.round_num + 1,
            'current_prize': self.prize_cards[self.round_num] if self.round_num < len(self.prize_cards) else None,
            'player1_cards': self.player1_cards.copy(),
            'player2_cards': self.player2_cards.copy(),
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'history': self.history.copy()
        }
    
    def play_round(self, player1_bid, player2_bid):
        """Play a single round"""
        if self.round_num >= len(self.prize_cards):
            return "Game is over!"
        
        current_prize = self.prize_cards[self.round_num]
        
        # Validate bids
        if player1_bid not in self.player1_cards:
            return f"Invalid bid for Player 1: {player1_bid} not in available cards {self.player1_cards}"
        if player2_bid not in self.player2_cards:
            return f"Invalid bid for Player 2: {player2_bid} not in available cards {self.player2_cards}"
        
        # Remove played cards
        self.player1_cards.remove(player1_bid)
        self.player2_cards.remove(player2_bid)
        
        # Record history
        self.history['player1_played'].append(player1_bid)
        self.history['player2_played'].append(player2_bid)
        
        # Determine winner and award points
        if player1_bid > player2_bid:
            self.player1_score += current_prize
            winner = "Player 1"
            self.history['prizes_won'].append(1)
        elif player2_bid > player1_bid:
            self.player2_score += current_prize
            winner = "Player 2"
            self.history['prizes_won'].append(2)
        else:
            # Tie - no one gets the points
            winner = "Tie"
            self.history['prizes_won'].append(0)
        
        result = {
            'round': self.round_num + 1,
            'prize': current_prize,
            'player1_bid': player1_bid,
            'player2_bid': player2_bid,
            'winner': winner,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score
        }
        
        self.round_num += 1
        return result
    
    def is_game_over(self):
        """Check if the game is over"""
        return self.round_num >= len(self.prize_cards)
    
    def get_winner(self):
        """Get the final winner"""
        if not self.is_game_over():
            return "Game not finished"
        
        if self.player1_score > self.player2_score:
            return "Player 1"
        elif self.player2_score > self.player1_score:
            return "Player 2"
        else:
            return "Tie"

def play_human_vs_human():
    """Play a human vs human game"""
    game = GOPS()
    print("=== 3-Card GOPS ===")
    print("Cards: 1, 2, 3")
    print("Prize order:", game.prize_cards)
    print()
    
    while not game.is_game_over():
        state = game.get_current_state()
        print(f"Round {state['round']}")
        print(f"Prize card: {state['current_prize']}")
        print(f"Player 1 available cards: {state['player1_cards']}")
        print(f"Player 2 available cards: {state['player2_cards']}")
        print(f"Current scores - Player 1: {state['player1_score']}, Player 2: {state['player2_score']}")
        print()
        
        # Get bids
        try:
            p1_bid = int(input("Player 1, enter your bid: "))
            p2_bid = int(input("Player 2, enter your bid: "))
        except ValueError:
            print("Please enter valid numbers!")
            continue
        
        result = game.play_round(p1_bid, p2_bid)
        
        if isinstance(result, str):  # Error message
            print(result)
            continue
        
        print(f"\nRound {result['round']} Results:")
        print(f"Prize: {result['prize']}")
        print(f"Player 1 bid: {result['player1_bid']}")
        print(f"Player 2 bid: {result['player2_bid']}")
        print(f"Winner: {result['winner']}")
        print(f"Scores - Player 1: {result['player1_score']}, Player 2: {result['player2_score']}")
        print("-" * 40)
    
    print(f"\nFinal Winner: {game.get_winner()}")
    print(f"Final Scores - Player 1: {game.player1_score}, Player 2: {game.player2_score}")

def play_human_vs_random():
    """Play human vs random computer"""
    game = GOPS()
    print("=== 3-Card GOPS vs Random Computer ===")
    print("Cards: 1, 2, 3")
    print("Prize order:", game.prize_cards)
    print()
    
    while not game.is_game_over():
        state = game.get_current_state()
        print(f"Round {state['round']}")
        print(f"Prize card: {state['current_prize']}")
        print(f"Your available cards: {state['player1_cards']}")
        print(f"Current scores - You: {state['player1_score']}, Computer: {state['player2_score']}")
        print()
        
        # Get human bid
        try:
            p1_bid = int(input("Enter your bid: "))
        except ValueError:
            print("Please enter a valid number!")
            continue
        
        # Computer plays randomly
        p2_bid = random.choice(state['player2_cards'])
        
        result = game.play_round(p1_bid, p2_bid)
        
        if isinstance(result, str):  # Error message
            print(result)
            continue
        
        print(f"\nRound {result['round']} Results:")
        print(f"Prize: {result['prize']}")
        print(f"Your bid: {result['player1_bid']}")
        print(f"Computer bid: {result['player2_bid']}")
        print(f"Winner: {result['winner']}")
        print(f"Scores - You: {result['player1_score']}, Computer: {result['player2_score']}")
        print("-" * 40)
    
    print(f"\nFinal Winner: {game.get_winner()}")
    print(f"Final Scores - You: {game.player1_score}, Computer: {game.player2_score}")

# Nash Equilibrium Solver for 3-Card GOPS
from collections import defaultdict

class GOPSSolver:
    def __init__(self):
        self.cards = [1, 2, 3]
        self.all_prize_orders = list(itertools.permutations(self.cards))
        self.game_tree = {}
        self.strategies = {}
        
    def get_information_set(self, my_cards, opp_cards_count, prize_history, current_prize):
        """
        Information set for a player: what they know at decision time
        - Their remaining cards
        - Number of opponent's remaining cards (but not which ones)
        - History of prizes and who won them
        - Current prize card
        """
        return (tuple(sorted(my_cards)), opp_cards_count, tuple(prize_history), current_prize)
    
    def generate_all_game_states(self):
        """Generate all possible game states for 3-card GOPS"""
        states = set()
        
        # For each possible prize order
        for prize_order in self.all_prize_orders:
            # Simulate all possible game paths
            self._enumerate_game_paths(
                p1_cards=[1,2,3], 
                p2_cards=[1,2,3], 
                prize_order=list(prize_order),
                round_num=0,
                prize_history=[],
                states=states
            )
        
        return states
    
    def _enumerate_game_paths(self, p1_cards, p2_cards, prize_order, round_num, prize_history, states):
        """Recursively enumerate all possible game paths"""
        if round_num >= len(prize_order):
            return
            
        current_prize = prize_order[round_num]
        
        # Add current information sets for both players
        p1_info_set = self.get_information_set(p1_cards, len(p2_cards), prize_history, current_prize)
        p2_info_set = self.get_information_set(p2_cards, len(p1_cards), prize_history, current_prize)
        
        states.add(('p1', p1_info_set))
        states.add(('p2', p2_info_set))
        
        # Try all possible bid combinations
        for p1_bid in p1_cards:
            for p2_bid in p2_cards:
                new_p1_cards = [c for c in p1_cards if c != p1_bid]
                new_p2_cards = [c for c in p2_cards if c != p2_bid]
                
                # Determine winner
                if p1_bid > p2_bid:
                    winner = 1
                elif p2_bid > p1_bid:
                    winner = 2
                else:
                    winner = 0  # tie
                
                new_history = prize_history + [(current_prize, winner)]
                
                # Recurse
                self._enumerate_game_paths(
                    new_p1_cards, new_p2_cards, prize_order, 
                    round_num + 1, new_history, states
                )
    
    def compute_payoff_matrix(self, info_set):
        """Compute expected payoff matrix for a given information set"""
        player, (my_cards, opp_cards_count, prize_history, current_prize) = info_set
        
        # Possible opponent cards given the count and history
        all_cards = set([1, 2, 3])
        used_cards = set()
        for prize, winner in prize_history:
            # We need to infer which cards were used from history
            # This is simplified - in practice you'd track this more carefully
            pass
        
        # For simplicity, assume uniform distribution over possible opponent cards
        possible_opp_cards = list(itertools.combinations(all_cards, opp_cards_count))
        
        payoffs = {}
        for my_bid in my_cards:
            payoffs[my_bid] = {}
            for opp_card_set in possible_opp_cards:
                for opp_bid in opp_card_set:
                    if my_bid > opp_bid:
                        payoff = current_prize
                    elif opp_bid > my_bid:
                        payoff = -current_prize  # opponent gets it
                    else:
                        payoff = 0  # tie
                    
                    if opp_bid not in payoffs[my_bid]:
                        payoffs[my_bid][opp_bid] = []
                    payoffs[my_bid][opp_bid].append(payoff)
        
        return payoffs
    
    def fictitious_play(self, iterations=1000):
        """Use fictitious play to approximate Nash equilibrium"""
        states = self.generate_all_game_states()
        
        # Initialize uniform strategies
        strategies = {}
        belief_counts = defaultdict(lambda: defaultdict(int))
        
        for player, info_set in states:
            my_cards, opp_cards_count, prize_history, current_prize = info_set
            strategies[(player, info_set)] = {card: 1.0/len(my_cards) for card in my_cards}
        
        print(f"Running fictitious play for {iterations} iterations...")
        print(f"Found {len(states)} information sets")
        
        for iteration in range(iterations):
            for player, info_set in states:
                my_cards, opp_cards_count, prize_history, current_prize = info_set
                
                # Compute best response against current opponent strategy
                best_responses = {}
                for my_bid in my_cards:
                    expected_payoff = 0
                    
                    # This is simplified - in practice you'd compute exact expected values
                    # against opponent's mixed strategy
                    for opp_bid in [1, 2, 3]:
                        if opp_bid in my_cards:  # Can't be the same remaining card
                            continue
                            
                        if my_bid > opp_bid:
                            payoff = current_prize
                        elif opp_bid > my_bid:
                            payoff = -current_prize
                        else:
                            payoff = 0
                        
                        # Weight by opponent's probability (simplified)
                        prob_weight = 1.0 / max(1, opp_cards_count)
                        expected_payoff += payoff * prob_weight
                    
                    best_responses[my_bid] = expected_payoff
                
                # Find best response
                best_bid = max(best_responses.keys(), key=lambda x: best_responses[x])
                belief_counts[(player, info_set)][best_bid] += 1
                
                # Update strategy (average of all past best responses)
                total_count = sum(belief_counts[(player, info_set)].values())
                for bid in my_cards:
                    count = belief_counts[(player, info_set)][bid]
                    strategies[(player, info_set)][bid] = count / total_count
        
        self.strategies = strategies
        return strategies
    
    def solve_nash_equilibrium(self):
        """Main method to solve for Nash equilibrium"""
        print("Solving 3-Card GOPS Nash Equilibrium...")
        
        # Generate all game states
        states = self.generate_all_game_states()
        print(f"Generated {len(states)} information sets")
        
        # Use fictitious play to approximate equilibrium
        strategies = self.fictitious_play(iterations=1000)
        
        print("\nNash Equilibrium Strategies:")
        print("=" * 50)
        
        # Group strategies by round for easier reading
        round_strategies = defaultdict(list)
        
        for (player, info_set), strategy in strategies.items():
            my_cards, opp_cards_count, prize_history, current_prize = info_set
            round_num = len(prize_history)
            round_strategies[round_num].append((player, info_set, strategy))
        
        for round_num in sorted(round_strategies.keys()):
            print(f"\nRound {round_num + 1} Strategies:")
            print("-" * 30)
            
            for player, info_set, strategy in round_strategies[round_num]:
                my_cards, opp_cards_count, prize_history, current_prize = info_set
                print(f"{player.upper()} - Prize: {current_prize}, Cards: {list(my_cards)}")
                for bid, prob in strategy.items():
                    print(f"  Bid {bid}: {prob:.3f}")
                print()
        
        return strategies

def play_with_equilibrium_strategy():
    """Play against the computed Nash equilibrium strategy"""
    solver = GOPSSolver()
    strategies = solver.solve_nash_equilibrium()
    
    game = GOPS()
    print("\n" + "="*50)
    print("Playing against Nash Equilibrium Computer")
    print("="*50)
    print("Prize order:", game.prize_cards)
    print()
    
    while not game.is_game_over():
        state = game.get_current_state()
        print(f"Round {state['round']}")
        print(f"Prize card: {state['current_prize']}")
        print(f"Your available cards: {state['player1_cards']}")
        print(f"Current scores - You: {state['player1_score']}, Computer: {state['player2_score']}")
        print()
        
        # Get human bid
        try:
            p1_bid = int(input("Enter your bid: "))
        except ValueError:
            print("Please enter a valid number!")
            continue
        
        # Computer plays using Nash equilibrium strategy
        comp_cards = state['player2_cards']
        prize_history = [(game.prize_cards[i], game.history['prizes_won'][i]) 
                        for i in range(len(game.history['prizes_won']))]
        
        comp_info_set = solver.get_information_set(
            comp_cards, len(state['player1_cards']), 
            prize_history, state['current_prize']
        )
        
        # Find strategy for this information set
        strategy_key = ('p2', comp_info_set)
        if strategy_key in strategies:
            strategy = strategies[strategy_key]
            # Sample from the mixed strategy
            cards = list(strategy.keys())
            probs = list(strategy.values())
            
            # Sample without numpy
            rand_val = random.random()
            cumsum = 0
            for i, prob in enumerate(probs):
                cumsum += prob
                if rand_val <= cumsum:
                    p2_bid = cards[i]
                    break
            else:
                p2_bid = cards[-1]  # fallback
        else:
            # Fallback to random if state not found
            p2_bid = random.choice(comp_cards)
            print("(Computer using fallback random strategy)")
        
        result = game.play_round(p1_bid, p2_bid)
        
        if isinstance(result, str):  # Error message
            print(result)
            continue
        
        print(f"\nRound {result['round']} Results:")
        print(f"Prize: {result['prize']}")
        print(f"Your bid: {result['player1_bid']}")
        print(f"Computer bid: {result['player2_bid']}")
        print(f"Winner: {result['winner']}")
        print(f"Scores - You: {result['player1_score']}, Computer: {result['player2_score']}")
        print("-" * 40)
    
    print(f"\nFinal Winner: {game.get_winner()}")
    print(f"Final Scores - You: {game.player1_score}, Computer: {game.player2_score}")

# Strategy framework for later solver development
# Strategy is a dictionary that maps each state to a mixed strategy.
# A state contains the history of cards played by you, the opponent, the point cards, and the current point card.

def state_to_key(player_cards, opponent_cards, prize_history, current_prize):
    """Convert game state to a hashable key for strategy lookup"""
    return (tuple(sorted(player_cards)), 
            tuple(sorted(opponent_cards)), 
            tuple(prize_history), 
            current_prize)

example_state = [[1],[2],[3],2]

# Example of a uniform strategy (for future solver):
def uniform_strategy(available_cards):
    """Return uniform probabilities over available cards"""
    n = len(available_cards)
    return {card: 1/n for card in available_cards}

if __name__ == "__main__":
    print("Choose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Random Computer")
    print("3. Solve Nash Equilibrium")
    print("4. Human vs Nash Equilibrium Computer")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        play_human_vs_human()
    elif choice == "2":
        play_human_vs_random()
    elif choice == "3":
        solver = GOPSSolver()
        solver.solve_nash_equilibrium()
    elif choice == "4":
        if not NUMPY_AVAILABLE:
            print("Note: Running without NumPy (install with: pip install numpy for better performance)")
        play_with_equilibrium_strategy()
    else:
        print("Invalid choice!")
        print("Running Human vs Random Computer by default...")
        play_human_vs_random()