import random
from collections import defaultdict, Counter

def empty_board():
    return "." * 9

def print_board(state):
    for i in range(0, 9, 3):
        row = state[i:i + 3].replace('.', ' ')
        print(" " + " | ".join(row))
        if i < 6:
            print("---+---+---")
    print()


def legal_moves(state):
    return [i for i, c in enumerate(state) if c == "."]


def is_winner(state, player):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  
        (0, 4, 8), (2, 4, 6)           
    ]
    for a, b, c in lines:
        if state[a] == state[b] == state[c] == player:
            return True
    return False

def is_draw(state):
    return "." not in state and not is_winner(state, 'X') and not is_winner(state, 'O')


def apply_move(state, action, player):
    state_list = list(state)
    state_list[action] = player
    return "".join(state_list)


class Menace:
    def __init__(self,
                 initial_beads_per_move=3,
                 reward_win=3,
                 reward_draw=1,
                 reward_loss=-1):
        self.matchboxes = {}  
        self.initial_beads_per_move = initial_beads_per_move
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.episode_history = []

    def _get_matchbox(self, state):
        if state not in self.matchboxes:
            moves = legal_moves(state)
            self.matchboxes[state] = Counter({
                m: self.initial_beads_per_move for m in moves
            })
        return self.matchboxes[state]

    def select_action(self, state):
        box = self._get_matchbox(state)
        if sum(box.values()) <= 0:
            moves = legal_moves(state)
            if not moves:
                return None
            for m in moves:
                box[m] = 1

        total = sum(box.values())
        r = random.uniform(0, total)
        cum = 0.0
        for action, count in box.items():
            cum += count
            if r <= cum:
                self.episode_history.append((state, action))
                return action
        action = random.choice(list(box.keys()))
        self.episode_history.append((state, action))
        return action

    def update_from_result(self, result):
        if result == 'win':
            r = self.reward_win
        elif result == 'loss':
            r = self.reward_loss
        else:
            r = self.reward_draw

        for state, action in self.episode_history:
            box = self._get_matchbox(state)
            box[action] += r
            if box[action] < 0:
                box[action] = 0

        self.episode_history = []

def random_opponent(state):
    moves = legal_moves(state)
    return random.choice(moves) if moves else None

def play_game(menace, opponent_policy=random_opponent, verbose=False):
    state = empty_board()
    current_player = 'X'  

    if verbose:
        print("New game:")
        print_board(state)

    while True:
        if current_player == 'X':
            action = menace.select_action(state)
            if action is None:
                result = 'draw'
                break
            state = apply_move(state, action, 'X')
        else:
            action = opponent_policy(state)
            if action is None:
                result = 'draw'
                break
            state = apply_move(state, action, 'O')

        if verbose:
            print(f"Player {current_player} played at {action}")
            print_board(state)

        if is_winner(state, 'X'):
            result = 'win'
            break
        if is_winner(state, 'O'):
            result = 'loss'
            break
        if is_draw(state):
            result = 'draw'
            break

        current_player = 'O' if current_player == 'X' else 'X'

    menace.update_from_result(result)
    return result

def train_menace(episodes=10000, verbose_every=1000):
    menace = Menace()
    results = {"win": 0, "loss": 0, "draw": 0}

    for i in range(1, episodes + 1):
        result = play_game(menace, random_opponent, verbose=False)
        results[result] += 1

        if i % verbose_every == 0:
            total = i
            win_rate = results["win"] / total
            draw_rate = results["draw"] / total
            loss_rate = results["loss"] / total
            print(f"After {i} games: "
                  f"W {results['win']} ({win_rate:.2f}), "
                  f"D {results['draw']} ({draw_rate:.2f}), "
                  f"L {results['loss']} ({loss_rate:.2f})")

    return menace, results

def human_vs_menace(menace):
    state = empty_board()
    current_player = 'X'  

    while True:
        print_board(state)

        if current_player == 'X':
            print("MENACE is thinking...")
            action = menace.select_action(state)
            state = apply_move(state, action, 'X')
        else:
            moves = legal_moves(state)
            print(f"Your turn. Legal moves: {moves}")
            while True:
                try:
                    action = int(input("Enter position (0-8): "))
                    if action in moves:
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Enter an integer 0-8.")
            state = apply_move(state, action, 'O')

        if is_winner(state, 'X'):
            print_board(state)
            print("MENACE wins!")
            menace.update_from_result('win')
            break
        if is_winner(state, 'O'):
            print_board(state)
            print("You win!")
            menace.update_from_result('loss')
            break
        if is_draw(state):
            print_board(state)
            print("It's a draw.")
            menace.update_from_result('draw')
            break

        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    menace, stats = train_menace(episodes=5000, verbose_every=1000)
    print("\nTraining finished. Final stats:", stats)
