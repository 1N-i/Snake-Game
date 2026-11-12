# 🐍 Snake AI - Deep Q-Learning with PyTorch & Pygame

An Artificial Intelligence agent trained to play the classic Snake game from scratch using **Deep Reinforcement Learning (Deep Q-Network)**, implemented in Python with **PyTorch**, **Pygame**, and **NumPy**.

## 🧠 How It Works

The agent interacts with a custom Pygame environment (`SnakeGameAI`), learning through trial and error via a reward system and Experience Replay.

### 1. State Representation (11 Boolean Inputs)
At each frame, `agent.py` extracts an 11-element binary state vector representing the agent's immediate surroundings:
* **Immediate Danger (3):** Danger straight ahead, to the right, or to the left.
* **Current Direction (4):** Moving Left, Right, Up, or Down.
* **Relative Food Location (4):** Food is to the left, right, above, or below the snake's head.

### 2. Action Space (3 Outputs)
The neural network outputs Q-values for three relative moves:
* `[1, 0, 0]` — Keep going straight
* `[0, 1, 0]` — Turn right
* `[0, 0, 1]` — Turn left

### 3. Reward System & Game Rules (`game.py`)
* **Eating Food:** `+10`
* **Collision (Wall or Self) / Loop Timeout:** `-10` (Episodes terminate if `frame_iteration > 100 * len(snake)` to prevent infinite loops).
* **Step Without Food:** `0`

### 4. Neural Network & Training (`model.py` & `agent.py`)
* **Architecture (`Linear_QNet`):** Feed-forward neural network (`11 -> 256 -> 3`) with `ReLU` activation on the hidden layer.
* **Optimizer & Loss (`QTrainer`):** `Adam` optimizer (`lr = 0.001`) with Mean Squared Error (`MSELoss`) based on the Bellman Equation ($Q_{new} = r + \gamma \cdot \max(Q(s'))$, with $\gamma = 0.9$).
* **Exploration vs. Exploitation:** Epsilon-greedy strategy (`epsilon = 80 - n_games`), transitioning from random exploration to model-driven actions over the first 80 games.
* **Memory:** Short-term memory (trained step-by-step) combined with Long-term Experience Replay (`MAX_MEMORY = 100,000`, `BATCH_SIZE = 1,000`), automatically saving the weights to `./model/model.pth` whenever a new high score is achieved.

## 📁 Project Structure

```text
Snake-Game/
├── model/              # Directory where trained model weights (model.pth) are saved
├── .gitattributes      # Git line-ending normalization configuration
├── .gitignore          # Ignored cache, virtual environment, and model weight files
├── agent.py            # Deep Q-Learning agent, state extraction, memory, and training loop
├── game.py             # AI-controlled Snake environment (640x480, 40 FPS)
├── helper.py           # Live Matplotlib training progress plotter (Score & Mean Score)
├── model.py            # PyTorch Linear_QNet architecture and QTrainer Bellman update logic
└── snake_game.py       # Standalone human-playable version of Snake (20 FPS)
```

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed, then install the required libraries:

```bash
pip install torch pygame numpy matplotlib ipython
```

### Training the AI Agent

Run the training loop to watch the agent learn in real time alongside a live Matplotlib score chart:

```bash
python agent.py
```

### Playing Manually

To play the classic human-controlled version using the arrow keys:

```bash
python snake_game.py
```

## 🙏 Credits & Acknowledgments

This project was developed by following the *"Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame"* series by **Patrick Loeber** ([Python Engineer](https://www.youtube.com/@patloeber)).

* **Tutorial Video (Part 1):** [YouTube](https://youtu.be/PJl4iabBEz0)
* **Original Creator:** [Patrick Loeber on GitHub](https://github.com/patrickloeber)
