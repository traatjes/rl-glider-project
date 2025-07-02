# Autonomous Glider Reinforcement Learning  glider

This project is an implementation of a reinforcement learning agent that learns to pilot a glider, using thermal updrafts to maximize its travel distance. The agent operates in a custom 2D environment and learns its policy using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3.

This was created for the course AE4350 at TU Delft.

![GIF of a trained agent's flight path]
*(TODO: Add a GIF or image here of a successful flight path once your agent is trained!)*

---

## Features

* **Custom 2D Glider Environment:** A Gymnasium-compatible environment where a glider's altitude is affected by Gaussian thermal updrafts.
* **Reinforcement Learning Agent:** An agent trained with Stable Baselines3 to navigate the environment.
* **Configurable Parameters:** Easily tweak simulation and training parameters in `src/config.py`.
* **Result Visualization:** Automatically saves plots of flight paths and training rewards.

---

## Installation

To set up and run this project locally, follow these steps.

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)[YourUsername]/[Your-Repo-Name].git
cd [Your-Repo-Name]
```

**2. Create and activate a virtual environment:**

* **Windows:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
* **macOS / Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

To train a new agent, run the main training script:

```bash
python src/train.py
```

Trained models, logs, and plots will be saved in the `results/` directory.

---

## Project Structure

The project is organized as follows:

```
├── src/
│   ├── glider_env.py     # The custom Gymnasium environment
│   ├── agent.py          # Agent definition (if separated from training)
│   ├── train.py          # Main script to run training
│   ├── config.py         # All project parameters
│   └── utils.py          # Plotting functions and other helpers
├── results/              # Output folder for models, plots, and logs
├── .gitignore            # Files to be ignored by Git
├── requirements.txt      # Project dependencies
└── README.md             # You are here!
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
*(TODO: Create a file named `LICENSE` and add the MIT license text to it if you wish.)*