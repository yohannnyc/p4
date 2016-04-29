Team: Swingy Monkeys 

## How to Run ##

$> python stub.py

## Changing Parameters ##

We augmented the Learner class. It takes in the following parameters:

1. gamma - The discount factor
2. epsilon - The greedy parameter
3. alpha - The learning rate
4. bin_width - The width of discrete bins
5. ql - If True: QLearn else SARSA
6. specialinitialization - If True: Special initial policy else: random

All these parameters have to be specified when the learner is initialized as follows: 

(stub.py @ line 201)
agent = Learner(gamma=1,alpha=0.2,epsilon=1,bin_width=50,specialinitialization=True,ql=True)

# Enjoy and play the game

########################################################################

        .-"-.            .-"-.            .-"-.           .-"-.
     _/_-.-_\_        _/.-.-.\_        _/.-.-.\_       _/.-.-.\_
    / __} {__ \      /|( o o )|\      ( ( o o ) )     ( ( o o ) )
   / //  "  \\ \    | //  "  \\ |      |/  "  \|       |/  "  \|
  / / \'---'/ \ \  / / \'---'/ \ \      \'/^\'/         \ .-. /
  \ \_/`"""`\_/ /  \ \_/`"""`\_/ /      /`\ /`\         /`"""`\
   \           /    \           /      /  /|\  \       /       \

-={ see no evil }={ hear no evil }={ speak no evil }={ have no fun }=-
########################################################################