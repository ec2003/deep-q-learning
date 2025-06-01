from modules.game import TaxiDQL
import argparse

def main():

    parser = argparse.ArgumentParser(description='This is a program to train and/or test DQN model for the game "Taxi-v3"')
    parser.add_argument('--train', dest='train', help='Train the model or not (True/False)', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', dest='test', help='Test the model or not (True/False)', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train_episodes', dest='train_episodes', help='The number of episodes for training', default=10000, type=int)
    parser.add_argument('--test_episodes', dest='test_episodes', help='The number of episodes for testing', default=10, type=int)
    args = parser.parse_args()

    taxi_dql = TaxiDQL()
    if args.train: taxi_dql.train(args.train_episodes)
    try: 
        if args.test: taxi_dql.test(args.test_episodes)
    except FileNotFoundError:
        print('''The model for testing does not exist. 
Check if the model is in the folder or if you have trained and saved the model.             
              ''')

    return None

if __name__ == '__main__':    
    main()