import argparse
from lstm_model import cross_validate_and_save_models, train_on_entire_dataset

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--mode', choices=['cross-validate', 'train-full'], required=True,
                        help='Mode to run: cross-validation or train on full dataset')
    args = parser.parse_args()

    if args.mode == 'cross-validate':
        cross_validate_and_save_models(
            csv_file='../../data/preprocessed/train_final.csv',
            input_size=34,
            hidden_size=64,
            num_layers=2,
            num_classes=10,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            features=[
                'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                'appCat.finance', 'appCat.game', 'appCat.office',
                'appCat.social', 'appCat.travel', 'appCat.utilities',
                'appCat.weather', 'call', 'circumplex.arousal_lag_1', 'circumplex.arousal_lag_2',
                'circumplex.arousal_lag_3', 'circumplex.arousal_lag_4', 'circumplex.arousal_lag_5',
                'circumplex.valence_lag_1', 'circumplex.valence_lag_2', 'circumplex.valence_lag_3',
                'circumplex.valence_lag_4', 'circumplex.valence_lag_5', 'screen', 'sms', 'hour',
                'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos',
                'activity_lag_1', 'activity_lag_2', 'activity_lag_3', 'activity_lag_4', 'activity_lag_5'
            ],
            save_path='../../data/models/'
        )
    elif args.mode == 'train-full':
        train_on_entire_dataset(
            csv_file='../../data/train_set.csv',
            input_size=18,
            hidden_size=64,
            num_layers=2,
            num_classes=10,
            num_epochs=100,
            batch_size=32,
            learning_rate=0.001
        )

if __name__ == "__main__":
    main()
