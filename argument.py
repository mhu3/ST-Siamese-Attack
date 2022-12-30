import argparse


def create_parser():
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser(description='ST-SiameseFormer')

    # Model Input
    # trajectory type
    parser.add_argument('--traj_type', type=str, default='all', 
                        help='input type, seek, serve or all')
    # speed
    parser.add_argument('--with_speed', action='store_true', 
                        help='input trajs with speed')
    parser.add_argument('--without_speed', dest='with_speed', action='store_false', 
                        help='input trajs without speed')
    parser.set_defaults(with_speed=False)
    # profile data
    parser.add_argument('--with_profile', action='store_true', 
                        help='input trajs with profile features')
    parser.add_argument('--without_profile', dest='with_profile', action='store_false', 
                        help='input trajs without profile features')
    parser.set_defaults(with_profile=False)
    
    # Training data to use
    parser.add_argument('--num_plates', type=int, default=500,
                        help='number of training plates (default: 500)')
    parser.add_argument('--num_days', type=int, default=8,
                        help='number of training days (default: 8)')
    parser.add_argument('--num_trajs', type=int, default=10,
                        help='number of trajectories per plate per day (default: 10)')
    parser.add_argument('--padding_length', type=int, default=60,
                        help='padding length of trajectories (default: 60)')

    # Hyper-parameters
    parser.add_argument('--batch_size', type=int, default = 32,
                        help='number of training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default = 30,
                        help='number of iterations (default: 30)')

    # Directories
    # data directory
    parser.add_argument('--data_path', type=str, default='./dataset/')
    # model directory
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--log_path', type=str, default='./logs/')
    
    return parser
