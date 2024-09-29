import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_batch', type=int, default=64, help='Batch size for training') # 64
    parser.add_argument('--test_batch', type=int, default=100, help='Batch size for testing')  # 100
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    parser.add_argument('--bw', type=int, default=64, help='Bandwidth') 
    parser.add_argument('--n_e', type=int, default=16777216, help='Number of embeddings') 
    parser.add_argument('--snr_db', type=int, default=10, help='SNR_dB') 
    # 50 75 175 325 600git remote add origin https://github.com/your-username/your-repo-name.git

    # training setting
    parser.add_argument('-epoch', type=int, default = 80)
    parser.add_argument('-lr', type=float, default = 1e-4)
    parser.add_argument('--lr_step_size', default = 20)
    parser.add_argument('--lr_gamma', default = 0.1)
    parser.add_argument('--weight_decay', default = 5e-04)
    parser.add_argument('--beta', default = 0.25)

    parser.add_argument('-optimizer', default = "Adam")

    # quantize
    parser.add_argument('-num_bits', default=15)

    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    # clipcaption
    parser.add_argument('-prefix_size', default=768)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)

    # transformer
    parser.add_argument('--bw_old', type=int, default=192, help='Transformer Bandwidth') 
    parser.add_argument('--levels', type=int, default=2, help='scalar quantization levels')

    args = parser.parse_args()

    return args