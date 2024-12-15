import os
import generate_sample_dataset

def main(args):
    print(os.getcwd())
    if not os.path.isdir('datasets'):
        os.makedirs('datasets')
    generate_sample_dataset.generate_icon_sample_dataset(fname_out="./datasets/sample_icon_1day.nc")

if __name__ == "__main__":
    main(0)