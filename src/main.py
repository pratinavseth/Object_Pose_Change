import argparse
from .utils.inference import inference_localization

def main(args):
    config = {
        'img_path': args.img_path,
        'output_dir': args.output_dir,
        'bb_model': args.bb_model,
        'SAM_model': args.SAM_model,
        'Text_Prompt': args.Prompt

    }
    inference_localization(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=False, default='')
    parser.add_argument('--output_dir', type=str, required=False, default='')
    parser.add_argument('--bb_model', type=str, required=False, default='owl_vit')
    parser.add_argument('--SAM_model', type=str, required=False, default='vit_h')
    parser.add_argument('--Prompt', type=str, required=False, default='prompt')

    arguments = parser.parse_args()
    main(arguments)