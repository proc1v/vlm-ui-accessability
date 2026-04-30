import tqdm
import torch
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda

# no test yet
def validate(val_loader, model_engine, processor, epoch, global_step, writer, args):
    model_engine.eval()

    metric = 0
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        with torch.no_grad():
            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                output_hidden_states=True,
            )
            if "image_sizes" in input_dict:
                forward_dict["image_grid_thw"] = input_dict["image_sizes"]
            for key in ["patch_assign", "patch_assign_len", "patch_pos", "select_mask"]:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
            output_dict = model_engine(**forward_dict)
            metric += output_dict['loss'].item()
    # a navie way to calculate the metric by their loss
    return 1 / metric