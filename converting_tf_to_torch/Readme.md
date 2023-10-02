# Converging Tensorflow model to Pytorch model

This small code converts a checkpoint from the original implementation of PhysNet in Tensorflow1 to a Pytorch model usable with
asparagus. 

To use it, you need to have tensorflow and pytorch installed. Then you can run the following command:
   ```
    python convert_tf_to_torch.py -cpt <path_to_checkpoint> --o <output_name>.pt
   ```

## Notes on the conversion
The file `parameters_convertion.npy` contains the parameters of the original model in Tensorflow1, and it's equivalent in asparagus. 
It should be mentioned that not all parameters of asparagus are available in the original model, so some of them are
to the default value. You might need to adjust them to your needs.

The file `parameters_convertion.npy` can be also use for the inverse procedure. However, you need to convert the 
torch tensors in the asparagus checkpoint to numpy arrays to be used by tensorflow 1. This is not implement (and most
likely won't be implemented) but you can easily extrapolate from the script here. 

To run the code you need to have the following files in the same folder: 
`*.data-00000-of-00001` and `*.index` (the checkpoint files from the original model)

## Contact
Questions or problems? Send me a mail to <luisitza.vazquezsalazar@unibas.ch>
