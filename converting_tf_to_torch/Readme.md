# Converging Tensorflow model to Pytorch model

This small code converts a checkpoint from the original implementation of the PhysNet to a Pytorch model usable with
asparagus. 

To use it, you need to have tensorflow and pytorch installed. Then you can run the following command:

The file `parameters_convertion.npy` contains the parameters of the original model in TF1, and it's equivalent in asparagus. 
It should be mentioned that not all parameters of asparagus are available in the original model, so some of them are
to the default value. You might need to adjust them to your needs.

To run the code you need to have the following files in the same folder: 
`*.data-00000-of-00001` and `*.index` (the checkpoint files from the original model)

Then you execute the code with the following line:

    ```
    python convert_tf_to_torch.py -cpt <path_to_checkpoint> --o <output_name>.pt
    ```

Questions or problems? Please contact me at <luisitza.vazquezsalazar@unibas.ch>
